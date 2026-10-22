"""Training-only supervision; all targets derive from the existing V3 fields."""
import torch
from torch.nn import functional as F
from .adaptive_ops import aligned_target, assign_regions, distance_vectors, gather


def flow_loss(model, state, target):
    x0 = state["initial"] + .01 * torch.randn_like(state["initial"])
    x1 = aligned_target(x0, target)
    time = 1 - torch.cos(torch.rand(len(x0), device=x0.device) * torch.pi / 2)
    query = time[:, None, None] * x1 + (1 - time[:, None, None]) * x0
    velocity = model.velocity(query, time, state)
    return F.mse_loss(velocity, x1 - x0)


def recovery_losses(model, state, gt, supervision):
    if not model.config.use_rgr:
        return {"rgr": gt.new_zeros(())}
    geometry, region = state["geometry"], state["regions"]
    q_target = supervision["reliability_target_input"].float()[..., None]
    reliability = F.binary_cross_entropy_with_logits(geometry["q_logit"], q_target)
    if not model.config.use_reliability:
        reliability = reliability * 0
    spacing = geometry["spacing"].detach().clamp_min(1e-4)
    offset = F.smooth_l1_loss(geometry["offset"] / spacing, supervision["denoise_offset"].float() / spacing)
    if not model.config.use_recovery:
        return {"rgr": reliability + offset, "reliability": reliability, "denoise": offset}
    centers = region["centers"].detach()
    with torch.no_grad():
        assignment = assign_regions(gt, centers)
        k = centers.shape[1]
        b = len(gt)
        total = gt.new_zeros(b, k).scatter_add_(1, assignment, torch.ones_like(assignment, dtype=gt.dtype))
        hole = supervision["hole_mask_gt"].float()
        missing = gt.new_zeros(b, k).scatter_add_(1, assignment, hole)
        h_target = missing / total.clamp_min(1)
        target_center = gt.new_zeros(b, k, 3)
        target_center.scatter_add_(1, assignment[..., None].expand(-1, -1, 3), gt * hole[..., None])
        target_center /= missing[..., None].clamp_min(1)
        # Project hole centroids back onto an actual labelled GT point.
        distance = torch.cdist(target_center, gt)
        distance = distance.masked_fill(hole[:, None] == 0, float("inf"))
        target_virtual = gather(gt, distance.argmin(-1))
        clean = supervision["clean_input_xyz"].float()
        support = torch.cdist(target_virtual, clean).min(-1).values
        r_target = torch.exp(-support / (2 * region["radius"].squeeze(-1).detach()).clamp_min(1e-4))
        present = missing > 0
        # This is a training-time geometric-risk proxy, not a true uncertainty label.
        surface_error = torch.cdist(region["virtual"].detach(), gt).min(-1).values
        u_target = (surface_error / region["radius"].squeeze(-1).detach().clamp_min(1e-4)).clamp(0, 1)
    hole_loss = F.binary_cross_entropy_with_logits(region["logits"][..., 0], h_target)
    risk = F.mse_loss(region["u"].squeeze(-1), u_target)
    if not model.config.use_uncertainty:
        risk = risk * 0
    if present.any():
        recoverability = F.binary_cross_entropy_with_logits(region["logits"][..., 1][present], r_target[present])
        query = F.smooth_l1_loss((region["virtual"] / region["radius"].detach())[present],
                               (target_virtual / region["radius"].detach())[present])
    else:
        recoverability = region["r"].sum() * 0
        query = region["virtual"].sum() * 0
    total_loss = reliability + offset + hole_loss + .25 * recoverability + .25 * query + .1 * risk
    return {"rgr": total_loss, "reliability": reliability, "denoise": offset, "hole": hole_loss,
            "recoverability_proxy": recoverability, "virtual_query": query, "risk_proxy": risk}


def retention_loss(prediction, state, sigma=.1):
    if state["budget"] is None:
        return prediction.new_zeros(())
    centers = state["regions"]["centers"].detach()
    probability = (-torch.cdist(prediction, centers).square() / sigma ** 2).softmax(-1)
    counts = probability.sum(1)
    target = (state["budget"] + state["regions"]["anchor_counts"]).detach()
    return ((counts - target) / target.clamp_min(1)).square().mean()


def gain_loss(model, features, budget, receiver, donor, delta, improvement):
    predicted = model.gain.transfer(features, budget, receiver, donor, delta)
    scaled_target = improvement * model.config.teacher_gain_scale
    regression = F.smooth_l1_loss(predicted, scaled_target)
    valid = scaled_target.abs() > 1e-5
    ranking = F.softplus(-scaled_target[valid].sign() * predicted[valid]).mean() if valid.any() else predicted.sum() * 0
    return regression + .05 * ranking, predicted
