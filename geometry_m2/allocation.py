"""Spatially unique midpoint candidates and capacity-aware integer exchanges."""
import torch
from models.adaptive_ops import assign_regions, fps_ids, gather, knn


@torch.no_grad()
def candidate_pools(points, region, capacity=64):
    centers = region["centers"]
    # Local unordered pairs avoid duplicate (a,b)/(b,a) candidates. Ownership is
    # assigned once globally, including when overlapping neighbourhoods share pairs.
    ids = knn(points, centers, min(32, points.shape[1]))
    pairs = torch.triu_indices(ids.shape[-1], ids.shape[-1], offset=1, device=points.device)
    pools = points.new_zeros(len(points), centers.shape[1], capacity, 3)
    capacities = torch.zeros(centers.shape[:2], dtype=torch.long, device=points.device)
    for batch in range(len(points)):
        endpoints = torch.stack((ids[batch, :, pairs[0]], ids[batch, :, pairs[1]]), -1)
        endpoints = torch.unique(endpoints.reshape(-1, 2).sort(-1).values, dim=0)
        candidates = points[batch, endpoints].mean(1)
        candidates = torch.unique(candidates, dim=0)
        # Input anchors already occupy these sites.
        candidates = candidates[torch.cdist(candidates[None], points[batch:batch+1])[0].amin(-1) > 1e-7]
        owner = assign_regions(candidates[None], centers[batch:batch+1])[0]
        for index in range(centers.shape[1]):
            local = candidates[owner == index]
            count = min(capacity, len(local))
            capacities[batch, index] = count
            if count:
                pools[batch, index, :count] = gather(local[None], fps_ids(local[None], count))[0]
    return pools, capacities


@torch.no_grad()
def quota(weights, total, minimum, capacities):
    if (capacities < minimum).any() or (capacities.sum(-1) < total).any() or weights.shape[1] * minimum > total:
        raise ValueError("Insufficient unique midpoint capacity for fixed budget and lower bound")
    result = torch.full_like(capacities, minimum)
    for batch in range(len(result)):
        while int(result[batch].sum()) < total:
            available = capacities[batch] - result[batch]
            probability = weights[batch].clamp_min(1e-8) * (available > 0)
            expected = probability / probability.sum() * (total - int(result[batch].sum()))
            additions = torch.minimum(expected.floor().long(), available)
            result[batch] += additions
            missing = total - int(result[batch].sum())
            if missing:
                residual = (expected - additions).masked_fill(result[batch] >= capacities[batch], -torch.inf)
                count = min(missing, int((result[batch] < capacities[batch]).sum()))
                result[batch, residual.argsort(descending=True, stable=True)[:count]] += 1
    return result


@torch.no_grad()
def allocate(model, state, method="gain"):
    region, capacities = state["regions"], state["capacities"]
    weights = torch.ones_like(region["scale"])
    if method == "density":
        weights = region["scale"] / region["scale"].mean(-1, keepdim=True)
    elif method == "curvature":
        eigen = region["eigen"].clamp_min(0)
        weights = .1 + eigen[..., 0] / eigen.sum(-1).clamp_min(1e-8)
    elif method == "random":
        weights = torch.exp(torch.randn_like(weights) * .7)
    elif method == "rule":
        weights = torch.exp(state["scalar"].clamp(-3, 3))
    elif method not in {"gain", "uniform"}:
        raise ValueError(method)
    budget = quota(weights, state["total"], model.config.minimum, capacities)
    if method != "gain":
        return budget
    step = model.config.delta
    valid = state["valid"]
    for _ in range(model.config.exchanges):
        addition = model.m2.gain.batch_gain(state["features"], budget, step)
        cost = model.m2.gain.batch_gain(state["features"], (budget-step).clamp_min(0), step)
        scores = addition[:, :, None] - cost[:, None, :]
        legal = (budget+step <= capacities) & valid
        donors = (budget-step >= model.config.minimum) & valid
        scores.masked_fill_(~(legal[:, :, None] & donors[:, None, :]), -torch.inf)
        scores.diagonal(dim1=1, dim2=2).fill_(-torch.inf)
        value, pair = scores.flatten(1).max(-1)
        active = value > model.config.gain_threshold
        if not active.any():
            break
        batch = torch.arange(len(budget), device=budget.device)[active]
        budget[batch, pair[active] // budget.shape[1]] += step
        budget[batch, pair[active] % budget.shape[1]] -= step
    return budget


def initialize(state, budget, minimum):
    if budget.dtype != torch.long or budget.shape != state["capacities"].shape:
        raise ValueError("Expected B x K integer budgets")
    if (budget < minimum).any() or (budget > state["capacities"]).any() or (budget.sum(-1) != state["total"]).any():
        raise ValueError("Illegal budget or insufficient candidate pool")
    clouds = []
    for batch in range(len(budget)):
        clouds.append(torch.cat([state["input"][batch]] +
                                [state["pools"][batch, index, :int(count)] for index, count in enumerate(budget[batch])]))
    return dict(state, budget=budget, initial=torch.stack(clouds))
