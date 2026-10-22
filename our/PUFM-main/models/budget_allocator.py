"""Unique budget ownership, bounded integer allocation, and nested candidate pools."""
import torch
from .adaptive_ops import assign_regions, fps_ids, gather, knn, region_counts


FEATURE_DIM = 76  # 64 learned recovery + 8 observable geometry + q,h,r,u


def build_regions(points, geometry, rgr, count):
    center_ids = fps_ids(geometry["xyz"].detach(), count)
    centers = gather(geometry["xyz"], center_ids)
    ownership = assign_regions(geometry["xyz"], centers)
    ids = knn(geometry["xyz"], centers, min(24, points.shape[1]))
    neighbors = gather(geometry["xyz"], ids)
    relative = neighbors - centers[:, :, None]
    distances = relative.norm(dim=-1)
    radius = distances.mean(-1, keepdim=True).clamp_min(1e-4)
    covariance = relative.transpose(-1, -2) @ relative / relative.shape[2]
    eigen = torch.linalg.eigvalsh(covariance).clamp_min(0)
    anchor_counts = region_counts(geometry["xyz"], centers)
    observable = torch.cat([eigen / eigen.sum(-1, keepdim=True).clamp_min(1e-8),
                            radius, distances.amax(-1, keepdim=True),
                            radius / radius.mean(1, keepdim=True),
                            anchor_counts[..., None] / points.shape[1],
                            distances.std(-1, keepdim=True, unbiased=False)], -1)
    if rgr is None:
        region = {"features": points.new_zeros(len(points), count, 64), "q": points.new_ones(len(points), count, 1),
                  "h": points.new_zeros(len(points), count, 1), "r": points.new_zeros(len(points), count, 1),
                  "u": points.new_zeros(len(points), count, 1), "virtual": centers, "radius": radius}
    else:
        region = rgr.recover_regions(geometry, centers)
    features = torch.cat([region["features"], observable, region["q"], region["h"], region["r"], region["u"]], -1)
    return dict(region, centers=centers, ownership=ownership, center_ids=center_ids,
                anchor_counts=anchor_counts, observable=observable, gain_features=features)


@torch.no_grad()
def quota(probability, total, minimum, capacity):
    b, k = probability.shape
    if k * minimum > total or k * capacity < total:
        raise ValueError("Infeasible region lower/upper bounds for requested point budget")
    result = torch.full((b, k), minimum, device=probability.device, dtype=torch.long)
    remaining = total - k * minimum
    probability = probability.clamp_min(1e-12)
    for batch in range(b):
        while remaining and int(result[batch].sum()) < total:
            available = result[batch] < capacity
            p = probability[batch] * available
            p = p / p.sum()
            missing = total - int(result[batch].sum())
            expected = p * missing
            additions = torch.minimum(expected.floor().long(), capacity - result[batch])
            result[batch] += additions
            missing = total - int(result[batch].sum())
            if missing:
                residual = expected - additions
                residual[result[batch] >= capacity] = -float("inf")
                eligible = min(missing, int((result[batch] < capacity).sum()))
                result[batch, residual.topk(eligible).indices] += 1
    return result


@torch.no_grad()
def allocate(regions, gain, total, minimum=4, capacity=64, step=8, iterations=32, method="gain"):
    observable = regions["observable"]
    # Bounded spatial coverage proxy, independent of GT and target labels.
    base = observable[..., 3].square().clamp_min(1e-6)
    base = base.clamp(max=4 * base.median(-1, keepdim=True).values.clamp_min(1e-6))
    if method == "uniform":
        base = torch.ones_like(base)
    elif method == "density":
        base = observable[..., 5].clamp(.25, 4)
    elif method == "curvature":
        base = .1 + observable[..., 0]
    elif method == "static":
        base = .7 * base / base.sum(-1, keepdim=True) + .3 * gain.scores(regions["gain_features"]).softmax(-1)
    budget = quota(base, total, minimum, capacity)
    if method != "gain":
        return budget
    for _ in range(iterations):
        changed = False
        additions = gain.batch_gain(regions["gain_features"], budget, step)
        costs = gain.batch_gain(regions["gain_features"], (budget - step).clamp_min(0), step)
        for batch in range(len(budget)):
            value = additions[batch, :, None] - costs[batch, None, :]
            value[budget[batch] + step > capacity] = -float("inf")
            value[:, budget[batch] - step < minimum] = -float("inf")
            value.fill_diagonal_(-float("inf"))
            best = value.argmax()
            receiver, donor = int(best // len(value)), int(best % len(value))
            if value[receiver, donor] > 0:
                budget[batch, receiver] += step
                budget[batch, donor] -= step
                changed = True
        if not changed:
            break
    if torch.any(budget.sum(-1) != total):
        raise AssertionError("Integer allocation changed the total point budget")
    return budget


def candidate_pool(geometry, regions, capacity=64, virtual=True):
    """One immutable ordered pool per region, reused across all budget swaps."""
    xyz, centers = geometry["xyz"], regions["centers"]
    ids = knn(xyz, centers, min(16, xyz.shape[1]))
    local = gather(xyz, ids)
    q = gather(geometry["q"], ids)
    # All center/local pair midpoints are finite; reliability downweights noisy displacements.
    local = centers[:, :, None] + q * (local - centers[:, :, None])
    middle = ((local[:, :, :, None] + local[:, :, None, :]) * .5).flatten(2, 3)
    b, k, size, _ = middle.shape
    if capacity > size:
        raise ValueError("Candidate capacity exceeds the local midpoint pool")
    order = fps_ids(middle.detach().reshape(b * k, size, 3), capacity).reshape(b, k, capacity)
    batch = torch.arange(b, device=xyz.device)[:, None, None]
    region_ids = torch.arange(k, device=xyz.device)[None, :, None]
    pool = middle[batch, region_ids, order]
    if virtual:
        gate = regions["h"] * regions["r"] * (1 - regions["u"])
        displacement = gate * (regions["virtual"] - centers)
        # Interleave ordinary and recovery candidates; quotas select nested prefixes.
        mask = (torch.arange(capacity, device=xyz.device) % 2).to(xyz.dtype)[None, None, :, None]
        pool = pool + mask * displacement[:, :, None]
    return pool


def initialize(anchors, pools, budget):
    clouds, owners = [], []
    for batch in range(len(anchors)):
        pieces = [anchors[batch]]
        owner = [torch.full((anchors.shape[1],), -1, device=anchors.device, dtype=torch.long)]
        for region in range(budget.shape[1]):
            count = int(budget[batch, region])
            pieces.append(pools[batch, region, :count])
            owner.append(torch.full((count,), region, device=anchors.device, dtype=torch.long))
        clouds.append(torch.cat(pieces))
        owners.append(torch.cat(owner))
    return torch.stack(clouds), torch.stack(owners)
