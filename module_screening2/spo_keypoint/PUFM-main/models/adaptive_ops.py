"""Portable B,N,C point operations; no ground truth or runtime extension builds."""
import torch
from adaptive_resampling.v3_transport import assignment_indices
from adaptive_resampling.v3_pointops import fps_indices


def gather(points, ids):
    batch = torch.arange(len(points), device=points.device)
    return points[batch.reshape((-1,) + (1,) * (ids.ndim - 1)), ids]


@torch.no_grad()
def fps_ids(points, count):
    if not 0 < count <= points.shape[1]:
        raise ValueError(f"FPS count {count} is outside 1..{points.shape[1]}")
    return fps_indices(points, count)


def fps(points, count):
    return gather(points, fps_ids(points, count))


def knn(points, query, count, chunk=512):
    if not 0 < count <= points.shape[1]:
        raise ValueError("Invalid neighbor count")
    with torch.no_grad():
        return torch.cat([torch.cdist(q, points).topk(count, largest=False, sorted=True).indices
                          for q in query.split(chunk, 1)], 1)


def midpoint(points, rate=4):
    neighbors = gather(points, knn(points, points, min(2 * rate, points.shape[1])))
    pool = ((points[:, :, None] + neighbors) * .5).flatten(1, 2)
    return fps(pool, points.shape[1] * rate)


def distance_vectors(prediction, target):
    # Chunks bound memory for 8192-point object evaluation and training losses.
    a = torch.cat([torch.cdist(p, target).min(-1).values for p in prediction.split(512, 1)], 1)
    b = torch.cat([torch.cdist(g, prediction).min(-1).values for g in target.split(512, 1)], 1)
    return a, b


def cd_l1(prediction, target):
    a, b = distance_vectors(prediction, target)
    return .5 * (a.mean(-1) + b.mean(-1))


def aligned_target(source, target):
    """Exact matching backend. Only call inside training / teacher construction."""
    return gather(target, assignment_indices(source, target))


def region_counts(points, centers):
    ids = knn(centers, points, 1).squeeze(-1)
    count = points.new_zeros(len(points), centers.shape[1])
    count.scatter_add_(1, ids, torch.ones_like(ids, dtype=points.dtype))
    return count


def assign_regions(points, centers):
    return knn(centers, points, 1).squeeze(-1)
