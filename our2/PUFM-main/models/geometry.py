"""Observation-only geometry. B,N,C layout; unique Voronoi accounting."""
import math
import torch
from .adaptive_ops import gather, knn, fps_ids


def pool(values, labels, count):
    # Dense membership avoids CUDA scatter-add nondeterminism; K <= 64 normally.
    membership = torch.nn.functional.one_hot(labels, count).to(values.dtype)
    return membership.transpose(1, 2) @ values / membership.sum(1).clamp_min(1)[..., None]


@torch.no_grad()
def geometry(points, divisor=32):
    k = math.ceil(points.shape[1] / divisor)
    anchors = gather(points, fps_ids(points, k))
    labels = knn(anchors, points, 1).squeeze(-1)
    ids = knn(points, points, min(32, points.shape[1]))
    rel = gather(points, ids) - points[:, :, None]
    scale = rel[..., 1:, :].norm(dim=-1).mean(-1).clamp_min(1e-6)
    columns = []
    normals = None
    for size in (16, 32):
        local = rel[:, :, :size]
        centered = local - local.mean(2, keepdim=True)
        cov = centered.transpose(-1, -2) @ centered / local.shape[2]
        eigen, vectors = torch.linalg.eigh(cov)
        eigen = eigen.clamp_min(0)
        normals = vectors[..., 0]
        columns.extend([eigen / eigen.sum(-1, keepdim=True).clamp_min(1e-12),
                        local.mean(2) / scale[..., None],
                        local.norm(dim=-1).mean(-1, keepdim=True) / scale[..., None]])
    median = scale.sort(1).values[:, (scale.shape[1]-1)//2:(scale.shape[1]-1)//2+1]
    columns.append((scale / median).log().unsqueeze(-1))
    features = torch.cat(columns, -1)  # 15 relative, dimensionless features
    region_features = pool(features, labels, k)
    region_scale = pool(scale[..., None], labels, k)
    # Candidate gap midpoints, projected onto the anchor's local tangent plane.
    a_ids = knn(points, anchors, 1).squeeze(-1)
    neighbors = gather(points, gather(ids, a_ids))
    delta = (neighbors[:, :, -1] - anchors) * .5
    normal = gather(normals, a_ids)
    delta = delta - (delta * normal).sum(-1, keepdim=True) * normal
    delta = delta * (region_scale / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1)
    candidates = anchors + delta
    return dict(points=points, anchors=anchors, labels=labels, ids=ids, scale=scale,
                features=features, region_features=region_features,
                region_scale=region_scale, candidates=candidates)
