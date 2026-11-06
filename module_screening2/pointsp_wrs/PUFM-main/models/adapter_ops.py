"""Observation-only geometry shared by value, physically copied per experiment."""
import hashlib
import math
import torch
from torch import nn
from .adaptive_ops import gather, knn, fps_ids
from .utils import get_knn_pts, FPS



def median_value(values, keepdim=False):
    # CUDA median-with-indices is nondeterministic even if only values are used.
    ordered = values.sort(dim=-1, stable=True).values
    result = ordered.select(-1, (values.shape[-1] - 1) // 2)
    return result.unsqueeze(-1) if keepdim else result


def midpoint_pool(points):
    # Preserve the original PUFM neighbor routine and candidate ordering exactly.
    xyz = points.transpose(1, 2).contiguous()
    neighbors = get_knn_pts(8, xyz, xyz)
    return ((neighbors + xyz[..., None]) * .5).flatten(2).transpose(1, 2)


def select_fps(points, count):
    return FPS(points.transpose(1, 2).contiguous(), count).transpose(1, 2)


def local_geometry(points, k=20):
    ids = knn(points, points, min(k, points.shape[1]))
    neighbors = gather(points, ids)
    delta = neighbors - points[:, :, None]
    distance = delta.square().sum(-1).clamp_min(0)
    return ids, neighbors, delta, distance


@torch.no_grad()
def pointsp_weight(points, k=20):
    # PointSP cal_weight: count neighbors below the median local mean d^2.
    ids, _, _, distance = local_geometry(points, k)
    threshold = torch.quantile(distance.mean(-1), .5, dim=-1)
    weights = (distance <= threshold[:, None, None]).sum(-1).float()
    return weights / weights.sum(-1, keepdim=True).clamp_min(1), ids


@torch.no_grad()
def geometry(points, k=20):
    ids, neighbors, delta, distance = local_geometry(points, k)
    centered = neighbors - neighbors.mean(2, keepdim=True)
    covariance = centered.transpose(-1, -2) @ centered / neighbors.shape[2]
    values, vectors = torch.linalg.eigh(covariance.float())
    normal = vectors[..., 0]
    spacing = median_value(distance[..., 1:].sqrt()).clamp_min(1e-5)
    return ids, distance, normal, spacing


def generator(points, seed, salt):
    # Independent CPU RNG; never shifts the common FM time/noise stream.
    digest = hashlib.sha256(points.detach().float().cpu().contiguous().numpy().tobytes())
    digest.update(f'{seed}:{salt}'.encode())
    return torch.Generator().manual_seed(int.from_bytes(digest.digest()[:8], 'little') % (2**63 - 1))


def weighted_ids(points, weights, count, seed, salt='sampling'):
    if weights.shape != points.shape[:2] or not 0 < count <= points.shape[1]:
        raise ValueError('Sampling requires B,N weights and 0 < count <= N')
    result = []
    for cloud, weight in zip(points, weights):
        weight = weight.detach().float().cpu().clamp_min(1e-8)
        if not torch.isfinite(weight).all():
            raise FloatingPointError('Invalid sampling weights')
        result.append(torch.multinomial(weight, count, replacement=False,
                                       generator=generator(cloud, seed, salt)))
    return torch.stack(result).to(points.device)


def grid(count, device, dtype):
    # A fixed square grid supports both patch and whole-object output budgets.
    side = max(2, math.ceil(math.sqrt(count)))
    axis = torch.linspace(-1, 1, side, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2)[:count]


class PointEncoder(nn.Module):
    """Permutation-equivariant local encoding; no BatchNorm/teacher buffer drift."""
    def __init__(self, width=64, k=16):
        super().__init__()
        self.k = k
        self.local = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.GELU())

    def forward(self, points):
        ids = knn(points, points, min(self.k, points.shape[1]))
        delta = gather(points, ids) - points[:, :, None]
        center = points[:, :, None].expand_as(delta)
        local = self.local(torch.cat([center, delta], -1)).max(2).values
        return self.fuse(torch.cat([local, local.mean(1, keepdim=True).expand_as(local)], -1))


def anchor_loss(anchors, batch):
    """Train-only precision plus missing-region-weighted GT coverage."""
    target = batch['gt_xyz']
    distance = torch.cdist(anchors, target).square()
    precision = distance.min(-1).values.mean()
    labels = batch.get('supervision', {})
    mask = labels.get('hole_mask_gt', torch.zeros_like(target[..., 0])).bool()
    mask = mask | labels.get('sparse_mask_gt', torch.zeros_like(target[..., 0])).bool()
    # Empty missing regions use all GT, rather than a zero/NaN auxiliary loss.
    mask = torch.where(mask.any(-1, keepdim=True), mask, torch.ones_like(mask))
    coverage = (distance.min(1).values * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return precision + coverage.mean()
