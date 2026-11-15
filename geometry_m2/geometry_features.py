"""Input-only geometry in a shared PCA frame, with explicit validity masks."""
import torch
from models.adaptive_ops import fps_ids, gather, knn, region_counts


@torch.no_grad()
def local_regions(points, regions=16, neighbors=24):
    if points.ndim != 3 or points.shape[-1] != 3 or not torch.isfinite(points).all():
        raise ValueError("Expected finite B x N x 3 coordinates")
    if points.shape[1] < neighbors + 1:
        raise ValueError("Geometry needs at least 25 input points")
    count = max(1, round(regions * points.shape[1] / 256))
    centers = gather(points, fps_ids(points, count))
    ids = knn(points, centers, neighbors + 1)[..., 1:]
    relative = gather(points, ids) - centers[:, :, None]
    scale = relative.norm(dim=-1).median(-1).values.clamp_min(1e-5)
    normalized = relative / scale[..., None, None]
    covariance = normalized.transpose(-1, -2) @ normalized / neighbors
    eigen, frame = torch.linalg.eigh(covariance)
    # Fix eigenvector signs by their largest coordinate. Squared statistics and
    # circular angular gaps are also invariant to the residual PCA ambiguities.
    pivot = frame.abs().argmax(-2, keepdim=True)
    sign = frame.gather(-2, pivot).sign()
    frame = frame * torch.where(sign == 0, torch.ones_like(sign), sign)
    local = normalized @ frame[..., [2, 1, 0]]
    valid = (eigen[..., 1] > 1e-5) & (eigen[..., 1] / eigen[..., 2].clamp_min(1e-8) > 1e-4)
    return dict(centers=centers, local=local, relative=normalized, scale=scale,
                valid=valid, eigen=eigen, anchor_counts=region_counts(points, centers))


def design(uv):
    u, v = uv.unbind(-1)
    return torch.stack((torch.ones_like(u), u, v, u*u, u*v, v*v), -1)


def fit_surface(local):
    matrix = design(local[..., :2])
    weights = torch.exp(-local[..., :2].square().sum(-1) / 2)
    gram = matrix.transpose(-1, -2) @ (weights[..., None] * matrix)
    eigen = torch.linalg.eigvalsh(gram)
    valid = (eigen[..., 0] > 1e-7) & (eigen[..., -1] / eigen[..., 0].clamp_min(1e-10) < 1e6)
    ridge = 1e-4 * gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
    rhs = matrix.transpose(-1, -2) @ (weights * local[..., 2])[..., None]
    coefficients = torch.linalg.solve(gram + ridge[..., None, None] * torch.eye(6, device=local.device), rhs)
    return coefficients, valid


@torch.no_grad()
def geometry_feature(region, mode):
    local = region["local"]
    valid = region["valid"].clone()
    if mode == "D1":
        angle = torch.atan2(local[..., 1], local[..., 0]).sort(-1).values
        gaps = torch.cat((angle[..., 1:] - angle[..., :-1],
                          angle[..., :1] + 2 * torch.pi - angle[..., -1:]), -1)
        value = gaps.amax(-1) / (2 * torch.pi)
        valid &= (local[..., :2].norm(dim=-1) > 1e-5).all(-1)
    elif mode == "D2":
        small, small_valid = fit_surface(local[..., :12, :])
        large, large_valid = fit_surface(local)
        # Identical observed query sites within both supports, without extrapolation.
        query = design(local[..., :12, :2])
        value = (query @ (small - large)).square().mean((-1, -2))
        valid &= small_valid & large_valid
    elif mode == "D3":
        # Test central supported sites. Require angular enclosure by retained
        # neighbours around each query, equivalent to 2D convex-hull membership.
        uv = local[..., :2]
        directions = uv[..., None, :, :] - uv[..., :, None, :]
        eye = torch.eye(local.shape[-2], device=local.device, dtype=torch.bool)
        angles = torch.atan2(directions[..., 1], directions[..., 0])
        angles = angles[..., ~eye].reshape(*local.shape[:-2], local.shape[-2], local.shape[-2]-1).sort(-1).values
        gaps = torch.cat((angles[..., 1:]-angles[..., :-1],
                          angles[..., :1]+2*torch.pi-angles[..., -1:]), -1)
        supported = gaps.amax(-1) < torch.pi-1e-5
        valid &= supported.sum(-1) >= 4
        distance = uv.square().sum(-1).masked_fill(~supported, torch.inf)
        order = distance.argsort(dim=-1, stable=True)
        sorted_local = local.gather(-2, order[..., None].expand_as(local))
        errors = []
        for index in range(4):
            kept = torch.cat((sorted_local[..., :index, :], sorted_local[..., index+1:, :]), -2)
            held = sorted_local[..., index, :]
            coefficients, fit_valid = fit_surface(kept)
            directions = kept[..., :2] - held[..., None, :2]
            angle = torch.atan2(directions[..., 1], directions[..., 0]).sort(-1).values
            gaps = torch.cat((angle[..., 1:] - angle[..., :-1],
                              angle[..., :1] + 2 * torch.pi - angle[..., -1:]), -1)
            valid &= fit_valid & (gaps.amax(-1) < torch.pi - 1e-5)
            estimate = (design(held[..., :2])[..., None, :] @ coefficients).squeeze(-1).squeeze(-1)
            errors.append((held[..., 2] - estimate).square())
        value = torch.stack(errors, -1).mean(-1)
    else:
        raise ValueError(f"Unknown geometry mode {mode}")
    valid &= torch.isfinite(value)
    return torch.where(valid, value, torch.zeros_like(value)), valid
