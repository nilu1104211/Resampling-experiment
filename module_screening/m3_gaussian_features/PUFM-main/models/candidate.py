"""M3a: covariance geometry, inspired by PU-Gaussian; not its generator.

Local covariance and sign-invariant normal outer products avoid eigenvector
sign flips. Geometry is computed on observed input only, reused at every t.
"""
import torch
from torch import nn
from .adaptive_ops import knn, gather


class FeatureBranch(nn.Module):
    kind = 'features'

    def __init__(self, config, control=False):
        super().__init__()
        self.k = config['geometry_neighbors']
        self.control = control
        width = config['branch_width']
        self.net = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, 128))
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)

    def descriptors(self, points):
        with torch.no_grad():
            local = gather(points, knn(points, points, min(self.k, points.shape[1])))
            center = local.mean(2)
            delta = local - center[:, :, None]
            covariance = delta.transpose(-1, -2) @ delta / local.shape[2]
            values, vectors = torch.linalg.eigh(covariance.float())
            values = values.clamp_min(0)
            trace = values.sum(-1, keepdim=True).clamp_min(1e-8)
            normal = vectors[..., 0]
            normal_outer = normal[..., :, None] * normal[..., None, :]
            tri = torch.triu_indices(3, 3, device=points.device)
            cov6 = covariance[..., tri[0], tri[1]] / trace
            normal6 = normal_outer[..., tri[0], tri[1]]
            # Six covariance terms + three eigenvalue ratios + six normal terms + scale.
            return torch.cat([cov6, values / trace, normal6, trace.sqrt()], -1)

    def forward(self, points):
        if self.control:
            # Same parameter count, input XYZ only, no neighbor geometry.
            descriptor = torch.cat([points.repeat(1, 1, 5), points.new_ones(*points.shape[:2], 1)], -1)
        else:
            descriptor = self.descriptors(points)
        return self.net(descriptor)


def build_candidate(config):
    return FeatureBranch(config, config['variant'] == 'mlp_control') if config['variant'] in {'m3a', 'mlp_control'} else None


def candidate_loss(prediction, target, config):
    raise RuntimeError('M3 uses the original FM loss, not endpoint supervision')
