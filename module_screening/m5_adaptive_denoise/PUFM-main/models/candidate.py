"""M5a: ASDN-inspired trainable denoiser + learned cloud-level stopping.

This lightweight V3 adaptation is not ASDN's official architecture/entropy
classifier and is not a per-point degradation classifier. PUFM stays frozen.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .adaptive_ops import knn, gather


class AdaptiveDenoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config['branch_width']
        self.edges = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.offset = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
        self.continue_head = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.normal_(self.offset[-1].weight, std=.001)
        nn.init.zeros_(self.offset[-1].bias)

    def proposal(self, points):
        neighbors = gather(points, knn(points, points, min(16, points.shape[1])))
        relative = neighbors - points[:, :, None]
        spacing = relative.norm(dim=-1)[..., 1:].mean(-1, keepdim=True).clamp_min(1e-5).detach()
        normalized = relative / spacing[:, :, None]
        edge = torch.cat([normalized, normalized.norm(dim=-1, keepdim=True)], -1)
        feature = self.edges(edge).mean(2)
        delta = self.offset(feature).tanh() * spacing * self.config['max_offset_spacing']
        pooled = torch.cat([feature.mean(1), feature.amax(1)], -1)
        logit = self.continue_head(pooled).squeeze(-1)
        return points + delta, logit, spacing

    def training_loss(self, points, clean):
        current = points
        total = points.new_zeros(())
        for _ in range(self.config['denoise_steps']):
            proposed, logit, spacing = self.proposal(current)
            reconstruction = ((proposed - clean) / spacing).square().mean()
            with torch.no_grad():
                before = ((current - clean) / spacing).square().mean((1, 2))
                after = ((proposed - clean) / spacing).square().mean((1, 2))
                should_continue = (before - after > self.config['stop_improvement']).float()
            total = total + reconstruction + self.config['stop_loss_weight'] * F.binary_cross_entropy_with_logits(logit, should_continue)
            current = proposed.detach()
        return total / self.config['denoise_steps']

    def denoise(self, points):
        current = points
        active = torch.ones(len(points), dtype=torch.bool, device=points.device)
        steps = torch.zeros(len(points), dtype=torch.long, device=points.device)
        for _ in range(self.config['denoise_steps']):
            proposed, logit, _ = self.proposal(current)
            active = active & (logit.sigmoid() >= self.config['stop_threshold'])
            current = torch.where(active[:, None, None], proposed, current)
            steps += active.long()
            if not active.any():
                break
        return current, {'denoise_steps': steps, 'input_displacement_mean': (current - points).norm(dim=-1).mean(1),
                         'input_displacement_max': (current - points).norm(dim=-1).amax(1)}


def build_candidate(config):
    return AdaptiveDenoiser(config) if config['variant'] == 'm5a' else None


def candidate_loss(prediction, target, config):
    raise RuntimeError('M5 trains only its denoiser/stopping network')
