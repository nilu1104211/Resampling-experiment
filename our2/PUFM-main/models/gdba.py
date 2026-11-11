"""M2: independent soft budget supervision and exact constrained hard allocation.

The discrete selection is intentionally detached, not a claimed differentiable sampler.
"""
import torch
from torch import nn
from .adaptive_ops import gather, knn, fps_ids
from .geometry import pool


def bounded_soft_budget(logits, total, lower, upper, temperature=1.):
    if temperature <= 0 or (lower > upper).any() or (lower.sum(-1) > total).any() or (upper.sum(-1) < total).any():
        raise ValueError('Infeasible budget constraints')
    result = lower.to(logits.dtype)
    for _ in range(logits.shape[-1] + 1):
        remaining = (total - result.sum(-1, keepdim=True)).clamp_min(0)
        room = (upper - result).clamp_min(0)
        active = (room > 1e-5).to(logits.dtype)
        # Renormalize active logits at every saturation step to avoid underflow
        # when the only high-probability region has already reached capacity.
        w = torch.softmax((logits / temperature).masked_fill(active == 0, -1e30), -1) * active
        proposal = remaining * w / w.sum(-1, keepdim=True).clamp_min(1e-30)
        result = result + torch.minimum(proposal, room)
    return result


@torch.no_grad()
def integer_budget(soft, total, lower, upper):
    result = soft.floor().long().maximum(lower.long()).minimum(upper.long())
    for b in range(len(result)):
        left = total - int(result[b].sum())
        while left:
            feasible = result[b] < upper[b] if left > 0 else result[b] > lower[b]
            score = soft[b] - result[b] if left > 0 else result[b] - soft[b]
            if not feasible.any():
                raise ValueError('No feasible integer allocation')
            index = score.masked_fill(~feasible, -torch.inf).argmax()
            result[b, index] += 1 if left > 0 else -1
            left += -1 if left > 0 else 1
    return result


class GDBA(nn.Module):
    def __init__(self, width=64, minimum=.5, maximum=2., temperature=1.):
        super().__init__()
        self.independent = nn.Sequential(nn.Linear(30, width), nn.SiLU(), nn.Linear(width, 1))
        self.condition = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)
        self.minimum, self.maximum, self.temperature = minimum, maximum, temperature

    def forward(self, geo, m1=None, joint=True, policy='learned'):
        f = geo['region_features']
        logits = self.independent(torch.cat([f, f.mean(1, keepdim=True).expand_as(f)], -1)).squeeze(-1)
        if m1 is not None and joint:
            reliable = pool(m1['r'][..., None], geo['labels'], f.shape[1]).squeeze(-1)
            condition = torch.stack([reliable, m1['h']], -1)
            if self.training:
                condition = condition * (torch.rand(len(f), 1, 1, device=f.device) > .25)
            logits = logits + self.condition(condition).squeeze(-1)
        if policy == 'uniform': logits = torch.zeros_like(logits)
        elif policy == 'curvature': logits = f[..., 0].clamp_min(1e-5).log()
        elif policy == 'density': logits = f[..., -1]
        return logits

    @torch.no_grad()
    def candidates(self, initial, geo):
        anchors = geo['anchors']
        labels = knn(anchors, initial, 1).squeeze(-1)
        target_capacity = int(self.maximum * initial.shape[1] / anchors.shape[1])
        pools, capacities = [], []
        for b in range(len(initial)):
            regions = []
            for j in range(anchors.shape[1]):
                # All convex interpolants stay inside this Voronoi region.
                original = torch.cat([initial[b, labels[b] == j], geo['points'][b, geo['labels'][b] == j]], 0)
                original = torch.unique(original, dim=0)
                candidates = original
                for level in range(1, 9):
                    if len(candidates) >= target_capacity: break
                    alpha = level / 10.
                    candidates = torch.unique(torch.cat([candidates, original * (1-alpha) + anchors[b, j] * alpha]), dim=0)
                # Never duplicate a coordinate to fill an isolated/degenerate region.
                regions.append(candidates)
            pools.append(regions)
            capacities.append([min(len(p), target_capacity) for p in regions])
        upper = torch.tensor(capacities, device=initial.device)
        lower = torch.full_like(upper, int(self.minimum * initial.shape[1] / anchors.shape[1])).minimum(upper)
        return pools, lower, upper

    def allocate(self, initial, geo, logits, strength=1.):
        pools, lower, upper = self.candidates(initial, geo)
        soft = bounded_soft_budget(logits, initial.shape[1], lower, upper, self.temperature)
        if strength < 1:
            original_labels = knn(geo['anchors'], initial, 1).squeeze(-1)
            original_count = torch.nn.functional.one_hot(original_labels, logits.shape[1]).sum(1)
            # Respect safety constraints even during the transition.
            mixed_logits = ((1-strength)*original_count + strength*soft).clamp_min(1).log()
            soft = bounded_soft_budget(mixed_logits, initial.shape[1], lower, upper, 1.)
        hard = integer_budget(soft, initial.shape[1], lower, upper)
        output = []
        for b, regions in enumerate(pools):
            selected = [gather(p[None], fps_ids(p[None], int(hard[b,j])))[0]
                        for j,p in enumerate(regions) if hard[b,j] > 0]
            output.append(torch.cat(selected))
        return torch.stack(output), dict(soft=soft, hard=hard, lower=lower, upper=upper)
