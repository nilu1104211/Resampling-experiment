"""M1: shared reliability / candidate-hole estimation and bounded seed guidance."""
import torch
from torch import nn
from .adaptive_ops import gather, knn
from .geometry import pool


class RAGR(nn.Module):
    def __init__(self, width=64, max_move=.5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(30, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.reliability = nn.Linear(width, 1)
        self.hole = nn.Sequential(nn.Linear(width + 2, width), nn.SiLU(), nn.Linear(width, 1))
        self.direction = nn.Linear(width, 3)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.max_move = max_move

    def forward(self, geo):
        f = geo['features']
        feat = self.encoder(torch.cat([f, f.mean(1, keepdim=True).expand_as(f)], -1))
        rlogit = self.reliability(feat).squeeze(-1)
        region = pool(feat, geo['labels'], geo['anchors'].shape[1])
        distance = (geo['candidates'] - geo['anchors']).norm(dim=-1, keepdim=True) / geo['region_scale']
        nearest = knn(geo['points'], geo['candidates'], 1)
        gap = (gather(geo['points'], nearest).squeeze(2) - geo['candidates']).norm(dim=-1, keepdim=True) / geo['region_scale']
        hlogit = self.hole(torch.cat([region, distance, gap], -1)).squeeze(-1)
        return dict(rlogit=rlogit, hlogit=hlogit, r=rlogit.sigmoid(), h=hlogit.sigmoid(), feat=feat)

    def refine(self, initial, geo, state, use_reliability=True, use_hole=True):
        ids = knn(geo['points'], initial, 8)
        neighbors = gather(geo['points'], ids)
        rel = gather(state['r'][..., None], ids).squeeze(-1)
        weights = rel if use_reliability else torch.ones_like(rel)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        centroid = (neighbors * weights[..., None]).sum(2)
        local_scale = gather(geo['scale'][..., None], ids).mean(2)
        feature = (gather(state['feat'], ids) * weights[..., None]).sum(2)
        untrusted = 1 - rel.mean(-1, keepdim=True) if use_reliability else torch.zeros_like(local_scale)
        direction = (centroid - initial) * untrusted
        if use_hole:
            candidate_id = knn(geo['candidates'], initial, 1).squeeze(-1)
            delta = gather(geo['candidates'], candidate_id) - initial
            confidence = gather(state['h'][..., None], candidate_id)
            gate = torch.exp(-delta.square().sum(-1, keepdim=True) / (4 * local_scale.square())) * confidence
            direction = direction + gate * delta
        else:
            gate = torch.zeros_like(local_scale)
        direction = direction + (untrusted + gate).clamp_max(1) * self.direction(feature).tanh() * local_scale
        bound = self.max_move * local_scale
        direction = direction * (bound / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
        movement = self.gamma.tanh() * direction
        return initial + movement, movement
