"""Cache input-derived relation messages in X0 order for the PUFM trajectory."""
import torch
from torch import nn

from .adaptive_ops import gather, knn
from .budget_allocator import candidate_origins


class RelationFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(69, 32), nn.SiLU())
        self.relations = nn.ModuleList(nn.Linear(32, 32, bias=False) for _ in range(3))
        self.film = nn.Sequential(nn.Linear(32 + 8, 64), nn.SiLU(), nn.Linear(64, 256))
        nn.init.normal_(self.film[-1].weight, std=.001)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, state, virtual):
        initial = state["initial"]
        geometry, region = state["geometry"], state["regions"]
        ids = knn(geometry["xyz"], initial, 8)
        relative = gather(geometry["xyz"], ids) - initial[:, :, None]
        distance = relative.norm(dim=-1, keepdim=True)
        spacing = distance.mean(2, keepdim=True).clamp_min(1e-5)
        neighbor_q = gather(geometry["q"], ids)
        weights = (-distance / spacing + neighbor_q.clamp_min(1e-5).log()).softmax(2)
        q = (weights * neighbor_q).sum(2)
        feature = gather(geometry["features"], ids)
        edge = self.edge(torch.cat([feature, relative / spacing, distance / spacing, neighbor_q], -1))
        message = (weights * edge).sum(2)
        owner = state["owner"]
        region_id, kinds = candidate_origins(state, virtual)
        recovery = (kinds == 2).to(q.dtype)[..., None]
        budget_feature = torch.zeros_like(q)
        if owner is not None:
            n = geometry["xyz"].shape[1]
            q = torch.cat([geometry["q"], q[:, n:]], 1)
            budget = state["budget"]
            target = budget + region["anchor_counts"]
            budget_feature = gather((target / target.mean(-1, keepdim=True)).clamp_min(1e-5).log()[..., None], region_id)
        h, r, risk = [gather(region[key], region_id) for key in ("h", "r", "u")]
        recovery = recovery * h * r * (1 - risk)
        gates = [(1 - recovery) * q, (1 - recovery) * (1 - q), recovery]
        related = sum(layer(message) * gate for layer, gate in zip(self.relations, gates))
        context = torch.cat([related, q, h, r, risk, recovery, budget_feature,
                             spacing.squeeze(2).log(), (initial - gather(region["centers"], region_id)).norm(dim=-1, keepdim=True)], -1)
        scale, shift = self.film(context).transpose(1, 2).chunk(2, dim=1)
        return scale.tanh(), shift
