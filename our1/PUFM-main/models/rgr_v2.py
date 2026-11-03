"""Shared multiscale reliable geometry encoder and region recovery queries."""
import torch
from torch import nn
from .adaptive_ops import gather, knn


class RGR(nn.Module):
    feature_dim = 64

    def __init__(self, max_offset_spacing=2.):
        super().__init__()
        self.max_offset_spacing = max_offset_spacing
        self.scales = (8, 24)
        self.edges = nn.ModuleList([nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 64), nn.SiLU()) for _ in self.scales])
        self.fuse = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.point_head = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 4))
        self.region_fuse = nn.Sequential(nn.Linear(64 + 4, 64), nn.SiLU())
        self.region_head = nn.Linear(64, 6)
        # Start close to the identity, without zeroing the entire computation graph.
        nn.init.normal_(self.point_head[-1].weight, std=.001)
        nn.init.zeros_(self.point_head[-1].bias)
        nn.init.normal_(self.region_head.weight, std=.001)
        nn.init.zeros_(self.region_head.bias)

    def forward(self, points):
        encoded = []
        spacing = None
        for count, network in zip(self.scales, self.edges):
            ids = knn(points, points, min(count, points.shape[1]))
            relative = gather(points, ids) - points[:, :, None]
            distance = relative.norm(dim=-1, keepdim=True)
            encoded.append(network(torch.cat([points[:, :, None].expand_as(relative), relative, distance], -1)).amax(2))
            if spacing is None:
                spacing = distance[:, :, 1:].mean(2).clamp_min(1e-5)
        feature = self.fuse(torch.cat(encoded, -1))
        prediction = self.point_head(feature)
        offset = prediction[..., 1:].tanh() * spacing * self.max_offset_spacing
        return {"xyz": points + offset, "features": feature, "q": prediction[..., :1].sigmoid(),
                "q_logit": prediction[..., :1], "offset": offset, "spacing": spacing}

    def recover_regions(self, geometry, centers):
        ids = knn(geometry["xyz"], centers, min(24, geometry["xyz"].shape[1]))
        xyz = gather(geometry["xyz"], ids)
        relative = xyz - centers[:, :, None]
        radius = relative.norm(dim=-1).mean(2, keepdim=True).clamp_min(1e-4)
        q = gather(geometry["q"], ids)
        feature = (gather(geometry["features"], ids) * q).sum(2) / q.sum(2).clamp_min(1e-5)
        summary = torch.cat([q.mean(2), relative.mean(2),], -1)
        feature = self.region_fuse(torch.cat([feature, summary], -1))
        prediction = self.region_head(feature)
        h, r, u = prediction[..., 0:1].sigmoid(), prediction[..., 1:2].sigmoid(), prediction[..., 2:3].sigmoid()
        virtual = centers + prediction[..., 3:6].tanh() * 2 * radius
        return {"features": feature, "h": h, "r": r, "u": u, "virtual": virtual,
                "radius": radius, "q": q.mean(2), "logits": prediction[..., :3]}
