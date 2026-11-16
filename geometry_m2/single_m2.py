"""One geometry-conditioned allocator and a separately frozen PUFM generator."""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import nn
from models.diffusion import PUFM, PUFM_w_attn
from models.marginal_gain import GainNet
from models.adaptive_ops import gather, assign_regions, region_counts
from geometry_m2.geometry_features import local_regions, geometry_feature
from geometry_m2.allocation import allocate, candidate_pools, initialize


@dataclass
class Config:
    backbone: str = "pufm_w_attn"
    mode: str = "D1"
    regions: int = 16
    minimum: int = 4
    capacity: int = 64
    delta: int = 8
    exchanges: int = 32
    flow_steps: int = 5
    gain_threshold: float = .01  # In units of the frozen training gain scale.


class LocalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())

    def forward(self, relative):
        return self.net(relative).amax(-2)


class SingleM2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LocalEncoder()
        # Shared basics: centre xyz, local scale, input occupancy, valid flag.
        self.gain = GainNet(feature_dim=32 + 1 + 6)
        self.register_buffer("geometry_center", torch.zeros(()))
        self.register_buffer("geometry_scale", torch.ones(()))
        self.register_buffer("gain_scale", torch.ones(()))
        self.register_buffer("noise_threshold", torch.zeros(()))

    def forward(self, region, mode, input_size):
        value, valid = geometry_feature(region, mode)
        scalar = ((value - self.geometry_center) / self.geometry_scale).clamp(-5, 5)
        scalar = torch.where(valid, scalar, torch.zeros_like(scalar))
        basic = torch.cat((region["centers"], region["scale"][..., None],
                           region["anchor_counts"][..., None] / input_size, valid[..., None]), -1)
        features = torch.cat((self.encoder(region["relative"]), scalar[..., None], basic), -1)
        return features, scalar, valid


class SingleM2PUFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.mode not in {"D1", "D2", "D3"} or config.backbone not in {"pufm", "pufm_w_attn"}:
            raise ValueError("Invalid geometry or backbone")
        if config.flow_steps < 1 or config.regions < 1 or not 0 <= config.minimum < config.capacity or config.delta < 1:
            raise ValueError("Invalid flow or budget configuration")
        self.config = config
        self.flow = (PUFM_w_attn if config.backbone == "pufm_w_attn" else PUFM)(SimpleNamespace())
        self.local_encoder = LocalEncoder()
        self.conditioner = nn.Sequential(nn.Linear(37, 64), nn.SiLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)
        self.m2 = SingleM2()
        self.allocation = "uniform"

    def encode(self, points, output_size=None, *, with_m2=True):
        output_size = output_size or points.shape[1] * 4
        total = output_size - points.shape[1]
        region = local_regions(points, self.config.regions)
        pools, capacities = candidate_pools(points, region, self.config.capacity,
                                            minimum=self.config.minimum, total=total)
        state = dict(input=points, regions=region, pools=pools, capacities=capacities,
                     total=total, output_size=output_size,
                     condition_features=self.local_encoder(region["relative"]))
        if with_m2:
            state["features"], state["scalar"], state["valid"] = self.m2(region, self.config.mode, points.shape[1])
        return state

    def prepare(self, points, output_size=None):
        if self.allocation == "midpoint":
            from models.utils import midpoint_interpolate
            output_size = output_size or points.shape[1]*4
            initial = midpoint_interpolate(SimpleNamespace(up_rate=output_size//points.shape[1]),
                                           points.transpose(1, 2).contiguous()).transpose(1, 2)
            return dict(input=points, initial=initial, budget=None)
        state = self.encode(points, output_size, with_m2=self.allocation in {"gain", "rule"})
        return initialize(state, allocate(self, state, self.allocation), self.config.minimum)

    def velocity(self, query, time, state):
        velocity = self.flow(query.transpose(1, 2).contiguous(), state["initial"].transpose(1, 2).contiguous(), time).transpose(1, 2)
        if state["budget"] is None:
            return velocity
        owners = assign_regions(query, state["regions"]["centers"])
        features = gather(state["condition_features"], owners)
        relative = query - gather(state["regions"]["centers"], owners)
        budget_fraction = gather(state["budget"][..., None].float() / state["total"], owners)
        t = time[:, None, None].expand(-1, query.shape[1], 1)
        return velocity + self.conditioner(torch.cat((features, relative, budget_fraction, t), -1))

    def integrate(self, state):
        query = state["initial"]
        for step in range(self.config.flow_steps):
            time = query.new_full((len(query),), step/self.config.flow_steps)
            query = query + self.velocity(query, time, state) / self.config.flow_steps
        return query

    def forward(self, points, output_size=None):
        state = self.prepare(points, output_size)
        prediction = self.integrate(state)
        if prediction.shape != state["initial"].shape or not torch.isfinite(prediction).all():
            raise FloatingPointError("Invalid fixed-budget output")
        return prediction

    def diagnostics(self, state, prediction):
        return dict(budget=state["budget"], capacities=state["capacities"],
                    initial_counts=region_counts(state["initial"], state["regions"]["centers"]),
                    final_counts=region_counts(prediction, state["regions"]["centers"]),
                    valid_fraction=state.get("valid", state["regions"]["valid"]).float().mean(-1))
