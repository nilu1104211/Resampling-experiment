"""RGR -> marginal-gain budget -> original PUFM backbone, with real conditions."""
from dataclasses import dataclass, asdict
from types import SimpleNamespace
import torch
from torch import nn
from .diffusion import PUFM, PUFM_w_attn
from .adaptive_ops import gather, knn, fps, region_counts
from .utils import midpoint_interpolate
from .rgr_v2 import RGR
from .marginal_gain import GainNet
from .budget_allocator import build_regions, allocate, candidate_pool, initialize


@dataclass
class AdaptiveConfig:
    backbone: str = "pufm_w_attn"
    use_rgr: bool = True
    use_reliability: bool = True
    use_recovery: bool = True
    use_uncertainty: bool = True
    use_virtual: bool = True
    allocation: str = "gain"  # midpoint / uniform / density / curvature / static / gain
    gain_form: str = "monotone"
    condition: bool = False
    regions: int = 32  # per 256 observed points
    minimum: int = 4
    capacity: int = 64
    swap_step: int = 8
    swap_iterations: int = 32
    flow_steps: int = 5
    max_offset_spacing: float = 2.
    teacher_gain_scale: float = 1000.


class ConditionVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(69, 64), nn.SiLU(), nn.Linear(64, 3))
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)
        self.gate = nn.Parameter(torch.tensor(.1))

    def forward(self, query, state):
        geometry, regions = state["geometry"], state["regions"]
        ids = knn(geometry["xyz"], query, 3)
        xyz = gather(geometry["xyz"], ids)
        distance = (query[:, :, None] - xyz).norm(dim=-1).clamp_min(1e-5)
        weight = (1 / distance)
        weight = weight / weight.sum(-1, keepdim=True)
        feature = (gather(geometry["features"], ids) * weight[..., None]).sum(2)
        q = (gather(geometry["q"], ids) * weight[..., None]).sum(2)
        relative = query - (xyz * weight[..., None]).sum(2)
        region_id = knn(regions["centers"], query, 1).squeeze(-1)
        if state["budget"] is None:
            density = query.new_zeros(len(query), query.shape[1], 1)
        else:
            density = gather(torch.log1p(state["budget"].float())[..., None], region_id)
        return self.gate.tanh() * self.net(torch.cat([feature, q, density, relative], -1))


class AdaptivePUFM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or AdaptiveConfig()
        if self.config.flow_steps < 1 or self.config.regions < 1 or not 0 <= self.config.minimum <= self.config.capacity <= 256 or self.config.swap_step < 1 or self.config.swap_iterations < 0:
            raise ValueError("Invalid flow steps, region bounds or swap configuration")
        if self.config.backbone not in {"pufm", "pufm_w_attn"}:
            raise ValueError("Unknown PUFM backbone")
        backbone = PUFM_w_attn if self.config.backbone == "pufm_w_attn" else PUFM
        self.flow = backbone(SimpleNamespace())
        self.rgr = RGR(self.config.max_offset_spacing)
        self.gain = GainNet(form=self.config.gain_form)
        self.conditioner = ConditionVelocity()

    def encode(self, points, output_size=None, *, for_teacher=False):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 32 or not torch.isfinite(points).all():
            raise ValueError("Expected finite observed points [B,N,3], N>=32")
        output_size = points.shape[1] * 4 if output_size is None else int(output_size)
        if output_size < 1024:
            raise ValueError("The retained original PUFM backbone needs at least 1024 query points")
        if output_size < points.shape[1]:
            raise ValueError("Output budget must contain all retained input anchors")
        if self.config.use_rgr:
            geometry = self.rgr(points)
            if not self.config.use_reliability:
                geometry["q"] = torch.ones_like(geometry["q"])
        else:
            geometry = {"xyz": points, "features": points.new_zeros(*points.shape[:2], 64),
                        "q": points.new_ones(*points.shape[:2], 1), "offset": torch.zeros_like(points)}
        if self.config.allocation == "midpoint" and not self.config.use_rgr and not self.config.condition and not for_teacher:
            return {"input": points, "geometry": geometry, "regions": None, "pools": None,
                    "output_size": output_size, "budget": None, "initial": None}
        count = min(points.shape[1], max(4, round(self.config.regions * points.shape[1] / 256)))
        regions = build_regions(points, geometry, self.rgr if self.config.use_rgr else None, count)
        if self.config.use_rgr and not self.config.use_recovery:
            for key in ["h", "r", "u"]:
                regions[key] = torch.zeros_like(regions[key])
            regions["virtual"] = regions["centers"]
        elif not self.config.use_uncertainty:
            regions["u"] = torch.zeros_like(regions["u"])
        regions["gain_features"] = torch.cat([regions["features"], regions["observable"], regions["q"], regions["h"], regions["r"], regions["u"]], -1)
        pools = candidate_pool(geometry, regions, self.config.capacity, self.config.use_rgr and self.config.use_virtual) if for_teacher or self.config.allocation != "midpoint" else None
        return {"input": points, "geometry": geometry, "regions": regions, "pools": pools,
                "output_size": output_size, "budget": None, "initial": None}

    def initialize(self, state, budget=None):
        count = state["output_size"] - state["input"].shape[1]
        if self.config.allocation == "midpoint" and budget is None:
            if state["output_size"] % state["input"].shape[1]:
                raise ValueError("Original midpoint baseline needs an integer upsampling ratio")
            args = SimpleNamespace(up_rate=state["output_size"] // state["input"].shape[1])
            state["initial"] = midpoint_interpolate(args, state["geometry"]["xyz"].transpose(1, 2).contiguous()).transpose(1, 2)
            state["owner"] = None
            return state
        if budget is None:
            budget = allocate(state["regions"], self.gain, count, self.config.minimum, self.config.capacity,
                              self.config.swap_step, self.config.swap_iterations, self.config.allocation)
        budget = budget.to(state["input"].device, dtype=torch.long)
        expected = (len(state["input"]), state["regions"]["centers"].shape[1])
        if budget.shape != expected or torch.any(budget < 0) or torch.any(budget > self.config.capacity) or torch.any(budget.sum(-1) != count):
            raise ValueError("Illegal integer region budget")
        state["budget"] = budget
        state["initial"], state["owner"] = initialize(state["geometry"]["xyz"], state["pools"], budget)
        return state

    def prepare(self, points, output_size=None, budget=None):
        return self.initialize(self.encode(points, output_size), budget)

    def velocity(self, query, time, state):
        velocity = self.flow(query.transpose(1, 2).contiguous(), state["initial"].transpose(1, 2).contiguous(), time).transpose(1, 2)
        if self.config.condition:
            velocity = velocity + self.conditioner(query, state)
        return velocity

    def integrate(self, state, steps=None):
        steps = self.config.flow_steps if steps is None else steps
        if steps < 1:
            raise ValueError("Integration needs at least one step")
        query = state["initial"]
        for step in range(steps):
            t = query.new_full((len(query),), step / steps)
            query = query + self.velocity(query, t, state) / steps
        return query

    def forward(self, points, output_size=None, *, return_diagnostics=False):
        """Inference interface: observed geometry ONLY; no GT / label arguments."""
        state = self.prepare(points, output_size)
        prediction = self.integrate(state)
        if prediction.shape != (len(points), state["output_size"], 3) or not torch.isfinite(prediction).all():
            raise FloatingPointError("Invalid fixed-budget model output")
        return (prediction, self.diagnostics(state, prediction)) if return_diagnostics else prediction

    def diagnostics(self, state, prediction):
        region = state["regions"]
        if region is None:
            count = min(state["input"].shape[1], max(4, round(self.config.regions * state["input"].shape[1] / 256)))
            centers = fps(state["input"], count).detach()
            zeros = centers.new_zeros(*centers.shape[:2], 1)
            region = {"h": zeros, "r": zeros, "u": zeros, "virtual": centers}
        centers = region["centers"].detach() if "centers" in region else centers
        initial_count = region_counts(state["initial"], centers)
        final_count = region_counts(prediction, centers)
        result = {"region_centers": centers, "initial_counts": initial_count, "final_counts": final_count,
                  "density_total_variation": .5 * (initial_count - final_count).abs().sum(-1) / prediction.shape[1],
                  "input_reliability": state["geometry"]["q"], "offset": state["geometry"]["offset"],
                  "hole": region["h"], "recoverability": region["r"],
                  "uncertainty": region["u"], "virtual_queries": region["virtual"]}
        if state["budget"] is not None:
            result["budget"] = state["budget"]
            result["target_total_counts"] = state["budget"] + state["regions"]["anchor_counts"]
        return result
