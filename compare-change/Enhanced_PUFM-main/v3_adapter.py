"""PUFM++ RIN with observed-input initialization and its two training stages."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from models.rin import Denoiser_backbone
from v3_common import assignment_target, chamfer, dataset, fps, index_points, knn, midpoint, patch_predict


class Adapter(nn.Module):
    compatibility_keys = ()
    training_description = "PUFM++ RIN; s1 transport-aligned FM, s2 t=0 Chamfer fine-tuning; exact Hungarian transport backend; actual V3 input"
    inference_description = "KNN midpoint+FPS; RIN Euler integration; input-only patch normalization; 4x; optional saved schedule/manifold step"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Denoiser_backbone(num_x=1024, z_dim=512, k=16)
        if args.command == "predict" and not args.schedule and not args.uniform_schedule:
            raise ValueError("PUFM++ prediction requires --schedule from V3 training calibration, or explicit --uniform-schedule for an ablation")
        if args.schedule:
            saved = torch.load(args.schedule, map_location="cpu", weights_only=False)
            if "cdf" in saved and "grid" in saved:
                self.times = np.interp(np.linspace(0, 1, args.flow_steps + 1), saved["cdf"], saved["grid"]).tolist()
            else:
                self.times = saved["t_sched"].tolist()
            if saved.get("partial_debug_run"):
                raise ValueError("A debug schedule cannot be used for formal inference")
            if not self.times or abs(self.times[0]) > 1e-6 or abs(self.times[-1] - 1) > 1e-6 or any(b <= a for a, b in zip(self.times, self.times[1:])):
                raise ValueError("A time schedule must increase strictly from zero to one")
        else:
            self.times = np.linspace(0, 1, args.flow_steps + 1).tolist()

    def velocity(self, query, t, latent=None):
        return self.model(query.transpose(1, 2).contiguous(), t, None, latent)

    def training_loss(self, points, gt):
        if points.shape[1:] != (256, 3) or gt.shape[1:] != (1024, 3):
            raise ValueError("PUFM++ V3 training expects 256 -> 1024 patches")
        x0 = midpoint(points) + .02 * torch.randn(len(points), 1024, 3, device=points.device)
        if self.args.stage == "s1":
            target = assignment_target(x0, gt)
            t = 1 - torch.cos(torch.rand(len(points), device=points.device) * torch.pi / 2)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * target
        else:
            t = torch.zeros(len(points), device=points.device)
            xt = x0
        latent = None
        if self.args.stage == "s2" or torch.rand((), device=points.device) < .9:
            with torch.no_grad():
                _, latent = self.velocity(xt, t)
        velocity, _ = self.velocity(xt, t, latent)
        velocity = velocity.transpose(1, 2)
        if self.args.stage == "s2":
            return chamfer(x0 + velocity, gt)
        return (velocity - (target - x0)).square().sum() / len(points)

    def predict_patch(self, points):
        initial = midpoint(points)
        query = initial
        for start, stop in zip(self.times, self.times[1:]):
            t = torch.full((len(points),), start, device=points.device)
            velocity, _ = self.velocity(query, t)
            query = query + (stop - start) * velocity.transpose(1, 2)
            query = fps(torch.cat([query, points], dim=1), 1024)
            if self.args.manifold_step:
                nearest = index_points(initial, knn(initial, query, 1)).squeeze(2)
                # Gradient direction of the L1 nearest-surface proxy, no GT involved.
                direction = (query - nearest).sign()
                direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
                query = query - self.args.manifold_step * direction
        return query

    def predict(self, points, output_count):
        return patch_predict(points, output_count, self.predict_patch)

    @torch.no_grad()
    def calibrate_schedule(self, ds, args):
        from torch.utils.data import DataLoader
        from adaptive_resampling.v3_dataset import EpochSampler
        args.training_profile = ds.training_profile
        self.eval()
        sampler = EpochSampler(ds, shuffle=False)
        sampler.set_epoch(0)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
        grid = np.linspace(0, 1, args.schedule_grid)
        errors = []
        for time in grid:
            losses = []
            # Same online data and noise for every t: the curve measures time, not resampling.
            torch.manual_seed(args.seed)
            for step, batch in enumerate(loader):
                x = batch["input_xyz"].to(args.device).float()
                gt = batch["gt_xyz"].to(args.device).float()
                x0 = midpoint(x) + .02 * torch.randn_like(gt)
                x1 = assignment_target(x0, gt)
                t = torch.full((len(x),), float(time), device=x.device)
                xt = (1 - time) * x0 + time * x1
                _, latent = self.velocity(xt, t)
                v, _ = self.velocity(xt, t, latent)
                losses.append((v.transpose(1, 2) - (x1 - x0)).square().mean().item())
                if args.schedule_batches and step + 1 >= args.schedule_batches:
                    break
            errors.append(float(np.mean(losses)))
            print(json.dumps({"schedule_t": float(time), "mse": errors[-1]}), flush=True)
        # Match the source entropy scheduler, with epsilon for degenerate curves.
        density = np.maximum(np.asarray(errors), 1e-8)
        cdf = np.r_[0, np.cumsum((density[:-1] + density[1:]) * .5 * np.diff(grid))]
        cdf /= cdf[-1]
        times = np.interp(np.linspace(0, 1, args.flow_steps + 1), cdf, grid)
        result = {"t_sched": torch.tensor(times, dtype=torch.float32), "grid": grid,
                  "loss": errors, "protocol": "V3", "split": "train", "cdf": cdf,
                  "partial_debug_run": bool(args.schedule_batches), "args": vars(args)}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, args.output)
        ds.close()
        return {"status": "passed", "schedule": args.output, "times": times.tolist(),
                "partial_debug_run": bool(args.schedule_batches)}
