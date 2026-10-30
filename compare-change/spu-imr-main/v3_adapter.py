"""Input-only self-supervised SPU-IMR on the common V3 split."""
from pathlib import Path
import yaml
import torch
from torch import nn
from model4spu_imr import Network_Model
from tools import utils
from v3_common import fps, patch_predict, validate_prediction


class Adapter(nn.Module):
    compatibility_keys = ("spu_profile", "spu_groups", "spu_transport")
    training_description = "Original SPU-IMR iterative masked recovery on actual input only (including its degradation); no clean GT loss; exact transport backend"
    inference_description = "Repeated native masked recovery; source density filter; FPS to 4x; normalized input patches for objects"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.profile = getattr(args, "spu_profile", "shared-v3")
        self.native = self.profile == "native"
        self.shared = self.profile == "shared-v3"
        groups = getattr(args, "spu_groups", None)
        args.spu_groups = groups if groups is not None else (256 if self.native else 64)
        args.spu_profile = self.profile
        args.spu_transport = getattr(args, "spu_transport", None) or ("hungarian" if self.profile == "legacy-patch256" else "source")
        utils.configure_transport(args.spu_transport)
        if self.native and args.spu_groups != 256:
            raise ValueError("Native SPU uses 256 groups; use --spu-profile legacy-patch256 for the old adapted grouping")
        if self.shared and args.spu_groups != 64:
            raise ValueError("Shared V3 uses 256 observations and 64 groups to retain the source observations/groups ratio")
        cfg = yaml.safe_load((Path(__file__).parent / "cfgs/setting.yaml").read_text(encoding="utf-8"))
        cfg["data"]["up_rate"] = cfg["transformer"]["up_rate"] = 4
        cfg["model"]["num_group"] = args.spu_groups
        cfg["model"]["gen_points"] = 1024
        if not 4 <= args.spu_groups <= 256:
            raise ValueError("SPU group count must be between 4 and 256 for 256-point V3 patches")
        if args.spu_passes * args.spu_groups * cfg["model"]["group_size"] < 1024:
            raise ValueError("Not enough recovery candidates for 1024 outputs; increase --spu-passes")
        self.model = Network_Model(cfg)
        if self.native:
            self.training_description = "Historical native-1024 reference; excluded from shared-input training"
            self.inference_description = "SPU whole-input MMR; source bbox normalization, 8 seeded recoveries, density filter with original multiplicities, FPS to requested count"
        elif self.shared:
            self.training_description = "SPU-V3 shared unchanged 256-observation input; 64 groups x16; input-only self-supervision; " + args.spu_transport + " transport; cosine schedule"
            self.inference_description = "SPU-V3 256->1024 local MMR; shared input normalization; source seeded recovery/density filter; patch merge to exactly 8192 for 2048-input objects"

    def make_scheduler(self, optimizer, args):
        if self.profile == "legacy-patch256":
            return None
        from timm.scheduler import CosineLRScheduler
        return CosineLRScheduler(optimizer, t_initial=args.spu_schedule_epochs, cycle_mul=1,
                                lr_min=1e-5, cycle_decay=.5, warmup_lr_init=1e-5,
                                warmup_t=args.spu_warmup_epochs, cycle_limit=1, t_in_epochs=True)

    def training_loss(self, points, gt):
        # GT is deliberately unused: this is the native self-supervised comparison.
        if self.native:
            raise ValueError("Native-1024 is a historical inference reference; shared training requires --spu-profile shared-v3")
        if points.shape[1:] != (256, 3) or gt.shape[1:] != (1024, 3):
            raise ValueError("Shared SPU training requires the same 256-input/1024-GT records as the other methods; GT is not used in its self-supervised loss")
        return self.model(points, points.device).mean()

    def predict_patch(self, points):
        if self.shared:
            return self._source_predict(points, points.shape[1] * 4, bbox_normalize=False)
        outputs = []
        # The native vis path merges the batch axis, so use one cloud per call.
        for cloud in points.split(1):
            candidates = torch.cat([self.model(cloud, cloud.device, idxy=i, vis=True)
                                    for i in range(self.args.spu_passes)], dim=1)
            # Exact duplicate observations arise from overlapping local groups.
            # Never let repeated coordinates consume the fixed output budget.
            candidates = torch.unique(candidates[0], dim=0)[None]
            if candidates.shape[1] < 1024:
                raise RuntimeError("SPU generated fewer than 1024 unique candidates; increase --spu-passes")
            # Chunked KNN density filtering avoids the native all-pairs O(P^2) allocation.
            avg = []
            for chunk in candidates.split(512, dim=1):
                avg.append(torch.cdist(chunk, candidates).topk(min(30, candidates.shape[1]), largest=False).values.mean(-1))
            avg = torch.cat(avg, dim=1)
            keep = avg[0] < avg.mean() * 2
            filtered = candidates[:, keep]
            used_filter = filtered.shape[1] >= 1024
            # If the source filter removes too many points, use the generated pool.
            selected = filtered if used_filter else candidates
            outputs.append(fps(selected, 1024))
            self.last_diagnostics = {"generated_unique": candidates.shape[1], "after_density_filter": filtered.shape[1],
                                     "density_filter_used": used_filter}
        return torch.cat(outputs)

    def predict(self, points, output_count):
        if not self.native:
            return patch_predict(points, output_count, self.predict_patch)
        return self._source_predict(points, output_count, bbox_normalize=True)

    def _source_predict(self, points, output_count, *, bbox_normalize):
        if output_count != points.shape[1] * 4 or points.shape[1] < 256:
            raise ValueError("SPU comparison expects >=256 observed points and a 4x output budget")
        results = []
        for cloud in points.split(1):
            if bbox_normalize:
                low, high = cloud.amin(1, keepdim=True), cloud.amax(1, keepdim=True)
                center = (low + high) * .5
                scale = (high - low).amax().clamp_min(1e-8)
            else:
                # Preserve the common V3 model coordinates in training and testing.
                center, scale = cloud.new_zeros(1, 1, 3), cloud.new_ones(())
            normalized = (cloud - center) / scale
            outputs = []
            devices = [cloud.device.index] if cloud.is_cuda else []
            # The source evaluator resets the FPS RNG to idxy for each recovery.
            # Fork it so this method does not perturb the caller's random stream.
            with torch.random.fork_rng(devices=devices):
                for i in range(self.args.spu_passes):
                    torch.manual_seed(i)
                    outputs.append(self.model(normalized, cloud.device, idxy=i, vis=True))
                final_rng = torch.get_rng_state()
            candidates = torch.cat(outputs, 1)
            # Keep source multiplicities: deduplication changes density weights.
            average = torch.cat([torch.cdist(chunk, candidates).topk(30, largest=False).values.mean(-1)
                                 for chunk in candidates.split(512, dim=1)], 1)
            filtered = candidates[:, average[0] < average.mean() * 2]
            if filtered.shape[1] < output_count:
                raise RuntimeError("SPU density filter left fewer candidates than the output budget; no hidden padding/filter bypass is permitted")
            # Continue the exact CPU RNG state after the last source recovery.
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(final_rng)
                prediction = utils.fps(filtered * scale + center, output_count)
            if self.shared and torch.unique(prediction[0], dim=0).shape[0] != output_count:
                raise RuntimeError("Shared SPU generated insufficient distinct points; repeated coordinates cannot pad the output budget")
            results.append(prediction)
            self.last_diagnostics = {"profile": self.profile, "observed_points": cloud.shape[1],
                                     "groups": self.args.spu_groups, "generated": candidates.shape[1],
                                     "after_density_filter": filtered.shape[1], "deduplicated": False}
        return validate_prediction(torch.cat(results), len(points), output_count)
