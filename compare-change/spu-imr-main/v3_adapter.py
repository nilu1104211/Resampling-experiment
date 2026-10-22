"""Input-only self-supervised SPU-IMR on the common V3 split."""
from pathlib import Path
import yaml
import torch
from torch import nn
from model4spu_imr import Network_Model
from v3_common import fps, patch_predict


class Adapter(nn.Module):
    compatibility_keys = ("spu_groups",)
    training_description = "Original SPU-IMR iterative masked recovery on actual input only (including its degradation); no clean GT loss; exact transport backend"
    inference_description = "Repeated native masked recovery; source density filter; FPS to 4x; normalized input patches for objects"

    def __init__(self, args):
        super().__init__()
        self.args = args
        cfg = yaml.safe_load((Path(__file__).parent / "cfgs/setting.yaml").read_text(encoding="utf-8"))
        cfg["data"]["up_rate"] = cfg["transformer"]["up_rate"] = 4
        # Retain the original ratio: 1024 observations / 256 centers = 4.
        cfg["model"]["num_group"] = args.spu_groups
        cfg["model"]["gen_points"] = 1024
        if not 4 <= args.spu_groups <= 256:
            raise ValueError("SPU group count must be between 4 and 256 for 256-point V3 patches")
        if args.spu_passes * args.spu_groups * cfg["model"]["group_size"] < 1024:
            raise ValueError("Not enough recovery candidates for 1024 outputs; increase --spu-passes")
        self.model = Network_Model(cfg)

    def training_loss(self, points, gt):
        # GT is deliberately unused: this is the native self-supervised comparison.
        if points.shape[1:] != (256, 3):
            raise ValueError("SPU V3 training expects 256 observed points")
        return self.model(points, points.device).mean()

    def predict_patch(self, points):
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
        return patch_predict(points, output_count, self.predict_patch)
