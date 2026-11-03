"""D0-D5 on paired validation patches with the two frozen legacy checkpoints."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo / "our/PUFM-main"), str(repo)]
from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
from models.adaptive_wrapper import AdaptivePUFM, AdaptiveConfig
from models.adaptive_ops import distance_vectors


def load(path, device):
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model = AdaptivePUFM(AdaptiveConfig(**checkpoint["configuration"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def run(args):
    torch.manual_seed(21)
    np.random.seed(21)
    baseline = load(args.baseline, args.device)
    ours = load(args.ours, args.device)
    rows = []
    for condition in args.conditions:
        ds = AdaptiveResamplingDataset(args.root, "validation", condition=condition)
        for index in np.linspace(0, len(ds) - 1, min(args.samples, len(ds)), dtype=int):
            sample = ds[int(index)]
            x, gt = (torch.as_tensor(sample[key], device=args.device)[None] for key in ("input_xyz", "gt_xyz"))
            scale = float(sample["prediction_scale"])
            def record(name, prediction):
                precision, coverage = distance_vectors(prediction, gt)
                rows.append(dict(sample_id=sample["sample_id"], condition=condition, variant=name,
                                 cd_l1=float((precision.mean() + coverage.mean()) * .5) * scale,
                                 gt_to_output=float(coverage.mean()) * scale))

            ref = baseline.prepare(x)
            record("D0_init", ref["initial"])
            record("D0", baseline.integrate(ref))
            # Only change the legacy frontend. D1/D2 deliberately stress the
            # existing joint gain head without its RGR features.
            for use_rgr, allocation, label in [(False, "gain", "D2"), (True, "midpoint", "D3"), (True, "gain", "D4")]:
                ours.config.use_rgr, ours.config.allocation = use_rgr, allocation
                state = ours.prepare(x)
                record("D1" if label == "D2" else label + "_init", state["initial"])
                record(label, baseline.integrate(state))
                if label == "D4":
                    record("D5", ours.integrate(state))
        ds.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for name in dict.fromkeys(row["variant"] for row in rows):
        print(name, np.mean([r["cd_l1"] for r in rows if r["variant"] == name]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "baseline", "ours", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--conditions", nargs="+", default=["A0_normal", "C2_hole_10"])
    run(parser.parse_args())
