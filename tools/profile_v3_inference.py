"""Compare unchanged PUFM++ V3 inference with reference/fast FPS and batching.

Uses existing weights/schedule when supplied. Random weights require an explicit
flag and only validate execution and numerical behavior, never model quality.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
os.environ["V3_CODE_ROOT"] = str(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "compare-change/Enhanced_PUFM-main"))
sys.path.insert(0, str(REPO / "compare-change"))
import v3_common as common
from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
from adaptive_resampling.v3_pointops import fps_backend_status


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=common.DEFAULT_ROOT)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--schedule", type=Path)
    p.add_argument("--allow-random-weights", action="store_true")
    p.add_argument("--condition", default="A0_normal")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--patch-batch-size", type=int, default=4)
    p.add_argument("--include-batched", action="store_true", help="Also measure optional batching, which can change FPS ties")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=REPO / "outputs/inference_profile.json")
    args = p.parse_args()
    if min(args.samples, args.repeats, args.patch_batch_size) < 1:
        p.error("Sample, repeat and patch batch counts must be positive")
    if not args.checkpoint and not args.allow_random_weights:
        p.error("Supply --checkpoint or explicitly --allow-random-weights")
    if args.checkpoint and not args.schedule:
        p.error("Supply the existing --schedule for trained-weight profiling")
    options = SimpleNamespace(command="schedule", schedule=None, uniform_schedule=False,
                              flow_steps=6, manifold_step=.01, device=args.device,
                              model_name="pufm++", stage="s2", patch_batch_size=1)
    common.set_seed(21)
    model = common.load_adapter(REPO / "compare-change/Enhanced_PUFM-main", options).eval()
    if args.checkpoint:
        common.load_weights(model, args.checkpoint, options)
    if args.schedule:
        saved = torch.load(args.schedule, map_location="cpu", weights_only=False)
        model.times = (np.interp(np.linspace(0, 1, 7), saved["cdf"], saved["grid"]).tolist()
                       if "grid" in saved else np.asarray(saved["t_sched"]).tolist())
        if len(model.times) != 7 or abs(model.times[0]) > 1e-6 or abs(model.times[-1] - 1) > 1e-6 or any(b <= a for a, b in zip(model.times, model.times[1:])):
            raise ValueError("Expected the existing strictly increasing six-step schedule")
    ds = AdaptiveResamplingDataset(args.root, "pu1k_object", condition=args.condition, include_supervision=False)
    try:
        samples = [ds[i] for i in range(min(args.samples, len(ds))) ]
    finally:
        ds.close()
    xs = [torch.from_numpy(s["input_xyz"])[None].to(args.device) for s in samples]
    result = {"random_weights": args.checkpoint is None, "quality_evaluation": False,
              "hardware": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
              "torch": torch.__version__, "checkpoint": str(args.checkpoint), "times": model.times,
              "condition": args.condition, "sample_ids": [s["sample_id"] for s in samples], "runs": {}}
    reference = None
    configurations = [("reference", "torch", 1), ("fast_serial", "cuda", 1)]
    if args.include_batched:
        configurations.append(("fast_batched", "cuda", args.patch_batch_size))
    for name, backend, batch in configurations:
        os.environ["V3_FPS_BACKEND"] = backend
        options.patch_batch_size = batch
        with torch.inference_mode():
            model.predict(xs[0], 8192)  # warm up/JIT outside the measured interval
            synchronize(args.device)
            timings, predictions = [], []
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            for _ in range(args.repeats):
                for x in xs:
                    synchronize(args.device)
                    tick = time.perf_counter()
                    y = model.predict(x, 8192)
                    synchronize(args.device)
                    timings.append(time.perf_counter() - tick)
                    predictions.append(y.cpu())
        stacked = torch.cat(predictions)
        row = {"seconds_per_object": timings, "mean_seconds": float(np.mean(timings)),
               "fps": fps_backend_status(), "patch_batch_size": batch,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None}
        if reference is None:
            reference = stacked
        else:
            row["bitwise_equal"] = torch.equal(reference, stacked)
            row["max_coordinate_difference"] = float((reference - stacked).abs().max())
            nearest = [(cKDTree(a.numpy()).query(b.numpy())[0], cKDTree(b.numpy()).query(a.numpy())[0])
                       for a, b in zip(reference, stacked)]
            row["symmetric_mean_distance"] = float(np.mean([(a.mean() + b.mean()) * .5 for a, b in nearest]))
            row["symmetric_max_distance"] = float(max(max(a.max(), b.max()) for a, b in nearest))
        result["runs"][name] = row
        print(json.dumps({"run": name, **row}), flush=True)
    result["serial_equivalence_passed"] = result["runs"]["fast_serial"]["bitwise_equal"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Profile saved: {args.output}", flush=True)
    if not result["serial_equivalence_passed"]:
        raise RuntimeError("Fast serial output differs from reference; inspect the saved profile before formal evaluation")


if __name__ == "__main__":
    main()
