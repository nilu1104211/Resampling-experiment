"""Measure only exact CPU matching on real V3 PUFM++ S1 initialization costs.

Data loading, midpoint initialization, GPU transfers, network execution and
checkpoint saves are excluded. Use a normal short training run for total speed.
"""
from pathlib import Path
import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import scipy
import torch

ROOT = Path(os.environ.get("V3_CODE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
COMPARISON = ROOT / "compare-change"
if not COMPARISON.is_dir():
    COMPARISON = ROOT / "network_adaptation/compare-change"
sys.path.insert(0, str(COMPARISON))
from adaptive_resampling.v3_dataset import DEFAULT_ROOT, AdaptiveResamplingDataset
from adaptive_resampling.v3_training import DEFAULT_TRAIN_PATCHES_PER_OBJECT
from adaptive_resampling.v3_transport import batched_assignment_indices, configure_assignment_workers
from v3_common import midpoint, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--train-patches-per-object", type=int, default=DEFAULT_TRAIN_PATCHES_PER_OBJECT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.repeats, *args.threads) < 1:
        parser.error("Counts must be positive")
    threads = list(dict.fromkeys([1, *args.threads]))
    set_seed(args.seed)
    ds = AdaptiveResamplingDataset(args.root, "train", include_supervision=False, augment_rotation=True,
                                   train_patches_per_object=args.train_patches_per_object)
    rng = np.random.default_rng(args.seed)
    count = args.batch_size * args.batches
    if count > len(ds):
        parser.error("Not enough distinct training samples for the requested batches")
    sample_indices = rng.choice(len(ds), count, replace=False).tolist()
    costs, sample_ids = [], []
    try:
        for start in range(0, count, args.batch_size):
            samples = [ds[i] for i in sample_indices[start:start + args.batch_size]]
            sample_ids.extend(s["sample_id"] for s in samples)
            x = torch.from_numpy(np.stack([s["input_xyz"] for s in samples])).to(args.device)
            gt = torch.from_numpy(np.stack([s["gt_xyz"] for s in samples])).to(args.device)
            with torch.no_grad():
                x0 = midpoint(x) + .02 * torch.randn_like(gt)
                costs.append(torch.cdist(x0, gt).square().cpu().numpy())
    finally:
        ds.close()

    configure_assignment_workers(1)
    expected = [batched_assignment_indices(cost) for cost in costs]
    timings = {workers: [] for workers in threads}
    try:
        # Interleave configurations to reduce bias from CPU temperature/clock drift.
        for repeat in range(args.repeats):
            order = threads if repeat % 2 == 0 else list(reversed(threads))
            for workers in order:
                configure_assignment_workers(workers)
                # Exclude thread startup and verify every real-data batch before timing.
                for cost, reference in zip(costs, expected):
                    np.testing.assert_array_equal(batched_assignment_indices(cost), reference)
                start = time.perf_counter()
                actual = [batched_assignment_indices(cost) for cost in costs]
                timings[workers].append((time.perf_counter() - start) / len(costs))
                for indices, reference in zip(actual, expected):
                    np.testing.assert_array_equal(indices, reference)
    finally:
        configure_assignment_workers(1)
    serial = statistics.median(timings[1])
    result = {
        "scope": "CPU assignment solves only; not end-to-end training",
        "exact_indices_equal": True,
        "python": platform.python_version(), "torch": torch.__version__, "scipy": scipy.__version__,
        "logical_cpu_count": os.cpu_count(), "cost_device": args.device,
        "gpu": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else None,
        "batch_size": args.batch_size, "points": int(costs[0].shape[1]),
        "batches": args.batches, "repeats": args.repeats,
        "training_profile": ds.training_profile,
        "sample_indices": sample_indices, "sample_ids": sample_ids,
        "results": [{"assignment_workers": workers,
                     "median_ms_per_batch": statistics.median(timings[workers]) * 1000,
                     "speedup_vs_serial": serial / statistics.median(timings[workers]),
                     "seconds_per_batch": timings[workers]} for workers in threads],
    }
    report = json.dumps(result, indent=2, ensure_ascii=False)
    print(report, flush=True)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
