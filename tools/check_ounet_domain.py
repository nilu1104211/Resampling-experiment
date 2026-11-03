"""Check one fixed OUNet scale over every planned V3 training observation.

The GT radius bound is rotation invariant and conservative. This checks the
octree domain only; it never clips points or changes the shared normalization.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_resampling.v3_core import corrupt_base, training_condition
from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
from adaptive_resampling.v3_pipeline import read_row


def domain_radius_bound(input_xyz, gt_radius_from_origin):
    """Upper-bound every normalized GT coordinate under any training rotation.

    Shared input_transform uses the observed mean and maximum observed radius.
    ||gt - mean|| <= max(||gt||) + ||mean||, so no GT statistic selects a
    per-sample transform. The slack covers float32 normalization and rotation.
    """
    points = np.asarray(input_xyz, dtype="f8")
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).max())
    if not np.isfinite(points).all() or not math.isfinite(radius) or radius <= 1e-12:
        raise ValueError("Non-finite or degenerate model input during OUNet domain check")
    if not math.isfinite(gt_radius_from_origin):
        raise ValueError("Non-finite GT during OUNet domain check")
    bound = max(1., (gt_radius_from_origin + float(np.linalg.norm(center))) / radius)
    return bound * (1. + 2e-6) + 2e-6


def check_domain(root, *, epochs, octree_scale, training_mode="mixed", train_patches_per_object=32, progress=None):
    if epochs < 1 or not math.isfinite(octree_scale) or not 0 < octree_scale < 1:
        raise ValueError("Use a positive epoch horizon and octree scale in (0, 1)")
    started = time.monotonic()
    ds = AdaptiveResamplingDataset(root, "train", training_mode=training_mode,
                                   include_supervision=False, train_patches_per_object=train_patches_per_object)
    worst = {"radius_bound": 0.}
    checked = violations = 0
    # Read only fields used by observation generation; reference geometry and
    # labels are unnecessary. Process all epochs while the base stays in RAM.
    fields = ("pool_xyz", "input_count", "priority", "hole_rank", "sparse_mask_ref",
              "clean_indices", "noise_distance", "base_id", "gt_xyz")
    try:
        if not len(ds):
            raise ValueError("OUNet domain check requires a nonempty training selection")
        previous = 0
        for file_index, (path, rows) in enumerate(zip(ds.paths, ds.row_maps)):
            with h5py.File(path, "r") as handle:
                if not handle.attrs.get("complete", False):
                    raise ValueError(f"Incomplete V3 shard: {path}")
                for local, row in enumerate(rows):
                    base = read_row(handle, int(row), keys=fields)
                    gt = np.asarray(base["gt_xyz"], dtype="f8")
                    if not np.isfinite(gt).all():
                        raise ValueError(f"Non-finite GT: {base['base_id']}")
                    gt_radius = float(np.linalg.norm(gt, axis=1).max())
                    number = previous + local
                    source_index = ds.source_index(number)
                    for epoch in range(epochs):
                        condition, spec = training_condition(base["base_id"], source_index, epoch, training_mode)
                        coordinates = corrupt_base(base, condition, spec=spec, epoch=epoch, coordinates_only=True)
                        bound = domain_radius_bound(coordinates["input_xyz"], gt_radius)
                        checked += 1
                        violations += bound * octree_scale >= 1
                        if bound > worst["radius_bound"]:
                            worst = {"radius_bound": bound, "base_id": base["base_id"],
                                     "dataset_index": number, "source_index": source_index,
                                     "epoch": epoch, "condition": condition}
            previous = ds.ends[file_index]
            if progress and ((file_index + 1) % 40 == 0 or file_index + 1 == len(ds.paths)):
                progress({"objects": file_index + 1, "total_objects": len(ds.paths),
                          "observations_checked": checked, "worst_radius_bound": worst["radius_bound"],
                          "seconds": round(time.monotonic() - started, 1)})
        return {"status": "passed" if violations == 0 else "failed_conservative_bound",
                "check": "rotation-invariant conservative radius bound, including float32 slack",
                "epochs": epochs, "octree_scale": octree_scale, "training_profile": ds.training_profile,
                "observations_checked": checked, "conservative_violations": int(violations),
                "worst": worst, "worst_scaled_radius_bound": worst["radius_bound"] * octree_scale,
                "scale_for_10_percent_domain_margin": .9 / worst["radius_bound"],
                "seconds": round(time.monotonic() - started, 1)}
    finally:
        ds.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--octree-scale", type=float, required=True)
    parser.add_argument("--training-mode", choices=["mixed", "clean"], default="mixed")
    parser.add_argument("--train-patches-per-object", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = check_domain(args.root, epochs=args.epochs, octree_scale=args.octree_scale,
                          training_mode=args.training_mode, train_patches_per_object=args.train_patches_per_object,
                          progress=lambda row: print(json.dumps({"progress": row}), flush=True))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
