"""Evaluate ID-paired predictions with object-level aggregation.

NPZ fields: sample_id (Unicode array), prediction_xyz (S x M x 3).
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_resampling.v3_dataset import DEFAULT_ROOT, AdaptiveResamplingDataset
from adaptive_resampling.v3_metrics import evaluate_sample


def evaluate(path, root, split, condition, output, *, coordinates="model", p2f=False, allow_subset=False):
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["sample_id"].astype(str)
        predictions = archive["prediction_xyz"]
    if ids.ndim != 1 or predictions.ndim != 3 or len(ids) != len(predictions):
        raise ValueError("Expected sample_id[S] and prediction_xyz[S,M,3]")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate prediction sample IDs")
    positions = {name: i for i, name in enumerate(ids)}
    dataset = AdaptiveResamplingDataset(root, split, condition=condition, include_supervision=True, include_reference_metadata=p2f)
    rows, seen = [], set()
    try:
        for i in range(len(dataset)):
            sample = dataset[i]
            sample_id = sample["sample_id"]
            if sample_id not in positions:
                if allow_subset:
                    continue
                raise ValueError(f"Missing prediction: {sample_id}")
            metrics = evaluate_sample(predictions[positions[sample_id]], sample, coordinates=coordinates, include_p2f=p2f)
            rows.append(dict(sample_id=sample_id, object_id=sample["object_id"], **metrics))
            seen.add(sample_id)
            if len(rows) % 100 == 0:
                print(f"Evaluated {len(rows)} predictions", flush=True)
    finally:
        dataset.close()
    if seen != set(positions):
        raise ValueError("Prediction file includes IDs outside the requested split/condition")
    if not rows:
        raise ValueError("No predictions were evaluated")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    columns = sorted(set().union(*(row.keys() for row in rows)) - {"sample_id", "object_id"})
    with (destination / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "object_id"] + columns)
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["object_id"]].append(row)
    objects = []
    for object_id, members in sorted(grouped.items()):
        result = {"object_id": object_id, "patches": len(members)}
        for column in columns:
            values = [member[column] for member in members if column in member]
            if values:
                result[column] = float(np.mean(values))
        objects.append(result)
    with (destination / "per_object.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["object_id", "patches"] + columns)
        writer.writeheader()
        writer.writerows(objects)
    rng = np.random.default_rng(20261021)
    intervals = {}
    for column in columns:
        values = np.asarray([item[column] for item in objects if column in item])
        bootstrap = values[rng.integers(len(values), size=(1000, len(values)))].mean(1)
        intervals[column] = {"mean_over_objects": float(values.mean()),
                             "valid_objects": len(values),
                             "std_over_objects": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                             "bootstrap_95CI": np.percentile(bootstrap, [2.5, 97.5]).tolist()}
    summary = {"split": split, "condition": condition, "samples": len(rows), "objects": len(objects),
               "partial_debug_evaluation": allow_subset, "prediction_coordinates": coordinates,
               "aggregation": "mean patches within each object, then mean over objects",
               "metrics": intervals}
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--split", choices=["validation", "pu1k_patch", "modelnet40_patch", "pu1k_object"], required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coordinates", choices=["model", "canonical", "raw"], default="model")
    parser.add_argument("--p2f", action="store_true")
    parser.add_argument("--allow-subset", action="store_true", help="Debug only; never a full benchmark result")
    arguments = parser.parse_args()
    result = evaluate(arguments.predictions, arguments.root, arguments.split, arguments.condition,
                      arguments.output, coordinates=arguments.coordinates, p2f=arguments.p2f,
                      allow_subset=arguments.allow_subset)
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"}, ensure_ascii=False, indent=2))
