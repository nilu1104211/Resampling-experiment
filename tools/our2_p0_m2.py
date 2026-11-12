"""Frozen-flow M2 experiment and observation-only initialization diagnostics."""

from collections import defaultdict
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from our2 import run
from adaptive_resampling.v3_dataset import inverse_prediction
from adaptive_resampling.v3_metrics import evaluate_sample
from models.adaptive_ops import fps_ids, gather, knn
from models.gdba import bounded_soft_budget, integer_budget
from models.geometry import geometry

GROUPS = ("S0_midpoint", "S1_uniform", "S2_learned")
EXAMPLE_CONDITIONS = ("A0_normal", "C2_hole_10", "D7_outliers_02", "E2_mixed_outliers")


def timed(device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def sample_hard_budget(midpoint, pools, lower, upper, logits, temperature):
    """The same quota and per-region FPS operations as GDBA.allocate, on shared pools."""
    total = midpoint.shape[1]
    soft = bounded_soft_budget(logits, total, lower, upper, temperature)
    hard = integer_budget(soft, total, lower, upper)
    selected = []
    for b, regions in enumerate(pools):
        points = []
        for j, pool in enumerate(regions):
            count = int(hard[b, j])
            if count:
                points.append(gather(pool[None], fps_ids(pool[None], count))[0])
        selected.append(torch.cat(points))
    return torch.stack(selected), hard


def infer(flow_model, budget_head, observed, divisor):
    """Evaluate all three initializations on identical inputs and one frozen flow."""
    device = observed.device
    midpoint, midpoint_time = timed(device, lambda: flow_model.prepare(observed)["initial"])

    def shared_geometry():
        geo = geometry(observed, divisor)
        return geo, budget_head.candidates(midpoint, geo)

    (geo, (pools, lower, upper)), pool_time = timed(device, shared_geometry)
    logits, head_time = timed(device, lambda: budget_head(geo, None, False, "learned"))
    starts = {GROUPS[0]: midpoint}
    budgets = {}
    times = {GROUPS[0]: {"midpoint_seconds": midpoint_time, "pool_seconds": 0.0,
                        "budget_head_seconds": 0.0, "allocation_seconds": 0.0}}
    for group, scores, learned in ((GROUPS[1], torch.zeros_like(logits), False), (GROUPS[2], logits, True)):
        (starts[group], budgets[group]), elapsed = timed(
            device, lambda: sample_hard_budget(midpoint, pools, lower, upper, scores, budget_head.temperature),
        )
        times[group] = {"midpoint_seconds": midpoint_time, "pool_seconds": pool_time,
                        "budget_head_seconds": head_time if learned else 0.0, "allocation_seconds": elapsed}
    predictions = {}
    for group, start in starts.items():
        pred, elapsed = timed(device, lambda: flow_model.integrate({"initial": start}))
        if pred.shape != midpoint.shape or not torch.isfinite(pred).all():
            raise ValueError(f"{group}: invalid frozen-flow prediction")
        predictions[group] = pred
        times[group]["flow_seconds"] = elapsed
        times[group]["prediction_seconds"] = sum(times[group].values())
    return geo, pools[0], lower[0], upper[0], starts, budgets, predictions, times


def as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def describe_sample(sample, geo, pools, lower, upper, starts, budgets, predictions, times):
    """Compute offline diagnostics after prediction; labels never enter infer()."""
    condition, object_id = sample["condition"], sample["object_id"]
    sample_id = sample["sample_id"]
    anchors, labels = geo["anchors"], as_numpy(geo["labels"][0])
    count = anchors.shape[1]
    gt = inverse_prediction(sample["gt_xyz"], sample)
    gt_tree = cKDTree(gt)
    anchor_xyz = inverse_prediction(as_numpy(anchors[0]), sample)
    anchor_tree = cKDTree(anchor_xyz)
    gt_region = anchor_tree.query(gt)[1]
    ones = np.ones(count, dtype=np.int64)
    bounds_low, bounds_high = as_numpy(lower), as_numpy(upper)

    def assign(points):
        return as_numpy(knn(anchors, points, 1)[0, :, 0])

    def counts(indices):
        return np.bincount(indices, minlength=count)

    def label_counts(name, region):
        mask = sample["supervision"].get(name)
        return np.zeros_like(ones) if mask is None else counts(region[np.asarray(mask, dtype=bool)])

    midpoint_regions = assign(starts[GROUPS[0]])
    n0 = counts(midpoint_regions)
    observed_counts = counts(labels)
    outliers = label_counts("outlier_mask_input", labels)
    noisy = label_counts("noise_mask_input", labels)
    hole_boundary = label_counts("hole_boundary_mask_input", labels)
    sparse = label_counts("sparse_mask_gt", gt_region)
    hole = label_counts("hole_mask_gt", gt_region)
    scales = as_numpy(geo["region_scale"][0, :, 0])
    curvature = as_numpy(geo["region_features"][0, :, 0])
    midpoint_xyz = as_numpy(starts[GROUPS[0]][0])
    observed_xyz = as_numpy(geo["points"][0])
    pre_interpolation_duplicates = []
    for j in range(count):
        original = np.concatenate((midpoint_xyz[midpoint_regions == j], observed_xyz[labels == j]))
        pre_interpolation_duplicates.append(1 - len(np.unique(original, axis=0)) / max(len(original), 1))
    rows, region_rows = [], []
    for group in GROUPS:
        start, pred = starts[group], predictions[group]
        initial_counts, final_counts = counts(assign(start)), counts(assign(pred))
        quota = n0 if group == GROUPS[0] else as_numpy(budgets[group][0])
        if quota.sum() != start.shape[1] or (group != GROUPS[0] and (
            (quota < bounds_low).any() or (quota > bounds_high).any()
        )):
            raise ValueError(f"{group}: infeasible allocation")
        start_xyz = inverse_prediction(as_numpy(start[0]), sample)
        pred_xyz = inverse_prediction(as_numpy(pred[0]), sample)
        init_nn = cKDTree(start_xyz).query(start_xyz, k=2)[0][:, 1]
        start_to_gt = gt_tree.query(start_xyz)[0]
        gt_to_start = cKDTree(start_xyz).query(gt)[0]
        anchor_distances = anchor_tree.query(start_xyz, k=2)[0]
        # Relative gap between first and second anchors as an offline boundary proxy.
        boundary_fraction = float(np.mean(
            anchor_distances[:, 1] - anchor_distances[:, 0] <= 0.05 * anchor_distances[:, 1]
        ))
        n = len(start_xyz)
        fractions = quota / n
        positive = fractions[fractions > 0]
        row = {
            "group": group, "condition": condition, "object_id": object_id, "sample_id": sample_id,
            **evaluate_sample(pred[0], sample), **times[group],
            "pred_to_gt_mean_l2": float(gt_tree.query(pred_xyz)[0].mean()),
            "seed_to_gt_mean_l2": float(start_to_gt.mean()),
            "gt_to_seed_mean_l2": float(gt_to_start.mean()),
            "seed_nn_mean_l2": float(init_nn.mean()),
            "seed_nn_p01_l2": float(np.quantile(init_nn, .01)),
            "seed_nn_p50_l2": float(np.quantile(init_nn, .50)),
            "seed_nn_p99_l2": float(np.quantile(init_nn, .99)),
            "seed_near_duplicate_fraction": float(np.mean(init_nn <= 1e-6)),
            "seed_exact_duplicate_fraction": 1 - len(np.unique(as_numpy(start[0]), axis=0)) / n,
            "seed_boundary_fraction": boundary_fraction,
            "budget_min": int(quota.min()), "budget_max": int(quota.max()),
            "budget_entropy": float(-(positive * np.log(positive)).sum()),
            "budget_variance": float(quota.var()),
            "reallocated_fraction": float(np.abs(quota - n0).sum() / (2 * n)),
            "budget_seed_region_mismatch_fraction": float(np.abs(quota - initial_counts).sum() / (2 * n)),
            "budget_outlier_region_fraction": float(quota[outliers > 0].sum() / n),
            "budget_outlier_region_delta_points": int((quota - n0)[outliers > 0].sum()),
            "budget_sparse_region_fraction": float(quota[sparse > 0].sum() / n),
            "budget_sparse_region_delta_points": int((quota - n0)[sparse > 0].sum()),
        }
        if group != GROUPS[0]:
            row.update(
                budget_lower_saturation_fraction=float((quota == bounds_low).mean()),
                budget_upper_saturation_fraction=float((quota == bounds_high).mean()),
                regions_capacity_below_uniform=int((bounds_high < n // count).sum()),
                exact_uniform_quota=int(np.all(quota == n // count)),
            )
        rows.append(row)
        for j in range(count):
            region_rows.append({
                "group": group, "condition": condition, "object_id": object_id, "sample_id": sample_id,
                "region": j, "anchor_x": float(anchor_xyz[j, 0]), "anchor_y": float(anchor_xyz[j, 1]),
                "anchor_z": float(anchor_xyz[j, 2]), "observed_points": int(observed_counts[j]),
                "curvature_proxy": float(curvature[j]), "local_scale": float(scales[j]),
                "outlier_observed_points": int(outliers[j]), "noisy_observed_points": int(noisy[j]),
                "hole_boundary_observed_points": int(hole_boundary[j]), "sparse_gt_points": int(sparse[j]),
                "hole_gt_points": int(hole[j]), "midpoint_n0": int(n0[j]),
                "budget": int(quota[j]), "budget_minus_n0": int(quota[j] - n0[j]),
                "actual_seed_points": int(initial_counts[j]), "actual_prediction_points": int(final_counts[j]),
                "candidate_unique_points": len(pools[j]), "quota_lower": int(bounds_low[j]),
                "quota_upper": int(bounds_high[j]),
                "midpoint_observed_duplicate_fraction": pre_interpolation_duplicates[j],
            })
    return rows, region_rows


def save_example(path, sample, starts, predictions):
    def canonical(value):
        return inverse_prediction(as_numpy(value[0]), sample)

    np.savez_compressed(
        path, sample_id=sample["sample_id"], object_id=sample["object_id"], condition=sample["condition"],
        observed=inverse_prediction(sample["input_xyz"], sample), target=inverse_prediction(sample["gt_xyz"], sample),
        outlier_mask=np.asarray(sample["supervision"].get("outlier_mask_input",
                                np.zeros(len(sample["input_xyz"]), dtype=bool)), dtype=bool),
        **{f"{group}_seed": canonical(starts[group]) for group in GROUPS},
        **{f"{group}_prediction": canonical(predictions[group]) for group in GROUPS},
    )


@torch.inference_mode()
def diagnose(args, provenance):
    from our2_p0 import (
        CONDITIONS, SPLITS, assert_input, dataset_for, frozen_plan, historical_inputs, input_record, load_model,
        read_json, sample_indices, write_rows,
    )

    flow = load_model(args, "baseline")
    m2 = load_model(args, "m2_hard")
    head, divisor = m2.m2, m2.config["region_divisor"]
    if divisor != 32 or m2.config["allocation_strength"] != 1.0 or m2.config["budget_policy"] != "learned":
        raise ValueError("Unexpected M2 geometry or budget protocol")
    del m2
    model_identity = {"shared_flow_sha256": run.model_hash(flow.flow), "m2_head_sha256": run.model_hash(head)}
    suffix = f"v2_v3_debug_{args.limit_samples}" if args.limit_samples else "v2_v3"
    for split in SPLITS:
        output = args.output / suffix / split
        output.mkdir(parents=True, exist_ok=True)
        marker = output / "identity.json"
        plan = frozen_plan(args, split, output)
        experiment = {
            **provenance, **model_identity, "split": split, "selection_sha256": plan and plan["selection_sha256"],
            "partial_debug_run": bool(args.limit_samples), "limit_samples": args.limit_samples,
        }
        if marker.is_file() and read_json(marker) != experiment:
            raise ValueError(f"V2/V3 output belongs to another experiment: {output}")
        if (output / "summary.json").is_file():
            if not marker.is_file():
                raise ValueError(f"Completed result missing identity: {output}")
            print(f"V2/V3 already completed {split}", flush=True)
            continue
        run.write_json(marker, experiment)
        if plan is not None:
            run.write_json(output / "frozen_plan.json", plan)
        previous_inputs = historical_inputs(args, split)
        if plan is not None:
            example_ids = {item["object_id"] for item in plan["selection"]["objects"][:2]}
            (output / "examples").mkdir(exist_ok=True)
        else:
            example_ids = set()
        rows, regions, records = defaultdict(list), [], []
        warmed = False
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        for condition in CONDITIONS:
            dataset = dataset_for(args, plan, condition)
            try:
                indices = sample_indices(dataset, plan)
                if args.limit_samples:
                    indices = indices[:args.limit_samples]
                for index in indices:
                    sample = dataset[index]
                    observed = torch.as_tensor(sample["input_xyz"], device=args.device)[None].float()
                    expected = 256 if split == "validation" else 2048
                    if tuple(observed.shape) != (1, expected, 3):
                        raise ValueError(f"Unexpected input budget: {split} {observed.shape}")
                    record = input_record(condition, sample, observed)
                    assert_input(record, previous_inputs)
                    records.append(record)
                    if not warmed:
                        infer(flow, head, observed, divisor)
                        warmed = True
                    values = infer(flow, head, observed, divisor)
                    sample_rows, sample_regions = describe_sample(sample, *values)
                    for row in sample_rows:
                        rows[row["group"]].append({**row, "input_sha256": record[-1]})
                    regions.extend(sample_regions)
                    if condition in EXAMPLE_CONDITIONS and sample["object_id"] in example_ids:
                        name = f"{condition}_{run.signature(sample['object_id'])[:10]}.npz"
                        save_example(output / "examples" / name, sample, values[4], values[6])
            finally:
                dataset.close()
            print(f"V2/V3 {split} {condition}: {len(indices)} samples per group", flush=True)
        keys = {(condition, sample_id) for condition, sample_id, _ in records}
        if len(keys) != len(records) or (not args.limit_samples and keys != previous_inputs.keys()):
            raise ValueError(f"V2/V3 sample coverage mismatch: {split}")
        result = {
            "identity": experiment, "groups": {}, "input_selection_sha256": run.signature(records),
            "timing_scope": "synchronized inference component timings, shared preparation attributed per group",
            "peak_cuda_bytes_all_groups": torch.cuda.max_memory_allocated(torch.device(args.device))
            if args.device.startswith("cuda") else None,
        }
        by_condition, objects = [], []
        for group in GROUPS:
            summary, per_object = run.aggregate(rows[group])
            summary["active_parameter_count"] = sum(p.numel() for p in flow.parameters())
            if group == GROUPS[2]:
                summary["active_parameter_count"] += sum(p.numel() for p in head.independent.parameters())
            result["groups"][group] = summary
            by_condition.extend({"group": group, "condition": condition, **values}
                                for condition, values in summary["conditions"].items())
            objects.extend({"group": group, **row} for row in per_object)
        write_rows(output / "per_sample.csv", [row for group in GROUPS for row in rows[group]])
        write_rows(output / "per_object.csv", objects)
        write_rows(output / "per_condition.csv", by_condition)
        write_rows(output / "budgets.csv", regions)
        run.write_json(output / "summary.json", result)
        print(f"V2/V3 completed {split}", flush=True)
