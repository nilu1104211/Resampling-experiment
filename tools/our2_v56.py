"""Run the first, inference-only V5/V6 experiments on the pinned Our2 checkpoints."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

from our2_p0 import (
    CONDITIONS, REPO, REGRESSION_SPLITS, SPLITS, assert_input, dataset_for, frozen_plan,
    historical_inputs, historical_rows, identity, input_record, load_model, read_json,
    read_rows, run, sample_indices, write_rows,
)

import numpy as np
from scipy.spatial import cKDTree
import torch

from adaptive_resampling.v3_dataset import inverse_prediction
from adaptive_resampling.v3_metrics import evaluate_sample
from models.geometry import geometry
from models.utils import midpoint_interpolate
from our2_p0_m2 import EXAMPLE_CONDITIONS, as_numpy, timed
from our2_v6_seeds import LEARNED, make_pool, reallocate

M1_GROUPS = ("M1-full", "M1-r", "M1-h", "M1-off", "M1-no-move")
M1_SWITCHES = {"M1-full": (True, True), "M1-r": (True, False),
               "M1-h": (False, True), "M1-off": (False, False)}
QUALITY_CONDITIONS = ("macro", "A0_normal", "D7_outliers_02", "E2_mixed_outliers")
F_TOLERANCE, CD_TOLERANCE, HD_TOLERANCE = .01, .01, .02
MATCH_MINIMUM = .95
M1_HOLE_EFFECT = .01


def midpoint(observed):
    return midpoint_interpolate(SimpleNamespace(up_rate=4), observed.transpose(1, 2).contiguous()).transpose(1, 2)


def model_pair(args, experiment):
    if experiment == "v5":
        model = load_model(args, "m1_refine")
        if not (model.config["use_reliability"] and model.config["use_hole"]):
            raise ValueError("Expected the original M1-full checkpoint with both paths enabled")
        return model, None
    model = load_model(args, "baseline")
    m2 = load_model(args, "m2_hard")
    if m2.config["region_divisor"] != 32 or m2.config["budget_policy"] != "learned":
        raise ValueError("Unexpected source M2 budget protocol")
    return model, m2.m2


def m1_history(args, split):
    if split == "validation":
        stage = args.training_run / "m1_refine"
        epoch = read_json(stage / "best.json")["epoch"]
        path = stage / "validation" / f"epoch_{epoch:04d}" / "per_sample.csv"
    else:
        path = args.previous_eval / split / "m1_refine" / "per_sample.csv"
    return {(r["condition"], r["sample_id"]): r["prediction_sha256"] for r in read_rows(path)}


def infer(model, head, observed, groups, requested=None):
    """Forward-only path. No sample, GT, or degradation metadata enters this function."""
    device = observed.device
    original, midpoint_seconds = timed(device, lambda: midpoint(observed))
    if head is None:
        def m1_features():
            geo = geometry(observed, model.config["region_divisor"])
            return geo, model.m1(geo)
        (geo, scores), shared_seconds = timed(device, m1_features)
        context = (geo, scores)
    elif any(group != "D0" for group in groups):
        context, shared_seconds = timed(device, lambda: make_pool(original, observed, head, 32))
    else:
        context, shared_seconds = None, 0.0

    results = {}
    for group in groups:
        if head is None:
            if group == "M1-no-move":
                seeds, diagnostic, seconds = original, {}, 0.0
            else:
                flags = M1_SWITCHES[group]
                (seeds, _), seconds = timed(device, lambda: model.m1.refine(original, *context, *flags))
                diagnostic = {}
        else:
            pair = group[:-1] + "L"
            moves = None
            if group.endswith("U"):
                moves = results[pair][2]["moved"] if pair in results else requested[pair]
            (seeds, diagnostic), seconds = timed(device, lambda: reallocate(original, context, group, moves))
            diagnostic = diagnostic or {}
        # The equal-geometry M1 controls can share an output without changing the experiment.
        if group == "M1-no-move" and "M1-off" in results and torch.equal(seeds, results["M1-off"][0]):
            prediction, flow_seconds = results["M1-off"][1], results["M1-off"][3]["flow_seconds"]
        else:
            prediction, flow_seconds = timed(device, lambda: model.integrate({"initial": seeds}))
        if prediction.shape != original.shape or not torch.isfinite(prediction).all():
            raise ValueError(f"{group}: invalid fixed-budget prediction")
        initialization = midpoint_seconds + seconds + (shared_seconds if group != "D0" else 0.0)
        results[group] = seeds, prediction, diagnostic, {
            "midpoint_seconds": midpoint_seconds, "initialization_seconds": initialization,
            "flow_seconds": flow_seconds, "prediction_seconds": initialization + flow_seconds,
        }
    return original, context, results


def check_identity(args, experiment, model, head):
    path = args.output / experiment / "identity_check.json"
    if path.is_file():
        return
    rows = []
    groups = ("D0",) if experiment == "v6" else M1_GROUPS
    for split in SPLITS:
        plan = frozen_plan(args, split, args.output / experiment)
        history = historical_rows(args, split)
        m1_hashes = m1_history(args, split) if experiment == "v5" else None
        for condition in CONDITIONS:
            ds = dataset_for(args, plan, condition)
            try:
                sample = ds[sample_indices(ds, plan)[0]]
                observed = torch.as_tensor(sample["input_xyz"], device=args.device)[None].float()
                record = input_record(condition, sample, observed)
                past = history[tuple(record[:2])]
                if past["input_sha256"] != record[-1]:
                    raise ValueError(f"V5/V6 identity input changed: {record}")
                original, _, values = infer(model, head, observed, groups)
                output_points = 1024 if split == "validation" else 8192
                if original.shape != (1, output_points, 3):
                    raise ValueError(f"Identity check expected {output_points} output points")
                reference = "D0" if experiment == "v6" else "M1-full"
                torch.testing.assert_close(values[reference][1], model(observed), atol=0, rtol=0)
                expected = past["prediction_sha256"] if m1_hashes is None else m1_hashes[tuple(record[:2])]
                if run.tensor_hash(values[reference][1]) != expected:
                    raise ValueError(f"{reference} does not reproduce the archived prediction: {record[:2]}")
                if experiment == "v6":
                    assert values["D0"][0] is original
                    if len(np.unique(as_numpy(original[0]), axis=0)) != output_points:
                        raise ValueError("D0 original seeds contain duplicates")
                else:
                    for group in ("M1-off", "M1-no-move"):
                        torch.testing.assert_close(values[group][0], original, atol=0, rtol=0)
                    torch.testing.assert_close(values["M1-off"][1], model.integrate({"initial": original}),
                                               atol=0, rtol=0)
                rows.append({
                    "split": split, "condition": condition, "sample_id": sample["sample_id"],
                    "input_sha256": record[-1], "output_points": output_points,
                    "groups": {g: {"prediction_sha256": run.tensor_hash(v[1]),
                                   "changed_seed_count": int((v[0] != original).any(-1).sum()),
                                   "seed_shift_max": float((v[0] - original).norm(dim=-1).max())}
                               for g, v in values.items()},
                    "reference_prediction_sha256": expected,
                })
            finally:
                ds.close()
    run.write_json(path, {"status": "passed", "cases": rows})
    print(f"{experiment} identity passed ({len(rows)} cases)", flush=True)


def m1_offline(sample, context):
    geo, scores = context
    r, h = as_numpy(scores["r"][0]), as_numpy(scores["h"][0])
    target = sample["gt_xyz"]
    labels = sample["supervision"]
    distance, index = cKDTree(target).query(as_numpy(geo["candidates"][0]))
    hole = np.asarray(labels.get("hole_mask_gt", np.zeros(len(target))), dtype=bool)
    positive = hole[index] & (distance <= as_numpy(geo["region_scale"][0, :, 0]))
    detected = h >= .5
    stats = {"r_p10": float(np.quantile(r, .1)), "r_p50": float(np.quantile(r, .5)),
             "r_p90": float(np.quantile(r, .9)), "h_p10": float(np.quantile(h, .1)),
             "h_p50": float(np.quantile(h, .5)), "h_p90": float(np.quantile(h, .9)),
             "h_candidates": len(h), "h_tp": int((positive & detected).sum()),
             "h_fp": int((~positive & detected).sum()), "h_positive": int(positive.sum())}
    if positive.any():
        ordered = positive[np.argsort(-h, kind="stable")]
        stats["h_candidate_ap"] = float((np.cumsum(ordered) / np.arange(1, len(h) + 1))[ordered].mean())
        stats["h_candidate_recall_at_05"] = stats["h_tp"] / stats["h_positive"]
    if (~positive).any():
        stats["h_candidate_false_positive_rate_at_05"] = stats["h_fp"] / int((~positive).sum())
    outliers = np.asarray(labels.get("outlier_mask_input", np.zeros(len(r))), dtype=bool)
    if outliers.any():
        stats["r_outlier_mean"] = float(r[outliers].mean())
    if (~outliers).any():
        stats["r_non_outlier_mean"] = float(r[~outliers].mean())
    return stats


def score_sample(sample, original, context, results, experiment, record):
    target = inverse_prediction(sample["gt_xyz"], sample)
    gt_tree = cKDTree(target)
    original_xyz = inverse_prediction(as_numpy(original[0]), sample)
    hole = np.asarray(sample["supervision"].get("hole_mask_gt", np.zeros(len(target))), dtype=bool)
    near_hole = hole[gt_tree.query(original_xyz)[1]]
    auxiliary = m1_offline(sample, context) if experiment == "v5" else {}
    off = results.get("M1-off")
    if off is not None:
        off_prediction = inverse_prediction(as_numpy(off[1][0]), sample)
        output_near_hole = hole[gt_tree.query(off_prediction)[1]]
    output, regions = [], []
    for group, (seeds, pred, diag, timing) in results.items():
        xyz = inverse_prediction(as_numpy(seeds[0]), sample)
        nearest = cKDTree(xyz)
        nn = nearest.query(xyz, k=2)[0][:, 1]
        metrics = evaluate_sample(pred[0], sample)
        changed = int(np.any(as_numpy(seeds[0]) != as_numpy(original[0]), axis=1).sum())
        key = {"group": group, "condition": record[0], "sample_id": record[1], "object_id": sample["object_id"]}
        row = {
            **key, "input_sha256": record[-1], "prediction_sha256": run.tensor_hash(pred),
            **metrics, **timing, **auxiliary,
            "pred_to_gt_mean_l2": 2 * metrics["cd_l1"] - metrics["gt_coverage_mean_l2"],
            "seed_to_gt_mean_l2": float(gt_tree.query(xyz)[0].mean()),
            "gt_to_seed_mean_l2": float(nearest.query(target)[0].mean()),
            "seed_nn_p01": float(np.quantile(nn, .01)), "seed_nn_p50": float(np.quantile(nn, .50)),
            "seed_nn_p99": float(np.quantile(nn, .99)), "seed_duplicate_fraction": float((nn == 0).mean()),
            "seed_changed": changed, "seed_changed_fraction": changed / len(xyz),
        }
        if experiment == "v5":
            movement = np.linalg.norm(xyz - original_xyz, axis=1)
            row.update(seed_shift_mean=float(movement.mean()), seed_shift_p95=float(np.quantile(movement, .95)),
                       seed_shift_max=float(movement.max()))
            if near_hole.any():
                row["seed_shift_near_hole_mean"] = float(movement[near_hole].mean())
            if (~near_hole).any():
                row["seed_shift_elsewhere_mean"] = float(movement[~near_hole].mean())
            if off is not None:
                shift = np.linalg.norm(inverse_prediction(as_numpy(pred[0]), sample) - off_prediction, axis=1)
                row["prediction_shift_vs_off_mean"] = float(shift.mean())
                if output_near_hole.any():
                    row["prediction_shift_vs_off_near_hole_mean"] = float(shift[output_near_hole].mean())
                if (~output_near_hole).any():
                    row["prediction_shift_vs_off_elsewhere_mean"] = float(shift[~output_near_hole].mean())
        else:
            row.update(rho_B=0.0, requested_moves=0, budget_moves=0, retained_fraction=1.0)
            if group != "D0":
                budget = diag["budget"]
                if changed != diag["changed"] or diag["changed"] != diag["moved"]:
                    raise ValueError("V6 coordinate replacement and budget transfers disagree")
                fraction = budget[budget > 0] / len(xyz)
                row.update(rho_B=diag["moved"] / len(xyz), requested_moves=diag["requested"],
                           budget_moves=diag["moved"], retained_fraction=1 - changed / len(xyz),
                           budget_entropy=float(-(fraction * np.log(fraction)).sum()),
                           budget_sha256=run.signature(budget.tolist()), rho_max=diag["rho_max"],
                           matched_moves=int(diag["moved"] == diag["requested"]) if group.endswith("U") else None)
                if diag["lambda"] is not None:
                    row["lambda"] = diag["lambda"]
                for j, n0 in enumerate(context.n0):
                    regions.append({**key, "region": j, "n0": int(n0), "target": float(diag["target"][j]),
                                    "budget": int(budget[j]), "lower": int(context.lower[j]),
                                    "upper": int(context.upper[j]), "new_candidates": len(context.novel[j])})
        output.append(row)
    return output, regions


def save_example(path, sample, original, results):
    def canonical(tensor):
        return inverse_prediction(as_numpy(tensor[0]), sample)
    payload = {"sample_id": sample["sample_id"], "object_id": sample["object_id"],
               "condition": sample["condition"], "observed": inverse_prediction(sample["input_xyz"], sample),
               "target": inverse_prediction(sample["gt_xyz"], sample), "original_seed": canonical(original),
               "outlier_mask": np.asarray(sample["supervision"].get("outlier_mask_input",
                                           np.zeros(len(sample["input_xyz"]))), dtype=bool)}
    for group, (seed, pred, _, _) in results.items():
        payload[f"{group}_seed"], payload[f"{group}_prediction"] = canonical(seed), canonical(pred)
    np.savez_compressed(path, **payload)


def evaluate(args, model, head, experiment, split, output, groups, matched=None, examples_only=False):
    marker = output / "summary.json"
    if marker.is_file():
        saved = read_json(marker)
        if saved["requested_groups"] != list(groups) or saved["examples_only"] != examples_only:
            raise ValueError(f"Result already belongs to a different group matrix: {output}")
        return saved
    output.mkdir(parents=True, exist_ok=True)
    plan = frozen_plan(args, split, output)
    if plan is not None:
        run.write_json(output / "frozen_plan.json", plan)
        (output / "examples").mkdir(exist_ok=True)
    examples = {item["object_id"] for item in plan["selection"]["objects"][:2]} if plan else set()
    reference = historical_inputs(args, split)
    records, rows, budgets = [], defaultdict(list), []
    warmed = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for condition in (EXAMPLE_CONDITIONS if examples_only else CONDITIONS):
        ds = dataset_for(args, plan, condition)
        try:
            indices = sample_indices(ds, plan)
            if args.limit_samples:
                indices = indices[:args.limit_samples]
            for index in indices:
                sample = ds[index]
                example = condition in EXAMPLE_CONDITIONS and sample["object_id"] in examples
                if examples_only and not example:
                    continue
                observed = torch.as_tensor(sample["input_xyz"], device=device)[None].float()
                record = input_record(condition, sample, observed)
                assert_input(record, reference)
                moves = matched[sample["sample_id"]] if matched is not None else None
                if not warmed:
                    infer(model, head, observed, groups, moves)
                    warmed = True
                original, context, values = infer(model, head, observed, groups, moves)
                records.append(record)
                if not examples_only:
                    sample_rows, region_rows = score_sample(sample, original, context, values, experiment, record)
                    for row in sample_rows:
                        rows[row["group"]].append(row)
                    budgets.extend(region_rows)
                if example:
                    name = f"{condition}_{run.signature(sample['object_id'])[:10]}.npz"
                    save_example(output / "examples" / name, sample, original, values)
        finally:
            ds.close()
        print(f"{experiment} {output.parent.name}/{split} {condition} completed", flush=True)
    keys = {(condition, sample_id) for condition, sample_id, _ in records}
    if len(keys) != len(records) or (not examples_only and not args.limit_samples and keys != reference.keys()):
        raise ValueError(f"V5/V6 frozen sample coverage differs: {output}")
    result = {"requested_groups": list(groups), "examples_only": examples_only, "sample_count": len(records),
              "partial_debug_run": bool(args.limit_samples), "input_selection_sha256": run.signature(records),
              "timing_scope": "synchronized model timings; shared geometry attributed to each active group",
              "peak_cuda_bytes_shared_run": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "groups": {}}
    objects, conditions = [], []
    for group, group_rows in rows.items():
        summary, object_rows = run.aggregate(group_rows)
        result["groups"][group] = summary
        objects.extend({"group": group, **r} for r in object_rows)
        conditions.extend({"group": group, "condition": name, **r} for name, r in summary["conditions"].items())
    if rows:
        write_rows(output / "per_sample.csv", [r for group in groups for r in rows[group]])
        write_rows(output / "per_object.csv", objects)
        write_rows(output / "per_condition.csv", conditions)
    if budgets:
        write_rows(output / "budgets.csv", budgets)
    run.write_json(marker, result)
    return result


def quality_failures(group, control):
    failures = []
    checks = (("f_score", -F_TOLERANCE, True), ("cd_l1", CD_TOLERANCE, False),
              ("hausdorff_l2", HD_TOLERANCE, False))
    for condition in QUALITY_CONDITIONS:
        a = group["macro"] if condition == "macro" else group["conditions"][condition]
        b = control["macro"] if condition == "macro" else control["conditions"][condition]
        for key, tolerance, higher in checks:
            limit = (1 + tolerance) * b[key]
            if (a[key] < limit if higher else a[key] > limit):
                failures.append({"condition": condition, "metric": key, "value": a[key], "limit": limit})
    return failures


def matched_report(learned_rows, uniform_rows, group):
    pair = group[:-1] + "U"
    left = {r["sample_id"]: r for r in learned_rows if r["group"] == group}
    right = {r["sample_id"]: r for r in uniform_rows if r["group"] == pair}
    if not left or left.keys() != right.keys():
        raise ValueError(f"Matched controls have different sample coverage: {group}")
    matched = [key for key, r in left.items()
               if r["budget_moves"] == right[key]["budget_moves"] and r["seed_changed"] == right[key]["seed_changed"]]
    fields = ("cd_l1", "f_score", "hausdorff_l2", "sparse_coverage_mean_l2", "hole_coverage_mean_l2")
    usable = [key for key in matched if int(left[key]["budget_moves"]) > 0]
    differences = [{"group": group, "condition": left[key]["condition"], "object_id": left[key]["object_id"],
                    "sample_id": key, "budget_moves": int(left[key]["budget_moves"]),
                    **{f"learned_minus_uniform_{k}": float(left[key][k]) - float(right[key][k])
                       for k in fields if left[key].get(k) and right[key].get(k)}} for key in usable]
    coverage = {}
    for condition in CONDITIONS:
        total = sum(r["condition"] == condition for r in left.values())
        coverage[condition] = {"samples": total, "matches": sum(left[key]["condition"] == condition for key in matched),
                               "nonzero_matches": sum(left[key]["condition"] == condition for key in usable)}
    return {"exact_matches": len(matched), "samples": len(left), "nonzero_matches": len(usable),
            "match_rate": len(matched) / len(left), "conditions": coverage}, differences


def v6(args):
    model, head = model_pair(args, "v6")
    check_identity(args, "v6", model, head)
    base = args.output / "v6"
    path = base / "patch_learned" / "validation"
    learned = evaluate(args, model, head, "v6", "validation", path, ("D0", *LEARNED))["groups"]
    samples = read_rows(path / "per_sample.csv")
    budgets = {g: tuple(r["budget_sha256"] for r in samples if r["group"] == g) for g in LEARNED}
    eligible, duplicate_of, seen, screens = [], {}, {}, {}
    for group in LEARNED:
        failures = quality_failures(learned[group], learned["D0"])
        rho = learned[group]["macro"]["rho_B"]
        screens[group] = {"macro_rho_B": rho, "quality_failures": failures}
        if budgets[group] in seen:
            duplicate_of[group] = seen[budgets[group]]
        elif rho > 0 and not failures:
            eligible.append(group)
        seen.setdefault(budgets[group], group)
    ranked = sorted(eligible, key=lambda g: (-learned[g]["macro"]["f_score"], learned[g]["macro"]["cd_l1"]))
    chosen = ranked[:2]
    exact, promoted = {}, []
    if chosen:
        uniform = tuple(g[:-1] + "U" for g in chosen)
        requested = defaultdict(dict)
        for row in samples:
            if row["group"] in chosen:
                requested[row["sample_id"]][row["group"]] = int(row["budget_moves"])
        upath = base / "patch_uniform" / "validation"
        evaluate(args, model, head, "v6", "validation", upath, uniform, requested)
        uniform_samples = read_rows(upath / "per_sample.csv")
        differences = []
        for group in chosen:
            exact[group], rows = matched_report(samples, uniform_samples, group)
            differences.extend(rows)
            if exact[group]["match_rate"] >= MATCH_MINIMUM and exact[group]["nonzero_matches"] > 0:
                promoted.append(group)
        if differences:
            write_rows(base / "patch_matched_deltas.csv", differences)
    decision = {"patch_screen": screens, "selected_for_uniform": chosen,
                "exact_matching": exact, "duplicate_budget_groups": duplicate_of,
                "promoted_to_whole": [] if args.limit_samples else promoted}
    run.write_json(base / "decision.json", decision)
    for split in REGRESSION_SPLITS:
        if args.limit_samples:
            break
        if promoted:
            groups = ("D0", *[g for pair in promoted for g in (pair, pair[:-1] + "U")])
            path = base / "regression" / split
            evaluate(args, model, head, "v6", split, path, groups)
            whole_rows = read_rows(path / "per_sample.csv")
            matching, all_deltas = {}, []
            for group in promoted:
                matching[group], rows = matched_report(whole_rows, whole_rows, group)
                all_deltas.extend(rows)
            run.write_json(path / "matching.json", matching)
            if all_deltas:
                write_rows(path / "matched_deltas.csv", all_deltas)
        else:
            evaluate(args, model, head, "v6", split, base / "fixed_examples" / split,
                     ("D0", "D1-L", "D1-U"), examples_only=True)
    print("V6 DONE", json.dumps(decision, ensure_ascii=False), flush=True)


def v5(args):
    model, head = model_pair(args, "v5")
    check_identity(args, "v5", model, head)
    base = args.output / "v5"
    result = evaluate(args, model, head, "v5", "validation", base / "patch" / "validation", M1_GROUPS)
    groups = result["groups"]
    key = "hole_coverage_mean_l2"
    full = groups["M1-full"]["macro"][key]
    effects = {g: groups[g]["macro"][key] / full - 1 for g in ("M1-r", "M1-h", "M1-off")}
    promote = not args.limit_samples and any(abs(x) >= M1_HOLE_EFFECT for x in effects.values())
    decision = {"patch_relative_hole_distance_vs_full": effects, "promoted_to_whole": promote}
    run.write_json(base / "decision.json", decision)
    if not args.limit_samples:
        for split in REGRESSION_SPLITS:
            mode, matrix = ("regression", M1_GROUPS) if promote else ("fixed_examples", ("M1-full", "M1-off"))
            evaluate(args, model, head, "v5", split, base / mode / split, matrix, examples_only=not promote)
    print("V5 DONE", json.dumps(decision, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("all", "v6", "v5"))
    parser.add_argument("--root", type=Path, default=REPO / "adaptive_resampling_data_v3")
    parser.add_argument("--whole-data-root", type=Path, default=REPO / "modelnet40_object80_seed21")
    parser.add_argument("--baseline-checkpoint", type=Path,
                        default=REPO / "server_runs/patch32/pufm_common_b8_w0_a8_fused/epoch_0060.pt")
    parser.add_argument("--training-run", type=Path, default=REPO / "server_runs/our2_modules_from60_seed21_b8")
    parser.add_argument("--previous-eval", type=Path, default=REPO / "server_runs/our2_whole_from60_seed21")
    parser.add_argument("--p0-results", type=Path, default=REPO / "server_runs/our2_p0_from60_seed21")
    parser.add_argument("--output", type=Path, default=REPO / "server_runs/our2_v56_from60_seed21")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps-backend", choices=("cuda", "torch"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0, help="Patch-only debug samples per condition")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.limit_samples < 0:
        parser.error("Invalid CPU threads or debug sample count")
    for key in ("root", "whole_data_root", "baseline_checkpoint", "training_run", "previous_eval", "p0_results", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.is_relative_to(REPO / "our2"):
        parser.error("Keep outputs outside the trained model source tree")
    if args.limit_samples:
        args.output /= f"debug_{args.limit_samples}"
    run.configure_runtime(args)
    run.seed_all(21)
    provenance = identity(args)
    v0 = read_json(args.p0_results / "v0.json")
    if v0["status"] != "passed" or v0["identity"] != provenance:
        raise ValueError("V0 baseline/data/code/checkpoint identity differs")
    provenance.update(v56_source_sha256={name: run.file_hash(Path(__file__).with_name(name))
                                        for name in ("our2_v56.py", "our2_v6_seeds.py")},
                      paths={name: str(getattr(args, name)) for name in (
                          "root", "whole_data_root", "baseline_checkpoint", "training_run", "previous_eval", "p0_results")},
                      seed=21, learned_groups={g: list(x) for g, x in LEARNED.items()}, limit_samples=args.limit_samples,
                      gate={"f": F_TOLERANCE, "cd": CD_TOLERANCE, "hd": HD_TOLERANCE,
                            "uniform_match_min": MATCH_MINIMUM, "m1_hole_effect": M1_HOLE_EFFECT})
    marker = args.output / "identity.json"
    if marker.is_file() and read_json(marker) != provenance:
        raise ValueError("Output exists with different experiment identity; choose another --output")
    run.write_json(marker, provenance)
    with torch.inference_mode():
        if args.phase in ("v6", "all"):
            v6(args)
        if args.phase in ("v5", "all"):
            v5(args)
    print(f"ALL DONE: Our2 {args.phase} ({args.output})", flush=True)


if __name__ == "__main__":
    main()
