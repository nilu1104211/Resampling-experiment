"""Run paired inference and save stage measurements without GT in the forward path."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import torch

from adaptive_resampling.v3_dataset import inverse_prediction
from adaptive_resampling.v3_metrics import evaluate_sample
from models.adaptive_ops import gather, knn
from models.geometry import geometry
from our2_diagnosis_seeds import (
    DIRECTIONS, direction_seeds, flow_steps, head_rows, seed_geometry, shift_geometry,
)
from our2_v56 import M1_GROUPS, M1_SWITCHES, as_numpy, midpoint, run
from our2_v6_seeds import replace


GROUPS = {"m1": M1_GROUPS, "m2": DIRECTIONS, "joint": ("J0", "J1", "J2", "J3", "J4")}


def m1_paths(model, original, observed):
    geo = geometry(observed, model.config["region_divisor"])
    scores = model.m1(geo)
    refinements = {group: model.m1.refine(original, geo, scores, *M1_SWITCHES[group])
             for group in M1_GROUPS if group in M1_SWITCHES}
    paths = {group: result[0] for group, result in refinements.items()}
    paths["M1-no-move"] = original
    if not torch.equal(paths["M1-off"], original):
        raise ValueError("M1-off and M1-no-move diverge")
    ids = knn(geo["points"], original, 8)
    scale = gather(geo["scale"][..., None], ids).mean(2)
    closest = knn(geo["candidates"], original, 1).squeeze(-1)
    distance = (gather(geo["candidates"], closest) - original).norm(dim=-1, keepdim=True)
    gate = (-distance.square() / (4 * scale.square())).exp() * gather(scores["h"][..., None], closest)
    gamma = float(model.m1.gamma.tanh())
    actual = refinements["M1-full"][1]
    cap = abs(gamma) * model.m1.max_move * scale
    clipped = (actual.norm(dim=-1, keepdim=True) >= .999 * cap).float().mean() if gamma != 0 else None
    diagnostics = {"gamma_tanh": gamma, "hole_gate_mean": float(gate.mean()), "hole_gate_p95": float(gate.quantile(.95)),
                   "untrusted_gate_mean": float(1 - gather(scores["r"][..., None], ids).mean()),
                   "movement_cap_fraction": float(clipped) if clipped is not None else None}
    return paths, (geo, scores, diagnostics)


def joint_paths(model, original, observed):
    cfg = model.config
    if not cfg["joint_condition"] or cfg["budget_policy"] != "learned":
        raise ValueError("Joint condition ablation requires a learned, conditioned checkpoint")
    geo = geometry(observed, cfg["region_divisor"])
    m1 = model.m1(geo)
    logits_on = model.m2(geo, m1, True, "learned")
    logits_off = model.m2(geo, None, False, "learned")
    with_condition, budget_on = model.m2.allocate(original, geo, logits_on, cfg["allocation_strength"])
    no_condition, budget_off = model.m2.allocate(original, geo, logits_off, cfg["allocation_strength"])

    def refine(seeds):
        return model.m1.refine(seeds, geo, m1, cfg["use_reliability"], cfg["use_hole"])[0]

    paths = {"J0": original, "J1": refine(original), "J2": no_condition,
             "J3": refine(with_condition), "J4": refine(no_condition)}
    state = {
        "geo": geo, "m1": m1,
        "pre_refine": dict(J0=original, J1=original, J2=no_condition, J3=with_condition, J4=no_condition),
        "logits": {"J2": logits_off, "J3": logits_on, "J4": logits_off},
        "budget": {"J2": budget_off["hard"], "J3": budget_on["hard"], "J4": budget_off["hard"]},
    }
    return paths, state


def predict_paths(mode, model, observed, head=None, key="", groups=None):
    original = midpoint(observed)
    if mode == "m1":
        seeds, context = m1_paths(model, original, observed)
    elif mode == "m2":
        context = direction_seeds(original, observed, head, key, groups or DIRECTIONS)
        seeds = {g: value[0] for g, value in context[1].items()}
    else:
        seeds, context = joint_paths(model, original, observed)
    if any(value.shape != original.shape for value in seeds.values()):
        raise ValueError("Seed budget changed")
    traces = {}
    for group, points in seeds.items():
        if groups is not None and group not in groups:
            continue
        if group == "M1-no-move":
            traces[group] = traces["M1-off"]
        else:
            traces[group] = flow_steps(model, points)
    return original, context, traces


def budget_rows(pool, values, key, sample):
    rows, matches = [], []
    learned_count = values["D1-L"][2]
    gt = sample["gt_xyz"]
    gt_regions = pool.anchor_tree.query(gt)[1]
    gt_tree = cKDTree(gt)
    original_distance = cKDTree(pool.coordinates).query(gt)[0]
    for group, (seeds, budget, changed, strategy) in values.items():
        xyz = as_numpy(seeds[0])
        changed_mask = (xyz != pool.coordinates).any(axis=1)
        distance = cKDTree(xyz).query(gt)[0]
        gained = (distance < original_distance - 1e-8).sum()
        lost = (distance > original_distance + 1e-8).sum()
        matches.append({**key, "group": group, "requested_m": learned_count if group != "D0" else 0,
                        "actual_m": changed, "matched": group != "D0" and changed == learned_count,
                        "strategy": strategy, "gt_points_nearer": int(gained), "gt_points_farther": int(lost),
                        "added_seed_to_gt": gt_tree.query(xyz[changed_mask])[0].mean() if changed else None,
                        "removed_seed_to_gt": gt_tree.query(pool.coordinates[changed_mask])[0].mean() if changed else None})
        for j, count in enumerate(pool.n0):
            mask = gt_regions == j
            rows.append({**key, "group": group, "region": j, "n0": count, "budget": budget[j],
                         "delta": budget[j] - count, "lower": pool.lower[j], "upper": pool.upper[j],
                         "learned_target": pool.learned[j], "uniform_target": pool.uniform[j],
                         "new_candidates": len(pool.novel[j]),
                         "region_gt_distance_delta": (distance[mask] - original_distance[mask]).mean() if mask.any() else None})
        # Slot-preserving replacement, with neighborhoods recomputed by PUFM from current XYZ.
        if int(changed_mask.sum()) != changed:
            raise ValueError("Actual coordinate changes do not match the recorded budget")
    return rows, matches


def proxy_rows(pool, sample, key):
    """Measure local change with one added seed, paid for by one other region."""
    gt = sample["gt_xyz"]
    labels = pool.anchor_tree.query(gt)[1]
    original = cKDTree(pool.coordinates).query(gt)[0]
    rows = []
    for region in np.flatnonzero(pool.upper > pool.n0):
        donors = np.flatnonzero((pool.n0 > pool.lower) & (np.arange(len(pool.n0)) != region))
        mask = labels == region
        if not len(donors) or not mask.any():
            continue
        donor = donors[np.argmax((pool.n0 - pool.lower)[donors])]
        budget = pool.n0.copy()
        budget[region] += 1
        budget[donor] -= 1
        seeds, _ = replace(pool, budget)
        delta = original - cKDTree(as_numpy(seeds[0])).query(gt)[0]
        rows.append({**key, "region": region, "donor": donor, "need": pool.learned[region],
                     "local_gain": delta[mask].mean(), "global_gain": delta.mean()})
    if len(rows) >= 3:
        need = [r["need"] for r in rows]
        gains = [r["local_gain"] for r in rows]
        if np.ptp(need) and np.ptp(gains):
            rows[0]["within_sample_spearman"] = float(spearmanr(need, gains).statistic)
    return rows


def measure(sample, mode, original, context, traces, key, evidence, full_trace=True):
    reference = {"m1": "M1-off", "m2": "D0", "joint": "J0"}[mode]
    rows = []
    if mode == "m1":
        evidence["m1_heads"].extend(head_rows(sample, *context[:2], key))
    elif mode == "joint":
        evidence["m1_heads"].extend(head_rows(sample, context["geo"], context["m1"], key))
    elif mode == "m2":
        pool, values = context
        regions, moves = budget_rows(pool, values, key, sample)
        evidence["m2_budgets"].extend(regions)
        evidence["matched_moves"].extend(moves)
    for group, steps in traces.items():
        row = {**key, "group": group, "prediction_sha256": run.tensor_hash(steps[-1]),
               **evaluate_sample(steps[-1][0], sample)}
        for region in ("hole", "sparse"):
            if f"{region}_coverage_mean_l2" not in row:
                row[f"{region}_reason"] = "NO_VALID_REGION"
        row["pred_to_gt_mean_l2"] = 2 * row["cd_l1"] - row["gt_coverage_mean_l2"]
        rows.append(row)
        if mode == "joint":
            logits = context["logits"].get(group)
            budget = context["budget"].get(group)
            before = context["pre_refine"][group]
            evidence["joint_interaction"].append({
                **row, **shift_geometry(sample, before, steps[0]),
                "pre_refine_seed_to_gt": seed_geometry(sample, before)["seed_to_gt"],
                "hard_budget": "|".join(map(str, as_numpy(budget[0]))) if budget is not None else "",
                "logits": "|".join(map(str, as_numpy(logits[0]))) if logits is not None else "",
                "budget_l1_vs_j3": int((budget - context["budget"]["J3"]).abs().sum()) if budget is not None else None,
                "logit_l2_vs_j3": float((logits - context["logits"]["J3"]).norm()) if logits is not None else None,
            })
        initial_gap = float((steps[0] - traces[reference][0]).norm(dim=-1).mean())
        for step in (range(6) if full_trace else (0, 5)):
            points = steps[step]
            gap = float((points - traces[reference][step]).norm(dim=-1).mean())
            evidence[f"{mode}_seed_flow"].append({
                **key, "group": group, "step": step, "time": step / 5,
                "diagnostic_units": "network_normalized_xyz",
                "mean_index_displacement_vs_control": gap,
                "retention_vs_initial": gap / initial_gap if initial_gap > 0 else None,
                **seed_geometry(sample, points),
                **(shift_geometry(sample, original, points) if step == 0 and mode == "m1" else {}),
                **(context[2] if step == 0 and mode == "m1" and group == "M1-full" else {}),
            })
    return rows


def save_example(path, sample, original, context, traces, mode):
    def xyz(tensor):
        return inverse_prediction(as_numpy(tensor[0]), sample)

    payload = {"condition": sample["condition"], "object_id": sample["object_id"],
               "sample_id": sample["sample_id"], "original_seed": xyz(original),
               "observed": inverse_prediction(sample["input_xyz"], sample),
               "target": inverse_prediction(sample["gt_xyz"], sample),
               "hole_mask": sample["supervision"]["hole_mask_gt"].astype(bool),
               "outlier_mask": sample["supervision"]["outlier_mask_input"].astype(bool)}
    for group, steps in traces.items():
        payload[f"{group}_seed"] = xyz(steps[0])
        payload[f"{group}_prediction"] = xyz(steps[-1])
        payload[f"{group}_trajectory"] = np.stack([xyz(points) for points in steps])
        if mode == "m2":
            payload[f"{group}_seed_changed"] = (as_numpy(steps[0][0]) != as_numpy(original[0])).any(axis=1)
    if mode == "m2":
        pool, values = context
        payload["anchors"] = inverse_prediction(pool.anchors, sample)
        for group, (_, budget, _, _) in values.items():
            payload[f"{group}_budget_delta"] = budget - pool.n0
    if mode in ("m1", "joint"):
        geo, heads = context[:2] if mode == "m1" else (context["geo"], context["m1"])
        payload.update(r=as_numpy(heads["r"][0]), h=as_numpy(heads["h"][0]),
                       candidates=xyz(geo["candidates"]), observed_knn=as_numpy(geo["ids"][0]),
                       region_id=as_numpy(geo["labels"][0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def render_example(path: Path, destination: Path, mode: str):
    from plot_our2_p0 import plt

    primary, control = {"m1": ("M1-full", "M1-off"), "m2": ("D1-L", "D1-U"),
                        "joint": ("J3", "J4")}[mode]
    panels = [("observed", "Input"), ("original_seed", "Midpoint"),
              (f"{primary}_seed", f"{primary} seeds"), (f"{control}_seed", f"{control} seeds"),
              ("target", "Ground truth"), (f"{primary}_prediction", f"{primary} output"),
              (f"{control}_prediction", f"{control} output")]
    with np.load(path, allow_pickle=False) as data:
        combined = np.concatenate([data[key] for key, _ in panels])
        lo, hi = combined.min(axis=0), combined.max(axis=0)
        center, radius = (lo + hi) / 2, max((hi - lo).max() * .53, 1e-5)
        fig = plt.figure(figsize=(12, 6), layout="constrained")
        for i, (key, title) in enumerate(panels, 1):
            axis = fig.add_subplot(2, 4, i, projection="3d")
            xyz = data[key]
            mask_key = {"observed": "outlier_mask", "target": "hole_mask"}.get(key, f"{key}_changed")
            colors = np.where(data[mask_key], "#c35232", "#285b78") if mask_key in data else "#285b78"
            axis.scatter(*xyz.T, c=colors, s=1.4, linewidths=0, depthshade=False)
            budget_key = key.replace("_seed", "_budget_delta")
            if mode == "m2" and key.endswith("_seed") and budget_key in data:
                change = data[budget_key]
                active = change != 0
                axis.scatter(*data["anchors"][active].T, c=np.where(change[active] > 0, "#338630", "#82368c"),
                             s=24, depthshade=False)
            axis.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                     zlim=(center[2]-radius, center[2]+radius), title=title)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=18, azim=45)
            axis.set_proj_type("ortho")
            axis.set_axis_off()
        fig.suptitle(f"{data['condition']} | {data['object_id']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=130)
        plt.close(fig)
