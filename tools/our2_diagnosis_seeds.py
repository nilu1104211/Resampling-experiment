"""Fixed D1 direction controls and paired geometry diagnostics."""

import hashlib

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from our2_v56 import as_numpy
from our2_v6_seeds import make_pool, quota, reallocate, replace, shares


RANDOM_GROUPS = ("D1-R21", "D1-R22", "D1-R23")
DIRECTIONS = ("D0", "D1-L", "D1-U", *RANDOM_GROUPS)


def feasible_partition(pool, count, donor_order):
    """Choose disjoint donors and recipients, preserving enough capacity on both sides."""
    outgoing = pool.n0 - pool.lower
    incoming = pool.upper - pool.n0
    room = int(incoming.sum()) - count
    states = {0: (0, ())}
    for region in donor_order:
        if not outgoing[region]:
            continue
        for sent, (cost, selected) in list(states.items()):
            new_sent = min(count, sent + int(outgoing[region]))
            new_cost = cost + int(incoming[region])
            if new_cost <= room and (new_sent not in states or new_cost < states[new_sent][0]):
                states[new_sent] = new_cost, (*selected, int(region))
        if count in states:
            break
    if count not in states:
        raise ValueError("No disjoint donor/recipient partition can realize the requested moves")
    mask = np.zeros(len(outgoing), dtype=bool)
    mask[list(states[count][1])] = True
    return outgoing * mask, incoming * ~mask


def control_budget(pool, group, count, key):
    if count == 0:
        return pool.n0.copy(), "zero"
    incoming = pool.upper - pool.n0
    outgoing = pool.n0 - pool.lower
    if group == "D1-U":
        target = pool.uniform
        budget, _, realized = quota(pool.n0, pool.lower, pool.upper, target, .25, .05, count)
        if realized == count:
            return budget, "uniform_target"
        # Widen only the capacity along the original direction before relaxing its signs.
        delta = target - pool.n0
        add = incoming * (delta > 1e-5)
        remove = outgoing * (delta < -1e-5)
        if min(add.sum(), remove.sum()) >= count:
            return pool.n0 + shares(add, count) - shares(remove, count), "uniform_direction_widened"
        remove, add = feasible_partition(pool, count, np.argsort(delta, kind="stable"))
        return pool.n0 + shares(add, count) - shares(remove, count), "uniform_feasible_fallback"

    digest = hashlib.sha256(f"{group}:{key}".encode()).digest()
    random = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    remove, add = feasible_partition(pool, count, random.permutation(len(pool.n0)))
    # Sequential random draws from feasible slots yield an exact, reproducible integer allocation.
    gained = np.bincount(random.choice(np.repeat(np.arange(len(add)), add), count, replace=False),
                         minlength=len(add))
    lost = np.bincount(random.choice(np.repeat(np.arange(len(remove)), remove), count, replace=False),
                       minlength=len(remove))
    return pool.n0 + gained - lost, "random_feasible"


def direction_seeds(original, observed, head, key, groups=DIRECTIONS):
    pool = make_pool(original, observed, head, 32)
    learned, info = reallocate(original, pool, "D1-L")
    count = info["changed"]
    values = {"D0": (original, pool.n0, 0, "midpoint"),
              "D1-L": (learned, info["budget"], count, "learned")}
    for group in groups:
        if group in values:
            continue
        budget, strategy = control_budget(pool, group, count, key)
        seeds, changed = replace(pool, budget)
        if changed != count:
            raise ValueError(f"{group} realized {changed} replacements, expected {count}")
        values[group] = seeds, budget, changed, strategy
    return pool, {group: values[group] for group in groups}


def binary_scores(scores, positive):
    predicted = scores >= .5
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    positives, negatives = tp + fn, fp + tn
    auc = ap = None
    if positives and negatives:
        auc = (rankdata(scores)[positive].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        order = np.argsort(-scores, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
        true = np.cumsum(positive[order])[ends]
        ap = ((true / (ends + 1)) * np.diff(np.r_[0, true])).sum() / positives
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                auroc=auc, auprc=ap)


def head_rows(sample, geo, scores, key):
    labels = sample["supervision"]
    r, h = as_numpy(scores["r"][0]), as_numpy(scores["h"][0])
    rows = []
    for name, label in (("outlier", "outlier_mask_input"), ("noise", "noise_mask_input")):
        if label not in labels:
            rows.append({**key, "head": "r", "label": name, "reason": "LABEL_UNAVAILABLE"})
            continue
        positive = np.asarray(labels[label], dtype=bool)
        rows.append({**key, "head": "r", "label": name, "n": len(r),
                     **binary_scores(1 - r, positive),
                     "p05": np.quantile(r, .05), "p50": np.quantile(r, .5), "p95": np.quantile(r, .95),
                     "positive_mean": r[positive].mean() if positive.any() else None,
                     "negative_mean": r[~positive].mean() if (~positive).any() else None,
                     "reason": "" if positive.any() and (~positive).any() else "ONE_CLASS"})
    points = as_numpy(geo["candidates"][0])
    distances, ids = cKDTree(sample["gt_xyz"]).query(points)
    scale = as_numpy(geo["region_scale"][0, :, 0])
    hole = labels.get("hole_mask_gt")
    if hole is None:
        rows.append({**key, "head": "h", "reason": "LABEL_UNAVAILABLE"})
    else:
        target = np.asarray(hole)[ids] * np.exp(-(distances / scale) ** 2)
        positive = target >= .5
        rows.append({**key, "head": "h", "label": "training_soft_target_at_0.5", "n": len(h),
                     **binary_scores(h, positive), "p05": np.quantile(h, .05),
                     "p50": np.quantile(h, .5), "p95": np.quantile(h, .95),
                     "reason": "" if positive.any() and (~positive).any() else "ONE_CLASS"})
    return rows


def seed_geometry(sample, seeds):
    target = sample["gt_xyz"]
    xyz = as_numpy(seeds[0])
    gt = cKDTree(target)
    cloud = cKDTree(xyz)
    to_gt = gt.query(xyz)[0]
    to_seed = cloud.query(target)[0]
    spacing = cloud.query(xyz, k=2)[0][:, 1]
    result = {"seed_to_gt": to_gt.mean(), "gt_to_seed": to_seed.mean(),
              "seed_spacing_p01": np.quantile(spacing, .01), "seed_spacing_p50": np.median(spacing),
              "duplicate_fraction": (spacing == 0).mean(), "near_duplicate_fraction": (spacing < 1e-6).mean()}
    for name in ("hole", "sparse"):
        mask = sample["supervision"].get(f"{name}_mask_gt")
        if mask is not None and mask.any():
            result[f"{name}_gt_to_seed"] = to_seed[mask.astype(bool)].mean()
        else:
            result[f"{name}_reason"] = "NO_VALID_REGION"
    return result


def shift_geometry(sample, original, corrected):
    xyz, moved = as_numpy(original[0]), as_numpy(corrected[0])
    spacing = cKDTree(xyz).query(xyz, k=2)[0][:, 1]
    delta = moved - xyz
    distance = np.linalg.norm(delta, axis=1)
    gt = sample["gt_xyz"]
    nearest = cKDTree(gt).query(xyz)[1]
    ideal = gt[nearest] - xyz
    magnitude = distance * np.linalg.norm(ideal, axis=1)
    valid = magnitude > 1e-12
    cosine = (delta[valid] * ideal[valid]).sum(1) / magnitude[valid]
    result = {"shift_mean": distance.mean(), "shift_p50": np.median(distance),
              "shift_p95": np.quantile(distance, .95), "shift_max": distance.max(),
              "shift_over_spacing": (distance / np.maximum(spacing, 1e-8)).mean(),
              "shift_nonzero_fraction": (distance > 0).mean(),
              "toward_gt_cosine": cosine.mean() if valid.any() else None}
    for name in ("hole", "sparse"):
        label = sample["supervision"].get(f"{name}_mask_gt")
        if label is not None and label.any():
            mask = label[nearest].astype(bool)
            if mask.any():
                result[f"{name}_shift_mean"] = distance[mask].mean()
    return result


def flow_steps(model, seeds):
    state = {"initial": seeds}
    points = [seeds]
    for step in range(5):
        query = points[-1]
        points.append(query + model.velocity(query, query.new_full((len(query),), step / 5), state) / 5)
    for value in points:
        if value.shape != seeds.shape or not torch.isfinite(value).all():
            raise ValueError("Invalid Flow trajectory")
    return points
