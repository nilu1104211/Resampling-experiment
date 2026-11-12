"""Observation-only, minimal-change Midpoint allocation for the V6 experiment."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.adaptive_ops import fps_ids
from models.gdba import bounded_soft_budget
from models.geometry import geometry


LEARNED = {"D1-L": (.25, .05), "D2-L": (.50, .10), "D3-L": (.75, .10)}


@dataclass
class Pool:
    seeds: torch.Tensor
    coordinates: np.ndarray
    indices: list
    novel: list
    anchors: np.ndarray
    anchor_tree: cKDTree
    n0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    learned: np.ndarray
    uniform: np.ndarray
    donor_order: dict = field(default_factory=dict)
    recipient_order: dict = field(default_factory=dict)


def make_pool(seeds, observed, head, divisor):
    if len(seeds) != 1 or len(observed) != 1:
        raise ValueError("V6 diagnostics use one input at a time")
    geo = geometry(observed, divisor)
    anchors = geo["anchors"][0].cpu().numpy()
    anchor_tree = cKDTree(anchors)
    coordinates = seeds[0].cpu().numpy()
    labels = anchor_tree.query(coordinates)[1]
    count = len(anchors)
    indices = [np.flatnonzero(labels == j) for j in range(count)]
    n0 = np.array([len(ids) for ids in indices], dtype=np.int64)
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("Original Midpoint contains duplicate coordinates; inspect input before V6")

    # Share the GDBA geometric candidate rule, but exclude every original seed
    # globally. Reassign boundary candidates using the same nearest-anchor rule.
    source, _, _ = head.candidates(seeds, geo)
    candidates = torch.cat(source[0]).cpu().numpy()
    unique, inverse = np.unique(np.concatenate((coordinates, candidates)), axis=0, return_inverse=True)
    fresh = unique[np.setdiff1d(inverse[len(coordinates):], inverse[:len(coordinates)])]
    if len(fresh):
        fresh_labels = anchor_tree.query(fresh)[1]
        novel = [fresh[fresh_labels == j] for j in range(count)]
    else:
        novel = [fresh for _ in range(count)]
    nominal_lower = int(head.minimum * len(coordinates) / count)
    nominal_upper = int(head.maximum * len(coordinates) / count)
    lower = np.minimum(n0, nominal_lower)
    upper = np.maximum(n0, np.minimum(nominal_upper, n0 + np.array([len(p) for p in novel])))
    limits = tuple(torch.as_tensor(x, device=seeds.device)[None] for x in (lower, upper))
    logits = head(geo, None, False, "learned")
    learned, uniform = (bounded_soft_budget(scores, len(coordinates), *limits, head.temperature)[0]
                        .cpu().numpy().astype(np.float64) for scores in (logits, torch.zeros_like(logits)))
    return Pool(seeds, coordinates, indices, novel, anchors, anchor_tree, n0, lower, upper, learned, uniform)


def shares(direction, moves):
    """Integer proportional apportionment with deterministic ties by region ID."""
    if not moves:
        return np.zeros(len(direction), dtype=np.int64)
    ideal = direction * (moves / direction.sum())
    caps = np.ceil(direction).astype(np.int64)
    actual = np.minimum(np.floor(ideal).astype(np.int64), caps)
    remaining = moves - int(actual.sum())
    order = np.argsort(-(ideal - actual), kind="stable")
    for index in order:
        if not remaining:
            break
        if actual[index] < caps[index]:
            actual[index] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Could not apportion the requested integer moves")
    return actual


def quota(n0, lower, upper, target, lam, rho_max, requested=None):
    """Mix toward the bounded soft target, then round balanced transfers."""
    delta = np.clip(target - n0, lower - n0, upper - n0)
    delta[np.abs(delta) < 1e-5] = 0
    incoming, outgoing = delta.clip(min=0), (-delta).clip(min=0)
    feasible = int(min(incoming.sum(), outgoing.sum()) + .50001)
    cap = min(int(rho_max * n0.sum()), feasible)
    wanted = int(lam * min(incoming.sum(), outgoing.sum()) + .50001) if requested is None else requested
    moves = min(wanted, cap)
    budget = n0 + shares(incoming, moves) - shares(outgoing, moves)
    if (budget.sum() != n0.sum() or (budget < lower).any() or (budget > upper).any()
            or np.abs(budget - n0).sum() != 2 * moves):
        raise ValueError("V6 integer budget violates bounds or transfer total")
    return budget, wanted, moves


def farthest_candidates(coordinates, candidates, anchor, count, prefix):
    """Greedy cover, seeded by every original point retained in the recipient."""
    if len(prefix) >= count:
        return prefix
    chosen = list(prefix)
    if not len(coordinates) and not chosen:
        chosen.append(int(np.sum((candidates - anchor) ** 2, axis=1).argmin()))
    if chosen:
        distance = cKDTree(candidates[chosen]).query(candidates)[0] ** 2
        if len(coordinates):
            distance = np.minimum(distance, cKDTree(coordinates).query(candidates)[0] ** 2)
    else:
        distance = cKDTree(coordinates).query(candidates)[0] ** 2
    distance[chosen] = -np.inf
    while len(chosen) < count:
        index = int(distance.argmax())
        chosen.append(index)
        distance = np.minimum(distance, np.sum((candidates - candidates[index]) ** 2, axis=1))
        distance[index] = -np.inf
    return chosen


def replace(pool, budget):
    change = budget - pool.n0
    moves = int(change.clip(min=0).sum())
    if not moves:
        return pool.seeds, 0
    removed, added = [], []
    for j, delta in enumerate(change):
        if delta < 0:
            indices = pool.indices[j]
            if j not in pool.donor_order:
                ranked = fps_ids(pool.seeds[:, indices], len(indices))[0].cpu().numpy()
                pool.donor_order[j] = indices[ranked]
            removed.extend(pool.donor_order[j][delta:].tolist())
        elif delta > 0:
            pool.recipient_order[j] = farthest_candidates(
                pool.coordinates[pool.indices[j]], pool.novel[j], pool.anchors[j], delta,
                pool.recipient_order.get(j, []),
            )
            added.append(pool.novel[j][pool.recipient_order[j][:delta]])
    result = pool.coordinates.copy()
    result[np.sort(removed)] = np.concatenate(added)
    changed = int(np.any(result != pool.coordinates, axis=1).sum())
    if changed != moves or len(np.unique(result, axis=0)) != len(result):
        raise ValueError("V6 changed unintended seeds or introduced duplicate coordinates")
    actual = np.bincount(pool.anchor_tree.query(result)[1], minlength=len(budget))
    if not np.array_equal(actual, budget):
        raise ValueError("V6 seed coordinates do not realize the assigned budget")
    return torch.as_tensor(result, device=pool.seeds.device)[None], changed


def reallocate(seeds, pool=None, group="D0", matched_moves=None):
    # Identity must not build regions, generate candidates or rerun FPS.
    if group == "D0":
        return seeds, None
    key = group[:-1] + "L"
    lam, rho = LEARNED[key]
    target = pool.learned if group.endswith("L") else pool.uniform
    if group.endswith("U") and matched_moves is None:
        raise ValueError("Uniform comparison requires the learned group's actual move count")
    budget, requested, moved = quota(pool.n0, pool.lower, pool.upper, target, lam, rho, matched_moves)
    result, changed = replace(pool, budget)
    return result, {
        "target": target, "budget": budget, "requested": requested, "moved": moved, "changed": changed,
        "rho_max": rho, "lambda": lam if group.endswith("L") else None,
    }
