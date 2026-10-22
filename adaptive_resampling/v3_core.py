"""V3 surface-backed bases and reproducible, correspondence-preserving corruptions.

All input conditions sample from an independent surface pool. Neither synthetic
labels nor reference normals are inference features.
"""
from __future__ import annotations

import json
import numpy as np
from scipy.spatial import cKDTree

from scipy.sparse.csgraph import connected_components
from .core import array_hash, normalize_pair, point_graph, stable_seed

VERSION = "3.0.0"
SEED = 20261021


def seed(*parts):
    return stable_seed(*parts, seed=SEED)


CONDITIONS = {
    "A0_normal": dict(kind=0),
    "B1_sparse_light": dict(kind=1, density=0.75),
    "B2_sparse_mid": dict(kind=1, density=0.50),
    "B3_sparse_heavy": dict(kind=1, density=0.25),
    "C1_hole_05": dict(kind=2, hole=0.05),
    "C2_hole_10": dict(kind=2, hole=0.10),
    "C3_hole_20": dict(kind=2, hole=0.20),
    "D1_noise_002": dict(kind=3, sigma=0.002, noise_fraction=0.25),
    "D2_noise_005": dict(kind=3, sigma=0.005, noise_fraction=0.25),
    "D3_noise_010": dict(kind=3, sigma=0.010, noise_fraction=0.25),
    "D4_global_gaussian_005": dict(kind=3, sigma=0.005, noise_fraction=1.0),
    "D5_local10_gaussian_005": dict(kind=3, sigma=0.005, noise_fraction=0.10),
    "D6_local50_gaussian_005": dict(kind=3, sigma=0.005, noise_fraction=0.50),
    "D7_outliers_02": dict(kind=3, outlier_fraction=0.02),
    # This distribution is deliberately excluded from online training.
    "D8_laplace_005_unseen": dict(kind=3, sigma=0.005, noise_fraction=0.25, distribution="laplace"),
    "E1_mixed": dict(kind=4, density=0.50, hole=0.10, sigma=0.005, noise_fraction=0.25),
    "E2_mixed_outliers": dict(kind=4, density=0.50, hole=0.10, sigma=0.005, noise_fraction=0.25, outlier_fraction=0.02),
}
VALIDATION_CONDITIONS = ["A0_normal", "B2_sparse_mid", "C2_hole_10", "D2_noise_005", "D7_outliers_02", "E1_mixed", "E2_mixed_outliers"]


def geometry_targets(points, normals, k=16):
    """Reference-only local surface variation and unoriented normal variation."""
    ids = cKDTree(points).query(points, k=min(k, len(points)), workers=1)[1]
    neighbors = points[ids].astype("f8")
    neighbors -= neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", neighbors, neighbors) / ids.shape[1]
    values = np.maximum(np.linalg.eigvalsh(covariance), 0)
    curvature = values[:, 0] / np.maximum(values.sum(axis=1), 1e-15)
    cosines = np.abs(np.einsum("ni,nki->nk", normals, normals[ids]))
    variation = (1 - np.clip(cosines, 0, 1)).mean(axis=1)
    return curvature.astype("f4"), variation.astype("f4")


def prepare_base(clean_pool, target, pool_normals, target_normals, *, base_id,
                 object_id, category, mesh_path, mesh_sha256, geometry_id,
                 gt_face_index, gt_barycentric, gt_sample_index, pool_sample_index,
                 input_count=256):
    """Freeze clean geometry once; subsequent conditions never alter the target."""
    rng = np.random.default_rng(seed(base_id, "input_priority"))
    priority = rng.permutation(len(clean_pool))
    clean_ids = priority[:input_count]
    clean, gt, center, scale = normalize_pair(clean_pool[clean_ids], target)
    pool = ((np.asarray(clean_pool, dtype="f8") - center) / scale).astype("f4")
    # Independent draws can still coincide after float32 rounding on tiny meshes.
    distances = cKDTree(gt).query(pool, workers=1)[0]
    if np.any(distances < 1e-7):
        raise ValueError("Input candidate coincides with target after float32 normalization")
    tree, graph = point_graph(pool)
    components = connected_components(graph, directed=False)[0]
    # A spatial ellipsoid is a well-defined contiguous occlusion volume even
    # when a CAD object has many disconnected surface components. Requiring a
    # 20%-area connected point-graph component would reject such objects.
    def spatial_order(name):
        local_rng = np.random.default_rng(seed(base_id, name))
        center_id = int(local_rng.integers(len(pool)))
        rotation, _ = np.linalg.qr(local_rng.standard_normal((3, 3)))
        axes = local_rng.uniform(0.65, 1.5, 3)
        distance = np.linalg.norm(((pool - pool[center_id]) @ rotation) / axes, axis=1)
        return np.argsort(distance, kind="stable"), distance, center_id, axes, rotation
    hole_order, hole_distance, hole_center, hole_axes, hole_rotation = spatial_order("hole")
    sparse_order, _, sparse_center, _, _ = spatial_order("sparse")
    _, noise_distance, noise_center, _, _ = spatial_order("noise")
    rank = np.full(len(pool), len(pool), dtype="i4")
    rank[hole_order] = np.arange(len(hole_order))
    sparse = np.zeros(len(pool), dtype="u1")
    sparse[sparse_order[:round(len(pool) * 0.30)]] = 1
    curvature, normal_variation = geometry_targets(gt, target_normals)
    spacing = float(np.median(tree.query(pool, k=2, workers=1)[0][:, 1]))
    if spacing <= 1e-8:
        raise ValueError("Degenerate surface candidate spacing")
    return {
        "pool_xyz": pool, "pool_normals": np.asarray(pool_normals, dtype="f4"),
        "gt_xyz": gt, "gt_normals": np.asarray(target_normals, dtype="f4"),
        "gt_curvature": curvature, "gt_normal_variation": normal_variation,
        "clean_indices": clean_ids.astype("i4"), "priority": priority.astype("i4"),
        "hole_rank": rank, "hole_distance": hole_distance.astype("f4"),
        "sparse_mask_ref": sparse, "noise_distance": noise_distance.astype("f4"),
        "gt_ref_index": tree.query(gt, workers=1)[1].astype("i4"),
        "gt_face_index": np.asarray(gt_face_index, dtype="i4"),
        "gt_barycentric": np.asarray(gt_barycentric, dtype="f8"),
        "gt_sample_index": np.asarray(gt_sample_index, dtype="i4"),
        "pool_sample_index": np.asarray(pool_sample_index, dtype="i4"),
        "center": center.astype("f8"), "scale": np.float64(scale),
        "spacing": np.float32(spacing), "input_count": np.int32(input_count),
        "hole_center": np.int32(hole_center), "sparse_center": np.int32(sparse_center),
        "noise_center": np.int32(noise_center), "graph_components": np.int32(components),
        "region_method": "spatial_ellipsoid_surface_fraction_v3",
        "hole_axes": hole_axes.astype("f4"), "hole_rotation": hole_rotation.astype("f4"),
        "base_id": base_id, "object_id": object_id, "category": category,
        "mesh_path": str(mesh_path), "mesh_sha256": mesh_sha256, "geometry_id": geometry_id,
        "gt_sha256": array_hash(gt),
    }


def training_condition(base_id, index, epoch, mode="mixed"):
    if mode == "clean":
        return "A0_normal", dict(CONDITIONS["A0_normal"])
    if mode != "mixed":
        raise ValueError("Training mode must be 'clean' or 'mixed'")
    rng = np.random.default_rng(seed(base_id, "online", epoch))
    category = (int(index) + int(epoch)) % 5
    if category == 0:
        return "online_normal", dict(kind=0)
    if category == 1:
        return "online_sparse", dict(kind=1, density=float(rng.uniform(0.25, 0.85)))
    if category == 2:
        return "online_hole", dict(kind=2, hole=float(rng.uniform(0.04, 0.20)))
    sigma = float(rng.uniform(0.002, 0.010))
    fraction = float(rng.uniform(0.10, 0.50))
    outliers = 0.02 if rng.random() < 0.30 else 0.0
    if category == 3:
        return "online_noise", dict(kind=3, sigma=sigma, noise_fraction=1.0 if rng.random() < 0.20 else fraction, outlier_fraction=outliers)
    return "online_mixed", dict(kind=4, hole=float(rng.uniform(0.04, 0.20)),
                                density=float(rng.uniform(0.25, 0.85)), sigma=sigma,
                                noise_fraction=fraction, outlier_fraction=outliers)


def corrupt_base(base, condition, *, spec=None, epoch=None):
    config = dict(CONDITIONS[condition] if spec is None else spec)
    kind = int(config["kind"])
    pool = base["pool_xyz"]
    count = int(base["input_count"])
    size = len(pool)
    priority = base["priority"]
    fraction = float(config.get("hole", 0))
    hole = base["hole_rank"] < round(size * fraction)
    sparse = base["sparse_mask_ref"].astype(bool) if kind in (1, 4) else np.zeros(size, dtype=bool)
    eligible = priority[~hole[priority]]
    density = float(config.get("density", 1.0))
    actual_density = 1.0
    if kind in (0, 3):
        selected = base["clean_indices"].copy()
    elif kind in (1, 4):
        inner, outer = eligible[sparse[eligible]], eligible[~sparse[eligible]]
        area_fraction = len(inner) / len(eligible)
        quota = round(count * density * area_fraction / (1 - area_fraction + density * area_fraction))
        quota = max(1, min(count - 1, quota))
        if len(inner) < quota or len(outer) < count - quota:
            raise ValueError("Region capacity is insufficient")
        selected = np.concatenate([inner[:quota], outer[:count - quota]])
        # Output order is deterministic but contains no region-block ordering cue.
        np.random.default_rng(seed(base["base_id"], "shuffle_selected")).shuffle(selected)
        actual_density = (quota / len(inner)) / ((count - quota) / len(outer))
    else:
        selected = eligible[:count].copy()
    if len(selected) != count or len(np.unique(selected)) != count:
        raise ValueError("Corruption must select exactly N distinct clean points")
    clean = pool[selected].copy()
    noisy = clean.copy()
    sigma = float(config.get("sigma", 0))
    noise_fraction = float(config.get("noise_fraction", 0))
    outlier_fraction = float(config.get("outlier_fraction", 0))
    noise_mask = np.zeros(count, dtype=bool)
    outlier_mask = np.zeros(count, dtype=bool)
    # Reuse the same standardized noise across fixed severity tests.
    rng = np.random.default_rng(seed(base["base_id"], "perturbation", -1 if epoch is None else epoch))
    noise_values = np.empty((0, 3), dtype="f4")
    if sigma:
        order = np.argsort(base["noise_distance"][selected], kind="stable")
        ids = order[:max(1, round(count * noise_fraction))]
        noise_mask[ids] = True
        if config.get("distribution", "gaussian") == "laplace":
            noise_values = rng.laplace(0, sigma / np.sqrt(2), (len(ids), 3)).astype("f4")
        else:
            noise_values = (rng.standard_normal((len(ids), 3)) * sigma).astype("f4")
        noisy[ids] += noise_values
    if outlier_fraction:
        ids = rng.choice(count, max(1, round(count * outlier_fraction)), replace=False)
        directions = rng.standard_normal((len(ids), 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        noisy[ids] += (directions * rng.uniform(0.05, 0.15, (len(ids), 1))).astype("f4")
        outlier_mask[ids] = True
        noise_mask[ids] = True
    gt_hole = hole[base["gt_ref_index"]]
    gt_sparse = sparse[base["gt_ref_index"]]
    tree = cKDTree(pool)
    boundary_mask = np.zeros(count, dtype="u1")
    boundary_distance = np.zeros(count, dtype="f4")
    if hole.any():
        neighbors = tree.query(clean, k=min(13, size), workers=1)[1]
        boundary_mask = hole[neighbors].any(axis=1).astype("u1")
        boundary_distance = cKDTree(pool[hole]).query(clean, workers=1)[0].astype("f4")
    displacement = noisy - clean
    magnitude = np.linalg.norm(displacement, axis=1)
    reliability = np.exp(-(magnitude / max(float(base["spacing"]), 1e-8)) ** 2).astype("f4")
    reliability[outlier_mask] = 0
    # No truncated/rejection-sampled Gaussian: report noisy-coordinate intrusions.
    noisy_intrusions = int(hole[tree.query(noisy, workers=1)[1]].sum()) if hole.any() else 0
    if hole[selected].any():
        raise ValueError("Underlying clean observations refilled the removed region")
    quality = {
        "hole_fraction_reference": float(hole.mean()), "hole_fraction_gt": float(gt_hole.mean()),
        "clean_hole_intrusions": 0.0, "noisy_hole_intrusions": float(noisy_intrusions),
        "sparse_density_ratio_target": density, "sparse_density_ratio_actual": float(actual_density),
        "noise_sigma": sigma, "noise_sigma_empirical": float(noise_values.std()) if len(noise_values) else 0.0,
        "noise_point_fraction": float(noise_mask.mean()), "outlier_point_fraction": float(outlier_mask.mean()),
        "input_gt_exact_overlap": float(np.mean(cKDTree(base["gt_xyz"]).query(clean, workers=1)[0] < 1e-7)),
    }
    return {
        "input_xyz": noisy.astype("f4"), "gt_xyz": base["gt_xyz"], "clean_input_xyz": clean,
        "clean_input_normals": base["pool_normals"][selected], "gt_normals": base["gt_normals"],
        "gt_curvature": base["gt_curvature"], "gt_normal_variation": base["gt_normal_variation"],
        "denoise_offset": (clean - noisy).astype("f4"),
        "hole_mask_gt": gt_hole.astype("u1"), "sparse_mask_gt": gt_sparse.astype("u1"),
        "noise_mask_input": noise_mask.astype("u1"), "outlier_mask_input": outlier_mask.astype("u1"),
        "hole_boundary_mask_input": boundary_mask, "hole_boundary_distance_input": boundary_distance,
        "reliability_target_input": reliability, "selected_pool_index": selected.astype("i4"),
        "center": base["center"], "scale": base["scale"],
        "sample_id": base["base_id"] + "/" + condition, "base_id": base["base_id"],
        "object_id": base["object_id"], "category": base["category"], "object_id_known": np.uint8(1),
        "mesh_path": base["mesh_path"], "mesh_sha256": base["mesh_sha256"], "geometry_id": base["geometry_id"],
        "gt_sha256": base["gt_sha256"], "condition": condition, "corruption_type": np.uint8(kind),
        "region_spec": json.dumps({"protocol": VERSION, "area_measure": "uniform_surface_sample_fraction_MC",
                                    "hole_fraction": fraction, "density_ratio": density,
                                    "noise_distribution": config.get("distribution", "gaussian"),
                                    "noise_sigma": sigma, "noise_fraction": noise_fraction,
                                    "outlier_fraction": outlier_fraction, "hole_geometry": base["region_method"],
                                    "gaussian_rejection_sampling": False}, sort_keys=True),
        "quality": quality,
    }
