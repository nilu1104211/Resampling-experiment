from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

CONDITIONS = {
    "A0_normal": (0, 0.0),
    "B1_sparse_light": (1, 0.75),
    "B2_sparse_mid": (1, 0.50),
    "B3_sparse_heavy": (1, 0.25),
    "C1_hole_05": (2, 0.05),
    "C2_hole_10": (2, 0.10),
    "C3_hole_20": (2, 0.20),
    "D1_noise_002": (3, 0.002),
    "D2_noise_005": (3, 0.005),
    "D3_noise_010": (3, 0.010),
    "E1_mixed": (4, 0.50),
}
REPRESENTATIVES = ["A0_normal", "B2_sparse_mid", "C2_hole_10", "D2_noise_005", "E1_mixed"]
GLOBAL_SEED = 20261020


def stable_seed(*parts, seed=None):
    seed = GLOBAL_SEED if seed is None else seed
    data = "|".join(map(str, (seed,) + parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little") & ((1 << 63) - 1)


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()


def normalize_pair(clean_input, gt):
    """Canonical transform: statistics of clean input, never GT statistics."""
    clean_input = np.asarray(clean_input, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if not np.isfinite(clean_input).all() or not np.isfinite(gt).all():
        raise ValueError("Non-finite source coordinates")
    center = clean_input.mean(axis=0)
    scale = float(np.linalg.norm(clean_input - center, axis=1).max())
    if scale <= 1e-12:
        raise ValueError("Degenerate source input")
    return ((clean_input - center) / scale).astype("f4"), ((gt - center) / scale).astype("f4"), center, scale


def input_transform(input_xyz, gt_xyz=None):
    """Deployable transform, using only the actual (possibly degraded) input."""
    x = np.asarray(input_xyz, dtype=np.float64)
    center = x.mean(axis=0)
    scale = float(np.linalg.norm(x - center, axis=1).max())
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("Degenerate model input")
    normalized = ((x - center) / scale).astype("f4")
    target = None if gt_xyz is None else ((np.asarray(gt_xyz) - center) / scale).astype("f4")
    return normalized, target, center.astype("f4"), scale


def _triangulate_polygon(ids, vertices):
    """Ear clipping, with a fast path for convex polygons. No fan across concavities."""
    ids = list(ids)
    if len(ids) == 3:
        return [tuple(ids)]
    xyz = vertices[ids]
    normal = np.cross(xyz, np.roll(xyz, -1, axis=0)).sum(axis=0)
    if np.linalg.norm(normal) < 1e-15:
        return []
    points = np.delete(xyz, np.argmax(np.abs(normal)), axis=1)
    signed_area = np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
    direction = 1.0 if signed_area >= 0 else -1.0
    tolerance = max(float(np.ptp(points, axis=0).max()) ** 2 * 1e-12, 1e-20)

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    turns = [direction * cross(points[i - 1], points[i], points[(i + 1) % len(ids)]) for i in range(len(ids))]
    if min(turns) >= -tolerance:
        return [(ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1)]
    remaining = list(range(len(ids)))
    result = []
    while len(remaining) > 3:
        found = False
        for j in range(len(remaining)):
            a, b, c = remaining[j - 1], remaining[j], remaining[(j + 1) % len(remaining)]
            if direction * cross(points[a], points[b], points[c]) <= tolerance:
                continue
            contains = False
            for q in remaining:
                if q in (a, b, c):
                    continue
                tests = [direction * cross(points[a], points[b], points[q]), direction * cross(points[b], points[c], points[q]), direction * cross(points[c], points[a], points[q])]
                if min(tests) >= -tolerance:
                    contains = True
                    break
            if not contains:
                result.append((ids[a], ids[b], ids[c]))
                remaining.pop(j)
                found = True
                break
        if not found:
            raise ValueError("Non-simple/degenerate polygon cannot be triangulated safely")
    result.append(tuple(ids[i] for i in remaining))
    return result


def read_off(path):
    """Read OFF and OFF<count...>, comments, polygonal faces and optional face colors."""
    def content_line(handle):
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected end of OFF: {path}")
            line = line.split("#", 1)[0].strip()
            if line:
                return line

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        first = content_line(handle)
        if not first.startswith("OFF"):
            raise ValueError(f"Unsupported OFF header: {first[:40]}")
        counts = first[3:].strip() or content_line(handle)
        nv, nf, _ = map(int, counts.split()[:3])
        if nv < 3 or nf < 1:
            raise ValueError("Empty mesh")
        vertex_text = "\n".join(content_line(handle) for _ in range(nv))
        vertices = np.loadtxt(io.StringIO(vertex_text), usecols=(0, 1, 2), ndmin=2)
        face_text = handle.read()
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite mesh vertices")
    try:
        matrix = np.loadtxt(io.StringIO(face_text), dtype=np.int64, ndmin=2, max_rows=nf)
    except ValueError:
        matrix = None
    if matrix is not None and len(matrix) == nf and np.all(matrix[:, 0] == 3):
        triangles = matrix[:, 1:4].copy()
    else:
        triangles = []
        read_faces = 0
        for line in face_text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            count = int(fields[0])
            ids = [int(x) for x in fields[1:1 + count]]
            if len(ids) != count or count < 3:
                raise ValueError("Malformed OFF face")
            if min(ids) < 0 or max(ids) >= nv:
                raise ValueError("OFF vertex index out of range")
            triangles.extend(_triangulate_polygon(ids, vertices))
            read_faces += 1
            if read_faces == nf:
                break
        if read_faces != nf:
            raise ValueError("OFF face count mismatch")
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if triangles.min() < 0 or triangles.max() >= nv:
        raise ValueError("OFF vertex index out of range")
    edges1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edges2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    area = np.linalg.norm(np.cross(edges1, edges2), axis=1) * 0.5
    keep = np.isfinite(area) & (area > 0)
    if not keep.any():
        raise ValueError("Mesh has no non-degenerate surface")
    return vertices, triangles[keep], area[keep], {"vertices": nv, "faces": nf, "triangles": int(keep.sum()), "zero_area_triangles_removed": int((~keep).sum()), "header_inline": first != "OFF"}


def sample_surface(vertices, triangles, area, count, rng):
    face_id = np.searchsorted(np.cumsum(area), rng.random(count) * area.sum(), side="right")
    face_id = np.minimum(face_id, len(triangles) - 1)
    face = vertices[triangles[face_id]]
    r = np.sqrt(rng.random(count))
    s = rng.random(count)
    barycentric = np.column_stack((1 - r, r * (1 - s), r * s))
    points = np.einsum("ni,nij->nj", barycentric, face)
    return points, face_id, barycentric


def point_graph(points, k=12):
    tree = cKDTree(points)
    distances, ids = tree.query(points, k=min(k + 1, len(points)), workers=1)
    rows = np.repeat(np.arange(len(points)), ids.shape[1] - 1)
    columns = ids[:, 1:].reshape(-1)
    weights = np.maximum(distances[:, 1:].reshape(-1), 1e-10)
    graph = coo_matrix((weights, (rows, columns)), shape=(len(points), len(points))).tocsr()
    graph = graph.maximum(graph.T)
    return tree, graph


def geodesic_order(graph, rng, minimum_fraction=0.40):
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    # A region stays in one component. Choose a sufficiently large component.
    eligible = np.flatnonzero(sizes >= max(2, int(len(labels) * minimum_fraction)))
    if not len(eligible):
        eligible = np.array([int(np.argmax(sizes))])
    component = int(eligible[np.argmax(sizes[eligible])])
    candidates = np.flatnonzero(labels == component)
    center = int(rng.choice(candidates))
    distances = dijkstra(graph, directed=False, indices=center)
    order = np.argsort(distances, kind="stable")
    finite = np.isfinite(distances[order])
    return order[finite], distances, center, count


def make_base(clean_input, gt, pool=None, *, base_id, object_id="", object_known=False,
              source_index=-1, metadata=None, deduplicate_reference=False):
    clean, target, center, scale = normalize_pair(clean_input, gt)
    reference = target.copy() if pool is None else ((np.asarray(pool) - center) / scale).astype("f4")
    metadata = dict(metadata or {})
    if deduplicate_reference:
        # Some official object references repeat exactly coincident coordinates.
        # Region membership must be a spatial property, not two different labels
        # attached to the same location. Keep original GT and sparse input intact.
        _, first = np.unique(reference, axis=0, return_index=True)
        keep = np.sort(first)
        metadata["reference_coincident_points_removed"] = int(len(reference) - len(keep))
        metadata["reference_policy"] = "unique_float32_positions_preserve_first_occurrence_v1"
        reference = reference[keep]
    tree, graph = point_graph(reference)
    hole_order, hole_distance, hole_center, components = geodesic_order(graph, np.random.default_rng(stable_seed(base_id, "hole")))
    sparse_order, sparse_distance, sparse_center, _ = geodesic_order(graph, np.random.default_rng(stable_seed(base_id, "sparse")))
    _, noise_distance, noise_center, _ = geodesic_order(graph, np.random.default_rng(stable_seed(base_id, "noise")))
    gt_ref = tree.query(target, workers=1)[1]
    clean_ref = tree.query(clean, workers=1)[1]
    rank = np.empty(len(reference), dtype=np.int64)
    rank.fill(len(reference))
    rank[hole_order] = np.arange(len(hole_order))
    sparse_mask = np.zeros(len(reference), dtype=bool)
    sparse_mask[sparse_order[:max(1, round(len(reference) * 0.30))]] = True
    return {"base_id": base_id, "object_id": object_id, "object_known": object_known,
            "source_index": source_index, "clean": clean, "gt": target, "pool": reference,
            "center": center, "scale": scale, "tree": tree, "graph": graph,
            "hole_order": hole_order, "hole_rank": rank, "hole_distance": hole_distance,
            "hole_center": hole_center, "sparse_mask": sparse_mask,
            "sparse_center": sparse_center, "noise_distance": noise_distance,
            "noise_center": noise_center, "gt_ref": gt_ref, "clean_ref": clean_ref,
            "graph_components": components, "metadata": metadata, "gt_hash": array_hash(target)}


def _choose(pool_ids, count, rng):
    if len(pool_ids) < count:
        raise ValueError(f"Insufficient clean candidate points: {len(pool_ids)} < {count}")
    return rng.choice(pool_ids, size=count, replace=False)


def corrupt(base, condition, *, input_count=None):
    kind, severity = CONDITIONS[condition]
    count = len(base["clean"]) if input_count is None else input_count
    rng = np.random.default_rng(stable_seed(base["base_id"], condition, "sampling"))
    pool = base["pool"]
    size = len(pool)
    fraction = severity if kind == 2 else 0.10 if kind == 4 else 0.0
    hole_count = round(size * fraction)
    if hole_count > len(base["hole_order"]):
        raise ValueError("Requested hole does not fit a connected reference component")
    hole = base["hole_rank"] < hole_count
    sparse = base["sparse_mask"] if kind in (1, 4) else np.zeros(size, dtype=bool)
    eligible = np.flatnonzero(~hole)
    density_ratio = severity if kind == 1 else 0.50 if kind == 4 else 1.0
    actual_density_ratio = 1.0
    if kind in (0, 3):
        clean = base["clean"].copy()
        selected = base["clean_ref"].copy()
    elif kind in (1, 4):
        sparse_ids = eligible[sparse[eligible]]
        normal_ids = eligible[~sparse[eligible]]
        area_fraction = len(sparse_ids) / len(eligible)
        quota = int(round(count * density_ratio * area_fraction / (1 - area_fraction + density_ratio * area_fraction)))
        quota = max(1, min(count - 1, quota))
        selected = np.concatenate((_choose(sparse_ids, quota, rng), _choose(normal_ids, count - quota, rng)))
        rng.shuffle(selected)
        clean = pool[selected].copy()
        actual_density_ratio = (quota / len(sparse_ids)) / ((count - quota) / len(normal_ids))
    else:
        selected = _choose(eligible, count, rng)
        clean = pool[selected].copy()
    noisy = clean.copy()
    noise_mask = np.zeros(count, dtype=bool)
    sigma = severity if kind == 3 else 0.005 if kind == 4 else 0.0
    rejections = 0
    if sigma:
        noise_order = np.argsort(base["noise_distance"][selected], kind="stable")
        noise_ids = noise_order[:max(1, round(count * 0.25))]
        noise_mask[noise_ids] = True
        noise_rng = np.random.default_rng(stable_seed(base["base_id"], "noise_standard_normal"))
        noisy[noise_ids] += (noise_rng.standard_normal((len(noise_ids), 3)) * sigma).astype("f4")
        if hole_count:
            # Keep noise from moving selected points back into the recorded hole cells.
            for _ in range(100):
                classified = base["tree"].query(noisy[noise_ids], workers=1)[1]
                bad = hole[classified]
                if not bad.any():
                    break
                bad_ids = noise_ids[bad]
                rejections += len(bad_ids)
                noisy[bad_ids] = clean[bad_ids] + noise_rng.standard_normal((len(bad_ids), 3)) * sigma
            else:
                raise ValueError("Could not preserve empty hole under local noise")
    _, _, model_center, model_scale = input_transform(noisy)
    gt_hole = hole[base["gt_ref"]]
    gt_sparse = sparse[base["gt_ref"]]
    displacement = noisy - clean
    hole_intrusions = int(hole[base["tree"].query(noisy, workers=1)[1]].sum()) if hole_count else 0
    induced = base["graph"][np.flatnonzero(hole)][:, np.flatnonzero(hole)]
    hole_components = connected_components(induced, directed=False)[0] if hole_count else 0
    if hole_count and hole_components != 1:
        raise ValueError("Hole is not connected on the reference graph")
    if hole_intrusions:
        raise ValueError("Corruption refilled the recorded hole")
    if not np.isfinite(noisy).all():
        raise ValueError("Non-finite corruption output")
    if not np.array_equal(displacement[~noise_mask], np.zeros_like(displacement[~noise_mask])):
        raise ValueError("Noise escaped the labelled region")
    noise_values = displacement[noise_mask]
    info = {"hole_fraction_reference": float(hole.mean()), "hole_fraction_gt": float(gt_hole.mean()),
            "hole_components": int(hole_components), "hole_intrusions": hole_intrusions,
            "sparse_density_ratio_target": float(density_ratio), "sparse_density_ratio_actual": float(actual_density_ratio),
            "noise_sigma": float(sigma), "noise_sigma_empirical": float(noise_values.std()) if len(noise_values) else 0.0,
            "noise_point_fraction": float(noise_mask.mean()), "noise_hole_rejections": int(rejections),
            "model_scale": float(model_scale), "graph_components": int(base["graph_components"])}
    region = {"method": "symmetric_knn12_geodesic_discrete_v1", "hole_reference_node": int(base["hole_center"]),
              "hole_count": int(hole_count), "reference_count": size,
              "sparse_reference_node": int(base["sparse_center"]), "noise_reference_node": int(base["noise_center"]),
              "area_measure": base["metadata"].get("area_measure", "discrete_GT_point_fraction_not_physical_area"),
              **base["metadata"]}
    return {"input_xyz": noisy.astype("f4"), "gt_xyz": base["gt"], "clean_input_xyz": clean.astype("f4"),
            "hole_mask_gt": gt_hole.astype("u1"), "sparse_mask_gt": gt_sparse.astype("u1"),
            "noise_mask_input": noise_mask.astype("u1"), "center": base["center"].astype("f8"),
            "scale": np.float64(base["scale"]), "inference_center": model_center,
            "inference_scale": np.float32(model_scale), "seed": np.int64(stable_seed(base["base_id"], condition, "sampling")),
            "corruption_type": np.uint8(kind), "severity": np.float32(severity),
            "source_index": np.int64(base["source_index"]), "object_id_known": np.uint8(base["object_known"]),
            "sample_id": base["base_id"] + "/" + condition, "base_id": base["base_id"],
            "object_id": base["object_id"], "condition": condition, "gt_sha256": base["gt_hash"],
            "region_spec": json.dumps(region, sort_keys=True), "quality": info}
