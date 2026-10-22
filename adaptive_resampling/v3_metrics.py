"""Shared V3 evaluation, including exact point-to-triangle mesh distances."""
from functools import lru_cache
import heapq

import numpy as np
from scipy.spatial import cKDTree

from .core import read_off
from .metrics import evaluate_points
from .v3_dataset import inverse_prediction


def point_triangle_squared(point, triangles):
    """Squared distances to triangle interiors/edges/vertices; no centroid proxy."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    normals = np.cross(ab, ac)
    norm2 = np.einsum("ij,ij->i", normals, normals)
    signed = np.einsum("ij,ij->i", point - a, normals)
    projection = point - (signed / np.maximum(norm2, 1e-30))[:, None] * normals
    relative = projection - a
    aa = np.einsum("ij,ij->i", ab, ab)
    bb = np.einsum("ij,ij->i", ab, ac)
    cc = np.einsum("ij,ij->i", ac, ac)
    ar = np.einsum("ij,ij->i", ab, relative)
    cr = np.einsum("ij,ij->i", ac, relative)
    denominator = aa * cc - bb * bb
    u = (cc * ar - bb * cr) / np.maximum(denominator, 1e-30)
    v = (aa * cr - bb * ar) / np.maximum(denominator, 1e-30)
    interior = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (norm2 > 1e-30)
    distance = np.where(interior, signed * signed / np.maximum(norm2, 1e-30), np.inf)
    for start, end in [(a, b), (b, c), (c, a)]:
        edge = end - start
        parameter = np.clip(np.einsum("ij,ij->i", point - start, edge) / np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-30), 0, 1)
        difference = point - (start + parameter[:, None] * edge)
        distance = np.minimum(distance, np.einsum("ij,ij->i", difference, difference))
    return np.maximum(distance, 0)


class TriangleBVH:
    """Exact branch-and-bound over triangle AABBs, with bounded leaf work."""
    def __init__(self, triangles, leaf_size=16):
        self.triangles = np.asarray(triangles, dtype="f8")
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3) or not len(self.triangles):
            raise ValueError("Expected nonempty triangles with shape T x 3 x 3")
        self.nodes = []
        low, high = self.triangles.min(1), self.triangles.max(1)
        centers = (low + high) / 2

        def build(ids):
            index = len(self.nodes)
            node = [low[ids].min(0), high[ids].max(0), None, None, None]
            self.nodes.append(node)
            if len(ids) <= leaf_size:
                node[4] = ids
            else:
                axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
                order = np.argpartition(centers[ids, axis], len(ids) // 2)
                middle = len(ids) // 2
                node[2] = build(ids[order[:middle]])
                node[3] = build(ids[order[middle:]])
            return index
        build(np.arange(len(self.triangles)))

    def query(self, points):
        distances, face_ids = [], []
        for point in np.asarray(points, dtype="f8"):
            def bound(index):
                low, high = self.nodes[index][:2]
                delta = np.maximum(np.maximum(low - point, point - high), 0)
                return float(delta @ delta)
            queue = [(bound(0), 0)]
            best, closest = np.inf, -1
            while queue:
                lower, index = heapq.heappop(queue)
                if lower > best:
                    break
                node = self.nodes[index]
                if node[4] is not None:
                    ids = node[4]
                    values = point_triangle_squared(point, self.triangles[ids])
                    candidate = int(np.argmin(values))
                    if values[candidate] < best:
                        best, closest = float(values[candidate]), int(ids[candidate])
                else:
                    for child in node[2:4]:
                        lower = bound(child)
                        if lower <= best:
                            heapq.heappush(queue, (lower, child))
            distances.append(np.sqrt(best))
            face_ids.append(closest)
        return np.asarray(distances), np.asarray(face_ids)


@lru_cache(maxsize=2)
def mesh_index(path):
    vertices, faces, _, _ = read_off(path)
    return TriangleBVH(vertices[faces])


def evaluate_sample(prediction, sample, *, coordinates="model", include_p2f=False, coverage_threshold=0.01):
    """Evaluate in the common canonical frame; keep physical scales explicit."""
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    prediction = np.asarray(prediction)
    target = inverse_prediction(sample["gt_xyz"], sample)
    if coordinates == "model":
        prediction = inverse_prediction(prediction, sample)
    elif coordinates == "raw":
        prediction = (prediction - sample["canonical_center"]) / sample["canonical_scale"]
    elif coordinates != "canonical":
        raise ValueError("coordinates must be model, canonical, or raw")
    if prediction.shape != target.shape:
        raise ValueError(f"Fixed-budget evaluation requires {len(target)} output points, got {len(prediction)}")
    labels = sample.get("supervision", {})
    result = evaluate_points(prediction, target, hole_mask_gt=labels.get("hole_mask_gt"), sparse_mask_gt=labels.get("sparse_mask_gt"))
    forward = cKDTree(target).query(prediction, workers=1)[0]
    backward = cKDTree(prediction).query(target, workers=1)[0]
    precision, recall = float(np.mean(forward <= coverage_threshold)), float(np.mean(backward <= coverage_threshold))
    result.update(coverage_threshold=coverage_threshold, precision=precision, recall=recall,
                  f_score=2 * precision * recall / max(precision + recall, 1e-15))
    for name in ["hole", "sparse"]:
        mask = labels.get(name + "_mask_gt")
        if mask is not None and np.any(mask):
            result[name + "_coverage_fraction"] = float(np.mean(backward[np.asarray(mask, dtype=bool)] <= coverage_threshold))
    if "gt_normal_variation" in labels:
        variation = np.asarray(labels["gt_normal_variation"])
        if np.ptp(variation) > 1e-6:
            detail = variation >= np.quantile(variation, 0.90)
            result["detail_coverage_mean_l2"] = float(backward[detail].mean())
            result["detail_reference_fraction"] = float(detail.mean())
    if include_p2f:
        if "reference" not in sample:
            raise ValueError("Load include_reference_metadata=True for point-to-mesh evaluation")
        raw_prediction = prediction * sample["canonical_scale"] + sample["canonical_center"]
        distances, _ = mesh_index(sample["reference"]["mesh_path"]).query(raw_prediction)
        canonical_distances = distances / sample["canonical_scale"]
        result.update(p2f_mean_l2=float(canonical_distances.mean()),
                      p2f_rmse=float(np.sqrt(np.mean(canonical_distances ** 2))),
                      p2f_max_l2=float(canonical_distances.max()))
    return result
