"""Shared geometric metrics. Distances are evaluated in canonical coordinates."""
import numpy as np
from scipy.spatial import cKDTree


def evaluate_points(prediction, target, *, hole_mask_gt=None, sparse_mask_gt=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    for name, points in [("prediction", prediction), ("target", target)]:
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"{name} must be a nonempty finite N x 3 array")
    pred_to_gt = cKDTree(target).query(prediction, workers=1)[0]
    gt_to_pred = cKDTree(prediction).query(target, workers=1)[0]
    result = {
        "cd_l1": float((pred_to_gt.mean() + gt_to_pred.mean()) * 0.5),
        "cd_l2_squared": float(((pred_to_gt ** 2).mean() + (gt_to_pred ** 2).mean()) * 0.5),
        "hausdorff_l2": float(max(pred_to_gt.max(), gt_to_pred.max())),
        "gt_coverage_mean_l2": float(gt_to_pred.mean()),
        "prediction_points": len(prediction), "target_points": len(target),
    }
    for name, mask in [("hole", hole_mask_gt), ("sparse", sparse_mask_gt)]:
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(target),):
            raise ValueError("Region mask must match target points")
        result[name + "_gt_fraction"] = float(mask.mean())
        if mask.any():
            result[name + "_coverage_mean_l2"] = float(gt_to_pred[mask].mean())
            result[name + "_coverage_mean_squared_l2"] = float((gt_to_pred[mask] ** 2).mean())
            result[name + "_coverage_max_l2"] = float(gt_to_pred[mask].max())
            nearest_region = mask[cKDTree(target).query(prediction, workers=1)[1]]
            result[name + "_prediction_fraction_by_nearest_gt"] = float(nearest_region.mean())
    return result
