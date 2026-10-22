"""Independent manifest, field and label checks for training/evaluation contracts."""
from pathlib import Path
import json

import h5py
import numpy as np

from .pipeline import read_json, write_json, log
from .v3_core import CONDITIONS, VALIDATION_CONDITIONS
from .v3_pipeline import DEFAULT


def verify(root):
    root = Path(root)
    manifest = read_json(root / "metadata/base_manifest.json")
    base_rows, fixed_rows = 0, 0
    for group, entries in manifest["groups"].items():
        conditions = VALIDATION_CONDITIONS if group == "validation" else list(CONDITIONS)
        expected_names = {Path(entry["path"]).name for entry in entries}
        if group != "train":
            actual_names = {p.name for p in (root / "processed" / group).glob("*.h5")}
            if actual_names != expected_names:
                raise ValueError(f"Missing/extra evaluation shards in {group}")
        for entry in entries:
            with h5py.File(root / entry["path"], "r") as bases:
                count = len(bases["base_id"])
                if count != entry["rows"]:
                    raise ValueError("Base manifest row count mismatch")
                for key in ["pool_normals", "gt_normals"]:
                    normals = bases[key][:]
                    if not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals, axis=-1), 1, atol=2e-6):
                        raise ValueError("Reference normal field is not finite and unit length")
                for key in ["gt_curvature", "gt_normal_variation"]:
                    values = bases[key][:]
                    if not np.isfinite(values).all() or values.min() < -1e-7 or values.max() > 1 + 1e-7:
                        raise ValueError("Invalid reference geometry targets")
                base_rows += count
                if group == "train":
                    continue
                path = root / "processed" / group / Path(entry["path"]).name
                with h5py.File(path, "r") as samples:
                    expected_ids = [base_id + "/" + condition for base_id in bases["base_id"].asstr()[:] for condition in conditions]
                    if list(samples["sample_id"].asstr()[:]) != expected_ids:
                        raise ValueError("Fixed condition/order/identity contract mismatch")
                    n, m = (2048, 8192) if group == "pu1k_object" else (256, 1024)
                    if samples["input_xyz"].shape != (len(expected_ids), n, 3) or samples["gt_xyz"].shape != (len(expected_ids), m, 3):
                        raise ValueError("Input/output point count contract mismatch")
                    reliability = samples["reliability_target_input"][:]
                    if not np.isfinite(reliability).all() or reliability.min() < 0 or reliability.max() > 1:
                        raise ValueError("Reliability target outside [0,1]")
                    if np.any(reliability[samples["outlier_mask_input"][:].astype(bool)] != 0):
                        raise ValueError("Injected outliers have nonzero reliability targets")
                    for base_index in range(count):
                        start = base_index * len(conditions)
                        stop = start + len(conditions)
                        selection = samples["selected_pool_index"][start:stop]
                        pool = bases["pool_xyz"][base_index]
                        if selection.min() < 0 or selection.max() >= len(pool):
                            raise ValueError("Candidate index outside independent pool")
                        if not np.array_equal(pool[selection], samples["clean_input_xyz"][start:stop]):
                            raise ValueError("Clean inputs were not drawn from their independent surface pool")
                        if not np.all(samples["gt_xyz"][start:stop] == bases["gt_xyz"][base_index]):
                            raise ValueError("Evaluation target differs from the frozen base target")
                        if not np.all(samples["scale"][start:stop] == bases["scale"][base_index]):
                            raise ValueError("Canonical scale changes across conditions")
                        ranks = bases["hole_rank"][base_index]
                        gt_ref = bases["gt_ref_index"][base_index]
                        sparse = bases["sparse_mask_ref"][base_index]
                        for offset, condition in enumerate(conditions):
                            row = start + offset
                            config = CONDITIONS[condition]
                            hole = ranks < round(len(pool) * config.get("hole", 0))
                            if not np.array_equal(samples["hole_mask_gt"][row], hole[gt_ref]):
                                raise ValueError("Hole reference labels do not match the frozen spatial region")
                            if hole[selection[offset]].any():
                                raise ValueError("An underlying clean observation lies in the deleted region")
                            expected_sparse = sparse[gt_ref] if config["kind"] in (1, 4) else np.zeros(m, dtype="u1")
                            if not np.array_equal(samples["sparse_mask_gt"][row], expected_sparse):
                                raise ValueError("Sparse reference label mismatch")
                    fixed_rows += len(expected_ids)
        log(f"V3 contract checks: {group} complete")
    result = {"status": "passed", "clean_bases_checked": base_rows, "fixed_records_checked": fixed_rows,
              "manifest_files_counts_and_order": "passed", "fixed_input_pool_correspondence": "passed",
              "reference_normals_and_geometry_targets": "passed", "hole_and_sparse_labels_recomputed": "passed",
              "fixed_coordinate_scale_and_reference_targets": "passed"}
    write_json(root / "checks/contract_validation.json", result)
    return result


if __name__ == "__main__":
    verify(DEFAULT["output_root"])
