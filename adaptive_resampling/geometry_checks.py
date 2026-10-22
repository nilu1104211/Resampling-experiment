"""Independent checks of persisted mesh GT origins and hole membership."""
from pathlib import Path
import json

import h5py
import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .core import point_graph, read_off
from .pipeline import DEFAULT, file_hash, log, write_json


def verify(config=DEFAULT, count=20):
    root = Path(config["output_root"])
    rows_checked = 0
    surface_points_checked = 0
    verified_origins = set()
    largest_surface_error = 0.0
    largest_canonical_error = 0.0
    cache_records = {}
    surface_records = []
    for dataset in ["pu1k_test_patch", "modelnet40_test_patch"]:
        for condition in ["C2_hole_10", "E1_mixed"]:
            path = root / "processed" / dataset / f"{condition}.h5"
            with h5py.File(path, "r") as h5:
                ids = np.linspace(0, len(h5["input_xyz"]) - 1, min(count, len(h5["input_xyz"])), dtype=int)
                for index in ids:
                    spec = json.loads(h5["region_spec"].asstr()[index])
                    cache_name, patch = spec["cache"], int(spec["patch_index"])
                    cache = root / "cache/mesh_patches" / cache_name
                    if cache_name not in cache_records:
                        with np.load(cache) as archive:
                            cache_records[cache_name] = {key: archive[key] for key in archive.files}
                    arrays = cache_records[cache_name]
                    center, scale = h5["center"][index], float(h5["scale"][index])
                    raw_gt = arrays[f"gt_{patch}"]
                    target = ((raw_gt - center) / scale).astype("f4")
                    stored_gt = h5["gt_xyz"][index]
                    largest_canonical_error = max(largest_canonical_error, float(np.abs(target - stored_gt).max()))
                    if not np.array_equal(target, stored_gt):
                        raise ValueError("Stored GT differs from independently transformed clean cache")
                    origin = (cache_name, patch)
                    if origin not in verified_origins:
                        mesh_path = Path(spec["mesh_path"])
                        if file_hash(mesh_path) != spec["mesh_sha256"]:
                            raise ValueError("Original mesh changed after generation")
                        vertices, triangles, _, _ = read_off(mesh_path)
                        face = vertices[triangles[arrays[f"gt_face_index_{patch}"]]]
                        restored = np.sum(face * arrays[f"gt_barycentric_{patch}"][:, :, None], axis=1)
                        relative_error = float(np.linalg.norm(restored - raw_gt, axis=1).max() / scale)
                        if relative_error > 1e-9:
                            raise ValueError("GT cannot be reconstructed from original mesh triangles")
                        largest_surface_error = max(largest_surface_error, relative_error)
                        surface_points_checked += len(raw_gt)
                        verified_origins.add(origin)
                        surface_records.append({"mesh": str(mesh_path), "patch": patch, "points": len(raw_gt), "relative_surface_error": relative_error})
                    reference = ((arrays[f"pool_{patch}"] - center) / scale).astype("f4")
                    tree, graph = point_graph(reference)
                    distances = dijkstra(graph, directed=False, indices=int(spec["hole_reference_node"]))
                    order = np.argsort(distances, kind="stable")
                    region = np.zeros(len(reference), dtype=bool)
                    region[order[:int(spec["hole_count"])]] = True
                    mapped_gt = region[tree.query(stored_gt, workers=1)[1]]
                    if not np.array_equal(mapped_gt, h5["hole_mask_gt"][index].astype(bool)):
                        raise ValueError("Persisted hole labels differ from independently reconstructed region")
                    if region[tree.query(h5["input_xyz"][index], workers=1)[1]].any():
                        raise ValueError("Persisted input refilled a hole")
                    selected = np.flatnonzero(region)
                    if connected_components(graph[selected][:, selected], directed=False)[0] != 1:
                        raise ValueError("Persisted hole is not graph-connected")
                    rows_checked += 1
            log(f"Independent geometry check: {dataset}/{condition}, {len(ids)} rows")
    report = {"status": "passed", "hole_rows_checked": rows_checked,
              "gt_surface_points_checked": surface_points_checked,
              "unique_gt_patches_checked": len(verified_origins),
              "max_raw_surface_error_over_canonical_scale": largest_surface_error,
              "max_canonical_cache_to_h5_error": largest_canonical_error,
              "original_mesh_hashes_verified": True, "surface_records": surface_records,
              "connectivity_interpretation": "reference sample graph only; not exact mesh-geodesic certification"}
    write_json(root / "checks/independent_geometry_verification.json", report)
    return report


if __name__ == "__main__":
    verify()
