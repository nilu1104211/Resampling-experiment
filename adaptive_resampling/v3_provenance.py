"""Upgrade only stored surface-provenance weights, without changing any points."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from .core import read_off, sample_surface
from .pipeline import read_json, write_json, log
from .v3_core import seed
from .v3_pipeline import DEFAULT


def upgrade(root):
    root = Path(root)
    manifest = read_json(root / "metadata/base_manifest.json")
    config = read_json(root / "metadata/generation_config.json")
    tasks = [(group, entry) for group, entries in manifest["groups"].items() for entry in entries]

    def one(task):
        group, entry = task
        path = root / entry["path"]
        with h5py.File(path, "r") as handle:
            if handle["gt_barycentric"].dtype == np.dtype("f8"):
                return {"updated": False, "bases": len(handle["base_id"]), "error": 0.0}
            mesh_path = handle["mesh_path"].asstr()[0]
            object_id = handle["object_id"].asstr()[0]
            ids = handle["gt_sample_index"][:]
            expected_faces = handle["gt_face_index"][:]
            target = handle["gt_xyz"][:]
            centers, scales = handle["center"][:], handle["scale"][:]
        vertices, triangles, areas, _ = read_off(mesh_path)
        rng = np.random.default_rng(seed(object_id, group, "surface"))
        points, face_ids, barycentric = sample_surface(vertices, triangles, areas, config["surface_samples"], rng)
        if not np.array_equal(face_ids[ids], expected_faces):
            raise ValueError("Surface RNG replay changed original triangle identities")
        restored = ((points[ids] - centers[:, None, :]) / scales[:, None, None]).astype("f4")
        if not np.array_equal(restored, target):
            raise ValueError("Surface replay would change target coordinates")
        precise = barycentric[ids].astype("f8")
        with h5py.File(path, "r+") as handle:
            staged = "_gt_barycentric_float64"
            if staged in handle:
                del handle[staged]
            handle.create_dataset(staged, data=precise, chunks=(1,) + precise.shape[1:], compression="lzf")
            handle.flush()
            del handle["gt_barycentric"]
            handle.move(staged, "gt_barycentric")
            handle.attrs["surface_provenance_precision"] = "float64"
            handle.flush()
        return {"updated": True, "bases": len(target), "error": 0.0}

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(tasks), 40):
            results.extend(pool.map(one, tasks[offset:offset + 40]))
            if len(results) % 200 == 0 or len(results) == len(tasks):
                log(f"V3 float64 provenance: {len(results)}/{len(tasks)} object files")
    report = {"status": "passed", "object_files": len(results), "upgraded_object_files": sum(r["updated"] for r in results),
              "clean_bases": sum(r["bases"] for r in results), "all_target_coordinates_bitwise_unchanged": True,
              "surface_triangle_identities_replayed": True, "stored_barycentric_dtype": "float64"}
    write_json(root / "checks/provenance_precision_upgrade.json", report)
    return report


if __name__ == "__main__":
    upgrade(DEFAULT["output_root"])
