"""Build V3 from real meshes, with object-level splits and independent samples.

python -m adaptive_resampling.v3_pipeline all
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .core import read_off, sample_surface, array_hash
from .pipeline import file_hash, write_json, read_json, log
from .v3_core import VERSION, SEED, CONDITIONS, VALIDATION_CONDITIONS, prepare_base, corrupt_base, seed

DEFAULT = {
    "version": VERSION, "seed": SEED,
    "output_root": "D:/dkwork/scdata/adaptive_resampling_data_v3",
    "raw_root": "D:/dkwork/baseline_compare/data",
    "pu1k_mesh_root": "D:/dkwork/baseline_compare/data/PU1K_raw_meshes",
    "primary_training_dataset": "pu1k", "validation_fraction": 0.10,
    "train_patches_per_object": 64, "validation_patches_per_object": 4,
    "test_patches_per_object": 4, "modelnet_test_patches_per_object": 1,
    "surface_samples": 32768, "local_samples": 4096,
    "input_points": 256, "target_points": 1024, "workers": 4,
    "conditions": CONDITIONS,
    "test_distributions_excluded_from_training": ["laplace"],
    "include_whole_object_tests": True,
}


def digest_config(config):
    relevant = {k: v for k, v in config.items() if k not in {"workers"}}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def initialize(config):
    root = Path(config["output_root"])
    for sub in ["metadata", "checks", "train_bases/pu1k", "validation_bases/pu1k", "test_bases/pu1k_patch",
                "test_bases/modelnet40_patch", "test_bases/pu1k_object", "processed", "visualization"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    destination = root / "metadata/generation_config.json"
    if destination.exists() and digest_config(read_json(destination)) != digest_config(config):
        raise ValueError("V3 configuration differs; choose another output directory")
    write_json(destination, config)
    return root


def geometry_fingerprint(vertices):
    """Translation/positive-scale and vertex-order invariant exact-geometry grouping.

This does not claim to identify rotated/remeshed semantic duplicates.
"""
    center = (vertices.min(0) + vertices.max(0)) / 2
    scale = float(np.max(np.ptp(vertices, axis=0)))
    if scale <= 1e-12:
        raise ValueError("Degenerate mesh bounds")
    quantized = np.rint((vertices - center) / scale * 1e6).astype("i8")
    unique = np.unique(quantized, axis=0)
    return hashlib.sha256(unique.astype("<i8").tobytes()).hexdigest()


def inspect_mesh(task):
    dataset, split, path = task
    path = Path(path)
    vertices, triangles, areas, _ = read_off(path)
    category = path.stem.split(".")[0] if dataset == "pu1k" else path.parent.parent.name
    return {"dataset": dataset, "source_split": split, "path": str(path), "object_id": dataset + "/" + path.stem,
            "category": category, "sha256": file_hash(path), "geometry_id": geometry_fingerprint(vertices),
            "vertices": len(vertices), "triangles": len(triangles), "surface_area_raw": float(areas.sum())}


def object_split(records, fraction=0.10):
    """Assign entire geometry groups, stratified by source category."""
    groups = defaultdict(list)
    for record in records:
        groups[record["geometry_id"]].append(record)
    categories = defaultdict(list)
    for geometry, members in groups.items():
        categories[min(m["category"] for m in members)].append(geometry)
    train, validation = [], []
    for category, geometries in sorted(categories.items()):
        geometries.sort(key=lambda item: seed("object_split", category, item))
        count = max(1, round(len(geometries) * fraction)) if len(geometries) > 1 else 0
        for i, geometry in enumerate(geometries):
            (validation if i < count else train).extend(groups[geometry])
    return sorted(train, key=lambda r: r["object_id"]), sorted(validation, key=lambda r: r["object_id"])


def audit(config):
    root = initialize(config)
    manifest_path = root / "metadata/mesh_inventory.json"
    if manifest_path.exists():
        records = read_json(manifest_path)["records"]
        for record in records:
            path = Path(record["path"])
            stat = path.stat()
            if stat.st_size != record["file_size"] or stat.st_mtime_ns != record["mtime_ns"]:
                raise ValueError("Mesh inventory changed after audit")
    else:
        raw = Path(config["raw_root"])
        pu = Path(config["pu1k_mesh_root"])
        tasks = [("pu1k", "train", p) for p in sorted((pu / "train/train_meshes").glob("*.off"))]
        tasks += [("pu1k", "test", p) for p in sorted((raw / "PU1K/PU1K/test/original_meshes").glob("*.off"))]
        tasks += [("modelnet40", "test", p) for p in sorted((raw / "ModelNet40/ModelNet40").glob("*/test/*.off"))]
        if not tasks or sum(t[0] == "pu1k" and t[1] == "train" for t in tasks) != 900:
            raise ValueError("Expected 900 real PU1K training meshes")
        records = []
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            for i, record in enumerate(executor.map(inspect_mesh, tasks)):
                stat = Path(record["path"]).stat()
                record.update(file_size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                records.append(record)
                if (i + 1) % 100 == 0:
                    log(f"V3 mesh audit: {i + 1}/{len(tasks)}")
        write_json(manifest_path, {"records": records, "fingerprint_precision": 1e-6,
                                   "limitation": "Rotation/remeshing invariant semantic deduplication is not claimed"})
    testing = [r for r in records if r["source_split"] == "test"]
    reserved = {r["geometry_id"] for r in testing}
    excluded = [r for r in records if r["source_split"] == "train" and r["geometry_id"] in reserved]
    eligible = [r for r in records if r["source_split"] == "train" and r["geometry_id"] not in reserved]
    # Aliases of identical geometry contribute a single object, not extra examples.
    representatives, aliases = {}, []
    for record in eligible:
        if record["geometry_id"] in representatives:
            aliases.append({"excluded_alias": record["object_id"], "kept": representatives[record["geometry_id"]]["object_id"]})
        else:
            representatives[record["geometry_id"]] = record
    train, validation = object_split(list(representatives.values()), config["validation_fraction"])
    split = {"train": train, "validation": validation,
             "pu1k_test": [r for r in testing if r["dataset"] == "pu1k"],
             "modelnet40_test": [r for r in testing if r["dataset"] == "modelnet40"],
             "excluded_train_test_geometry_matches": excluded, "excluded_train_aliases": aliases,
             "method": "category_stratified_real_object_and_exact_geometry_group_split",
             "training_geometry_intersects_test": False, "legacy_hdf5_used_for_training": False,
             "raw_1000_mesh_pool_used_as_additional_training": False}
    assert not ({r["geometry_id"] for r in train} & {r["geometry_id"] for r in validation})
    write_json(root / "metadata/dataset_splits.json", split)
    report = {"status": "passed", "training_objects": len(train), "validation_objects": len(validation),
              "pu1k_test_objects": len(split["pu1k_test"]), "modelnet_test_objects": len(split["modelnet40_test"]),
              "excluded_training_test_matches": len(excluded), "excluded_training_aliases": len(aliases),
              "training_source": "900 official split mesh files; legacy addpugan HDF5 is not reused",
              "train_validation_object_overlap": 0, "train_validation_geometry_overlap": 0,
              "train_test_geometry_overlap": 0,
              "identity_limitation": "Known source object IDs and quantized normalized vertex fingerprints; remeshed/rotated semantic aliases may remain"}
    write_json(root / "checks/source_audit.json", report)
    log(f"V3 split: {len(train)} training / {len(validation)} validation objects; {len(excluded)} test matches excluded")
    return split


def write_rows(path, rows, config, schema):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty shard")
    temporary = path.with_suffix(".partial.h5")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(version=VERSION, config_digest=digest_config(config), schema=schema,
                            complete=False, inference_features="input_xyz only", reference_labels_are_not_features=True)
        for key, value in rows[0].items():
            if isinstance(value, dict):
                group = handle.create_group(key)
                for name in value:
                    group.create_dataset(name, data=np.asarray([row[key][name] for row in rows], dtype="f8"))
            elif isinstance(value, str):
                handle.create_dataset(key, data=[row[key] for row in rows], dtype=h5py.string_dtype("utf-8"))
            else:
                array = np.asarray([row[key] for row in rows])
                chunk = (1,) + array.shape[1:] if array.ndim > 1 else (min(len(rows), 256),)
                handle.create_dataset(key, data=array, chunks=chunk, compression="lzf")
        handle.attrs["complete"] = True
    temporary.replace(path)


def read_row(handle, index, keys=None):
    row = {}
    for key in handle.keys() if keys is None else keys:
        item = handle[key]
        if isinstance(item, h5py.Group):
            row[key] = {name: value[index] for name, value in item.items()}
        elif h5py.check_string_dtype(item.dtype) is not None:
            row[key] = item.asstr()[index]
        else:
            row[key] = item[index]
    return row


def shard_valid(path, config, count):
    if not path.exists():
        return False
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema") == "independent_surface_base" and "region_method" not in handle:
            return False
        if not handle.attrs.get("complete", False) or handle.attrs.get("config_digest") != digest_config(config):
            raise ValueError(f"Incompatible V3 shard: {path}")
        if len(handle["base_id"]) != count:
            raise ValueError("Unexpected base count")
    return True


def mesh_bases(task):
    record, group, count, config = task
    root = Path(config["output_root"])
    directories = {"train": "train_bases/pu1k", "validation": "validation_bases/pu1k",
                   "pu1k_patch": "test_bases/pu1k_patch", "modelnet40_patch": "test_bases/modelnet40_patch",
                   "pu1k_object": "test_bases/pu1k_object"}
    destination = root / directories[group] / (Path(record["path"]).stem + ".h5")
    if shard_valid(destination, config, count):
        return {"path": destination.relative_to(root).as_posix(), "rows": count,
                "object_id": record["object_id"], "geometry_id": record["geometry_id"]}
    vertices, triangles, areas, _ = read_off(record["path"])
    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rng = np.random.default_rng(seed(record["object_id"], group, "surface"))
    points, face_ids, barycentric = sample_surface(vertices, triangles, areas, config["surface_samples"], rng)
    tree = cKDTree(points)
    anchors = [int(rng.integers(len(points)))]
    nearest = np.full(len(points), np.inf)
    for i in range(1, count):
        nearest = np.minimum(nearest, np.linalg.norm(points - points[anchors[-1]], axis=1))
        anchors.append(int(np.argmax(nearest)))
    result = []
    for patch, anchor in enumerate(anchors):
        if group == "pu1k_object":
            ids = rng.permutation(len(points))[:16384]
            target_count, input_count = 8192, 2048
        else:
            ids = tree.query(points[anchor], k=config["local_samples"], workers=1)[1]
            ids = ids[rng.permutation(len(ids))]
            target_count, input_count = config["target_points"], config["input_points"]
        target_ids, pool_ids = ids[:target_count], ids[target_count:]
        base_id = f"{record['object_id']}/{group}/patch_{patch:03d}"
        base = prepare_base(points[pool_ids], points[target_ids], normals[face_ids[pool_ids]], normals[face_ids[target_ids]],
                            base_id=base_id, object_id=record["object_id"], category=record["category"],
                            mesh_path=record["path"], mesh_sha256=record["sha256"], geometry_id=record["geometry_id"],
                            gt_face_index=face_ids[target_ids], gt_barycentric=barycentric[target_ids],
                            gt_sample_index=target_ids, pool_sample_index=pool_ids, input_count=input_count)
        result.append(base)
    write_rows(destination, result, config, "independent_surface_base")
    return {"path": destination.relative_to(root).as_posix(), "rows": count,
            "object_id": record["object_id"], "geometry_id": record["geometry_id"]}


def generate_bases(config, split):
    root = initialize(config)
    groups = {
        "train": (split["train"], config["train_patches_per_object"]),
        "validation": (split["validation"], config["validation_patches_per_object"]),
        "pu1k_patch": (split["pu1k_test"], config["test_patches_per_object"]),
        "modelnet40_patch": (split["modelnet40_test"], config["modelnet_test_patches_per_object"]),
    }
    if config["include_whole_object_tests"]:
        groups["pu1k_object"] = (split["pu1k_test"], 1)
    manifest = {"version": VERSION, "config_digest": digest_config(config), "groups": {}}
    for group, (records, count) in groups.items():
        entries = []
        # Bounded batches avoid retaining the outputs of every pending object.
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            for offset in range(0, len(records), config["workers"] * 2):
                tasks = [(record, group, count, config) for record in records[offset:offset + config["workers"] * 2]]
                entries.extend(executor.map(mesh_bases, tasks))
                if len(entries) % 80 == 0 or len(entries) == len(records):
                    log(f"V3 {group}: {len(entries)}/{len(records)} objects, {len(entries) * count:,} clean bases")
        manifest["groups"][group] = entries
        write_json(root / "metadata/base_manifest.json", manifest)
    return manifest


def generate_fixed(config, manifest):
    root = initialize(config)
    for group in ["validation", "pu1k_patch", "modelnet40_patch", "pu1k_object"]:
        if group not in manifest["groups"]:
            continue
        conditions = VALIDATION_CONDITIONS if group == "validation" else list(CONDITIONS)
        entries = manifest["groups"][group]
        destination = root / "processed" / group
        # Each object is an independent, restartable fixed evaluation shard.
        for offset, entry in enumerate(entries):
            path = destination / Path(entry["path"]).name
            expected = entry["rows"] * len(conditions)
            if shard_valid(path, config, expected):
                continue
            with h5py.File(root / entry["path"], "r") as handle:
                bases = [read_row(handle, i) for i in range(entry["rows"])]
            rows = [corrupt_base(base, condition) for base in bases for condition in conditions]
            write_rows(path, rows, config, "fixed_evaluation")
            if (offset + 1) % 40 == 0 or offset + 1 == len(entries):
                log(f"V3 fixed {group}: {offset + 1}/{len(entries)} objects, {len(conditions)} conditions")
    # Keep the original public normal-input benchmark distinct from custom mesh sampling.
    old = Path(config["raw_root"]) / "PU1K/PU1K/test/input_2048"
    official = []
    for path in sorted((old / "input_2048").glob("*.xyz")):
        target = old / "gt_8192" / path.name
        if not target.exists():
            raise ValueError("Official benchmark pairing mismatch")
        official.append({"sample_id": "pu1k_official/" + path.stem, "input_path": str(path), "target_path": str(target)})
    write_json(root / "metadata/official_benchmark.json", {
        "name": "PU1K official normal 2048->8192", "pairs": official,
        "purpose": "Original public sampling protocol; report separately from V3 mesh-resampled conditions"})


def validate(config, manifest=None):
    root = initialize(config)
    manifest = manifest or read_json(root / "metadata/base_manifest.json")
    split = read_json(root / "metadata/dataset_splits.json")
    train_ids = {x["object_id"] for x in split["train"]}
    val_ids = {x["object_id"] for x in split["validation"]}
    test_ids = {x["object_id"] for x in split["pu1k_test"] + split["modelnet40_test"]}
    train_geometry = {x["geometry_id"] for x in split["train"]}
    val_geometry = {x["geometry_id"] for x in split["validation"]}
    test_geometry = {x["geometry_id"] for x in split["pu1k_test"] + split["modelnet40_test"]}
    if train_ids & val_ids or train_ids & test_ids or train_geometry & (val_geometry | test_geometry):
        raise ValueError("Object/geometry split leakage")
    reports, all_gt_hashes = [], {"train": set(), "validation": set(), "test": set()}
    for group, entries in manifest["groups"].items():
        rows_checked = 0
        max_origin_error = 0.0
        for entry in entries:
            with h5py.File(root / entry["path"], "r") as handle:
                if not handle.attrs.get("complete", False):
                    raise ValueError("Incomplete base file")
                gt = handle["gt_xyz"][:]
                pool = handle["pool_xyz"][:]
                if not np.isfinite(gt).all() or not np.isfinite(pool).all():
                    raise ValueError("Non-finite base geometry")
                for i in range(len(gt)):
                    if array_hash(gt[i]) != handle["gt_sha256"].asstr()[i]:
                        raise ValueError("Changed reference geometry")
                    if np.intersect1d(handle["gt_sample_index"][i], handle["pool_sample_index"][i]).size:
                        raise ValueError("Input/target surface sample identities overlap")
                    if np.any(cKDTree(gt[i]).query(pool[i], workers=1)[0] < 1e-7):
                        raise ValueError("Input/target exact coordinate overlap")
                    all_gt_hashes["train" if group == "train" else "validation" if group == "validation" else "test"].add(array_hash(gt[i]))
                # Independently reconstruct one persisted target per object from original triangles.
                vertices, triangles, _, _ = read_off(handle["mesh_path"].asstr()[0])
                face = vertices[triangles[handle["gt_face_index"][0]]]
                restored = (face * handle["gt_barycentric"][0][:, :, None]).sum(axis=1)
                expected = (restored - handle["center"][0]) / handle["scale"][0]
                error = float(np.max(np.abs(expected - gt[0])))
                max_origin_error = max(max_origin_error, error)
                if error > 2e-5:
                    raise ValueError(f"Target surface provenance mismatch: {error}")
                rows_checked += len(gt)
        reports.append({"group": group, "clean_bases_checked": rows_checked, "objects": len(entries),
                        "max_mesh_origin_error_canonical": max_origin_error})
        log(f"V3 validated clean {group}: {rows_checked:,} bases")
    if all_gt_hashes["train"] & (all_gt_hashes["validation"] | all_gt_hashes["test"]):
        raise ValueError("Exact target geometry occurs in training and evaluation")
    fixed_reports = []
    for folder in sorted((root / "processed").iterdir()):
        if not folder.is_dir():
            continue
        conditions_seen = Counter()
        total = 0
        for path in sorted(folder.glob("*.h5")):
            with h5py.File(path, "r") as handle:
                if not handle.attrs.get("complete", False):
                    raise ValueError("Incomplete fixed evaluation file")
                x, gt, clean = handle["input_xyz"][:], handle["gt_xyz"][:], handle["clean_input_xyz"][:]
                if not all(np.isfinite(a).all() for a in [x, gt, clean]):
                    raise ValueError("Non-finite fixed evaluation arrays")
                if not np.array_equal((clean - x).astype("f4"), handle["denoise_offset"][:]):
                    raise ValueError("Denoising correspondence mismatch")
                masks = handle["noise_mask_input"][:].astype(bool)
                if np.any((x != clean).any(-1) != masks):
                    raise ValueError("Noise labels do not describe actual perturbations")
                if np.any(handle["quality/input_gt_exact_overlap"][:] != 0):
                    raise ValueError("Fixed inputs contain exact target samples")
                per_base = defaultdict(list)
                ids = handle["base_id"].asstr()[:]
                conds = handle["condition"].asstr()[:]
                for i, (base_id, condition) in enumerate(zip(ids, conds)):
                    per_base[base_id].append(i)
                    if array_hash(gt[i]) != handle["gt_sha256"].asstr()[i]:
                        raise ValueError("Fixed target hash mismatch")
                for indices in per_base.values():
                    if not np.all(gt[indices] == gt[indices[0]]):
                        raise ValueError("Target differs across conditions")
                    if not np.all(handle["center"][indices] == handle["center"][indices[0]]):
                        raise ValueError("Canonical frame differs across conditions")
                conditions_seen.update(conds)
                total += len(x)
        fixed_reports.append({"group": folder.name, "rows_checked": total, "conditions": dict(conditions_seen)})
        log(f"V3 validated fixed {folder.name}: {total:,} records")
    report = {"status": "passed", "version": VERSION, "base_groups": reports, "fixed_groups": fixed_reports,
              "train_validation_object_overlap": 0, "train_validation_geometry_overlap": 0,
              "train_validation_target_hash_overlap": 0, "train_test_target_hash_overlap": 0,
              "candidate_target_sample_id_overlap": 0, "candidate_target_exact_coordinate_overlap": 0,
              "synthetic_labels_are_inference_features": False,
              "limitations": ["Ellipsoidal spatial occlusion is contiguous in space, but its surface intersection may contain disconnected parts or nearby layers",
                              "Mesh-vertex fingerprints do not prove absence of all rotated/remeshed semantic duplicates",
                              "Reference curvature is a finite-sample PCA diagnostic, not an allocation target",
                              "Reliability targets describe synthetic displacement relative to local spacing",
                              "V3 is a newly sampled mesh protocol, not a reproduction of the legacy addpugan HDF5 protocol"]}
    write_json(root / "checks/full_validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "bases", "fixed", "validate", "all"])
    parser.add_argument("--output", default=DEFAULT["output_root"])
    parser.add_argument("--workers", type=int, default=DEFAULT["workers"])
    args = parser.parse_args()
    config = dict(DEFAULT, output_root=args.output, workers=args.workers)
    root = initialize(config)
    if args.command in {"audit", "all"}:
        split = audit(config)
    else:
        split = read_json(root / "metadata/dataset_splits.json")
    if args.command in {"bases", "all"}:
        manifest = generate_bases(config, split)
    elif args.command != "audit":
        manifest = read_json(root / "metadata/base_manifest.json")
    if args.command in {"fixed", "all"}:
        generate_fixed(config, manifest)
    if args.command in {"validate", "all"}:
        validate(config, manifest)


if __name__ == "__main__":
    main()
