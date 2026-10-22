from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

from . import VERSION, core
from .core import CONDITIONS, REPRESENTATIVES, array_hash, corrupt, input_transform, make_base, read_off, sample_surface, stable_seed

DEFAULT = {
    "version": VERSION, "seed": 20261020,
    "raw_root": "D:/dkwork/baseline_compare/data",
    "output_root": "D:/dkwork/scdata/adaptive_resampling_data",
    "validation_fraction": 0.10, "workers": 8, "batch_size": 256,
    "pu1k_patches_per_test_object": 4, "modelnet_patches_per_test_object": 1,
    "mesh_surface_sample_count": 32768, "mesh_local_sample_count": 4096,
    "prototype_count": 100, "preview_count_per_group": 20,
    "include_whole_object_tests": True,
    "canonical_normalization": "clean_input_centroid_max_radius_v1",
    "model_normalization": "actual_degraded_input_centroid_max_radius_v1",
    "regions": "symmetric_knn12_geodesic_discrete_v1",
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def config_digest(config):
    reproducible = {k: v for k, v in config.items() if k not in {"workers", "batch_size", "preview_count_per_group"}}
    return hashlib.sha256(json.dumps(reproducible, sort_keys=True).encode()).hexdigest()


def sources(config):
    raw = Path(config["raw_root"])
    pu = raw / "PU1K" / "PU1K"
    mn = raw / "ModelNet40" / "ModelNet40"
    return pu, mn, next((pu / "train").glob("*.h5"))


def initialize(config):
    core.GLOBAL_SEED = config["seed"]
    output = Path(config["output_root"])
    raw = Path(config["raw_root"]).resolve()
    if output.resolve() == raw or raw in output.resolve().parents:
        raise ValueError("Processed output must be outside the raw data directory")
    for folder in ["metadata", "checks", "processed", "visualization", "cache/mesh_patches"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    config_file = output / "metadata/generation_config.json"
    if config_file.exists() and config_digest(read_json(config_file)) != config_digest(config):
        raise ValueError("Output belongs to a different protocol. Use a new output directory.")
    write_json(config_file, config)
    write_json(output / "metadata/normalization_config.json", {
        "canonical": config["canonical_normalization"], "canonical_statistics": "unmodified sparse source input; shared by every condition of one base sample",
        "model": config["model_normalization"], "model_statistics": "computed from input_xyz only; GT receives the same transform for training/evaluation",
        "inverse_prediction": "canonical_prediction = model_prediction * inference_scale + inference_center; raw_prediction = canonical_prediction * scale + center",
        "noise_sigma_units": "canonical clean-input unit-radius coordinates, before model normalization",
        "sigma_in_model_coordinates": "noise_sigma / inference_scale",
        "center_scale_are_metadata": True,
        "unknown_training_object_id": "empty UTF-8 string and object_id_known=0; do not treat base_id as object_id",
    })
    return output


def audit(config):
    output = initialize(config)
    pu, mn, h5path = sources(config)
    log("Audit: scanning both complete training arrays (read-only)")
    summaries = defaultdict(list)
    invalid = []
    with h5py.File(h5path, "r") as f:
        keys = list(f.keys())
        n = len(f["poisson_256"])
        for start in range(0, n, 512):
            inp = f["poisson_256"][start:start + 512].astype("f8")
            gt = f["poisson_1024"][start:start + 512].astype("f8")
            center = inp.mean(axis=1)
            scale = np.linalg.norm(inp - center[:, None], axis=2).max(axis=1)
            gt_scale = np.linalg.norm(gt - gt.mean(axis=1)[:, None], axis=2).max(axis=1)
            valid = np.isfinite(inp).all(axis=(1, 2)) & np.isfinite(gt).all(axis=(1, 2)) & (scale > 1e-12)
            invalid.extend((start + np.flatnonzero(~valid)).tolist())
            summaries["input_radius"].extend(scale.tolist())
            summaries["gt_radius"].extend(gt_scale.tolist())
            summaries["gt_input_radius_ratio"].extend((gt_scale / np.maximum(scale, 1e-15)).tolist())
            summaries["centroid_distance_over_input_radius"].extend((np.linalg.norm(gt.mean(1) - center, axis=1) / np.maximum(scale, 1e-15)).tolist())
        attrs = dict(f.attrs)
    if invalid:
        raise ValueError(f"Invalid source rows: {invalid[:20]}")
    split_rng = np.random.default_rng(config["seed"])
    order = split_rng.permutation(n)
    nval = round(n * config["validation_fraction"])
    val = sorted(order[:nval].tolist())
    train = sorted(order[nval:].tolist())
    prototype = split_rng.choice(train, size=config["prototype_count"], replace=False).tolist()
    pu_objects = sorted(p.stem for p in (pu / "test/original_meshes").glob("*.off"))
    mn_test = sorted(p.relative_to(mn).as_posix() for p in mn.glob("*/test/*.off"))
    counts = Counter(p.parent.name for p in mn.glob("*/*/*.off"))
    splits = {"method": "fixed_seed_patch_random_validation_object_overlap_possible",
              "object_identity_recovered": False, "train_indices": train, "val_indices": val,
              "prototype_indices": prototype, "pu1k_test_objects": pu_objects,
              "modelnet40_test_meshes": mn_test,
              "limitations": "Original PU1K object IDs unavailable. Disjoint source rows do not establish disjoint objects. Official PU1K test remains untouched by validation."}
    write_json(output / "metadata/dataset_splits.json", splits)
    recovery = {"result": "not_recoverable_from_available_coordinates_and_metadata",
                "local_h5_keys": keys, "local_h5_attributes": attrs,
                "training_meshes_available": len(list((pu / "train").rglob("*.off"))),
                "sources_consulted": ["https://github.com/guochengqian/PU-GCN", "https://raw.githubusercontent.com/guochengqian/PU-GCN/master/docs/issue_fixes.md"],
                "finding": "Published loader keys and the addpugan explanation do not establish row-to-object mapping for this local legacy HDF5.",
                "not_assumed": "50 consecutive rows per object; HDF5 filename is not an identity manifest"}
    write_json(output / "checks/object_identity_audit.json", recovery)
    log("Audit: checking 80 representative ModelNet mesh files, including both OFF headers")
    mesh_reports = []
    for category in sorted(p for p in mn.iterdir() if p.is_dir()):
        for split in ["train", "test"]:
            path = next(iter(sorted((category / split).glob("*.off"))))
            _, _, _, info = read_off(path)
            mesh_reports.append({"path": path.relative_to(mn).as_posix(), **info})
    pairs = []
    for npoints in [256, 512, 1024, 2048]:
        folder = pu / "test" / f"input_{npoints}"
        names_in = {p.stem for p in (folder / f"input_{npoints}").glob("*.xyz")}
        names_gt = {p.stem for p in (folder / f"gt_{npoints * 4}").glob("*.xyz")}
        if names_in != names_gt or names_gt != set(pu_objects):
            raise ValueError("Official PU1K test pairing mismatch")
        pairs.append({"input_points": npoints, "gt_points": npoints * 4, "pairs": len(names_in)})
    report = {"status": "passed", "training_rows": n, "training_keys": keys, "source_h5_sha256": file_hash(h5path),
              "invalid_rows": invalid, "source_h5": str(h5path), "object_identity": recovery,
              "coordinate_statistics": {k: dict(zip(["min", "p01", "median", "p99", "max"], np.percentile(v, [0, 1, 50, 99, 100]).tolist())) for k, v in summaries.items()},
              "official_test_pairs": pairs, "modelnet_counts": dict(counts), "modelnet_categories": 40,
              "mesh_header_trials": mesh_reports, "python": sys.version, "platform": platform.platform(),
              "numpy": np.__version__, "h5py": h5py.__version__, "config_digest": config_digest(config)}
    write_json(output / "checks/source_audit.json", report)
    write_json(output / "metadata/raw_sources.json", {"raw_root": str(Path(config["raw_root"])), "h5_path": str(h5path), "h5_sha256": report["source_h5_sha256"], "raw_data_copied": False})
    render_alignment(config, prototype[:20])
    log(f"Audit passed: {n:,} valid pairs; {len(train):,} train / {len(val):,} patch-level validation")
    return report


class H5Writer:
    def __init__(self, path, length, config, protocol):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.length = length
        self.digest = config_digest(config)
        self.skip = False
        self.file = None
        self.offset = 0
        self.protocol = protocol
        if self.path.exists():
            with h5py.File(self.path, "r") as f:
                if f.attrs.get("config_digest") != self.digest or not f.attrs.get("complete", False) or len(f["input_xyz"]) != length:
                    raise ValueError(f"Existing dataset has incompatible metadata: {self.path}")
            self.skip = True
        self.partial = self.path.with_suffix(".partial.h5")

    def append(self, rows):
        if self.skip or not rows:
            return
        if self.file is None:
            self.file = h5py.File(self.partial, "w")
            self.file.attrs.update({"version": VERSION, "config_digest": self.digest, "protocol": self.protocol,
                                   "normalization": "canonical_clean_input_then_runtime_input_only", "complete": False,
                                   "object_id_empty_means_unknown": True, "masks_not_network_inputs": True})
            example = rows[0]
            chunk = min(self.length, 128)
            for key, value in example.items():
                if key == "quality":
                    group = self.file.create_group("quality")
                    for name in value:
                        group.create_dataset(name, (self.length,), dtype="f8", chunks=(chunk,), compression="lzf")
                elif isinstance(value, str):
                    self.file.create_dataset(key, (self.length,), dtype=h5py.string_dtype("utf-8"), chunks=(chunk,), compression="lzf")
                else:
                    a = np.asarray(value)
                    self.file.create_dataset(key, (self.length,) + a.shape, dtype=a.dtype,
                                             chunks=(chunk,) + a.shape, compression="lzf")
            # Both names are available, with a hard link rather than duplicated labels.
            self.file["noise_mask"] = self.file["noise_mask_input"]
        stop = self.offset + len(rows)
        if stop > self.length:
            raise ValueError("More rows than declared")
        for key in rows[0]:
            if key == "quality":
                for name in rows[0][key]:
                    self.file["quality"][name][self.offset:stop] = [row[key][name] for row in rows]
            elif isinstance(rows[0][key], str):
                self.file[key][self.offset:stop] = [row[key] for row in rows]
            else:
                self.file[key][self.offset:stop] = np.asarray([row[key] for row in rows])
        self.offset = stop

    def close(self):
        if self.skip:
            return
        if self.offset != self.length:
            if self.file:
                self.file.close()
            raise ValueError(f"Incomplete dataset: {self.path}: {self.offset}/{self.length}")
        self.file.attrs["complete"] = True
        self.file.flush()
        self.file.close()
        self.partial.replace(self.path)


def training_task(task):
    index, inp, gt, conditions = task
    base = make_base(inp, gt, base_id=f"pu1k_train/row_{index:05d}", source_index=index)
    return [corrupt(base, condition) for condition in conditions]


def train_condition(index, category):
    choices = {
        0: ["A0_normal"], 1: ["B1_sparse_light", "B2_sparse_mid", "B3_sparse_heavy"],
        2: ["C1_hole_05", "C2_hole_10", "C3_hole_20"],
        3: ["D1_noise_002", "D2_noise_005", "D3_noise_010"], 4: ["E1_mixed"],
    }[int(category)]
    return choices[stable_seed(index, "train_strength") % len(choices)]


def write_training_set(config, name, indices, *, mode):
    output = Path(config["output_root"])
    _, _, h5path = sources(config)
    count = len(indices) * (5 if mode == "val" else 1)
    writer = H5Writer(output / "processed" / name, count, config, f"PU1K patch {mode}; object IDs unknown")
    if writer.skip:
        log(f"Already complete: {name}")
        return
    categories = np.arange(len(indices)) % 5
    np.random.default_rng(stable_seed(mode, "balanced_categories")).shuffle(categories)
    tick = time.monotonic()
    with h5py.File(h5path, "r") as source, ThreadPoolExecutor(config["workers"]) as executor:
        for start in range(0, len(indices), config["batch_size"]):
            batch_ids = indices[start:start + config["batch_size"]]
            # h5py fancy indexing requires sorted indices. Retrieve once in sorted order.
            read_order = np.argsort(batch_ids)
            sorted_ids = np.asarray(batch_ids)[read_order]
            in_sorted = source["poisson_256"][sorted_ids]
            gt_sorted = source["poisson_1024"][sorted_ids]
            restore = np.argsort(read_order)
            inputs, targets = in_sorted[restore], gt_sorted[restore]
            tasks = []
            for j, index in enumerate(batch_ids):
                if mode == "val":
                    conditions = REPRESENTATIVES
                elif mode == "prototype":
                    conditions = [REPRESENTATIVES[int(categories[start + j])]]
                else:
                    conditions = [train_condition(index, categories[start + j])]
                tasks.append((index, inputs[j], targets[j], conditions))
            grouped = list(executor.map(training_task, tasks))
            writer.append([row for group in grouped for row in group])
            if time.monotonic() - tick > 12 or start + len(batch_ids) == len(indices):
                log(f"{name}: {writer.offset:,}/{count:,} rows")
                tick = time.monotonic()
    writer.close()


def prototype(config):
    output = initialize(config)
    if not (output / "checks/source_audit.json").exists():
        audit(config)
    splits = read_json(output / "metadata/dataset_splits.json")
    write_training_set(config, "prototype_100.h5", splits["prototype_indices"], mode="prototype")
    report = validate_file(output / "processed/prototype_100.h5", config)
    # Actual bit-for-bit replay, and invariants across strengths on the same bases.
    _, _, h5path = sources(config)
    replayed = 0
    with h5py.File(h5path, "r") as raw, h5py.File(output / "processed/prototype_100.h5", "r") as generated:
        for row in range(len(generated["input_xyz"])):
            index = int(generated["source_index"][row])
            condition = generated["condition"].asstr()[row]
            base = make_base(raw["poisson_256"][index], raw["poisson_1024"][index], base_id=f"pu1k_train/row_{index:05d}", source_index=index)
            repeated = corrupt(base, condition)
            for key in ["input_xyz", "gt_xyz", "hole_mask_gt", "noise_mask_input", "sparse_mask_gt"]:
                if not np.array_equal(repeated[key], generated[key][row]):
                    raise ValueError(f"Reproducibility failure: {row}:{key}")
            if row < 8:
                small, medium, large = [corrupt(base, c) for c in ["C1_hole_05", "C2_hole_10", "C3_hole_20"]]
                if np.any(small["hole_mask_gt"] > medium["hole_mask_gt"]) or np.any(medium["hole_mask_gt"] > large["hole_mask_gt"]):
                    raise ValueError("Hole masks are not nested by severity")
                density = [corrupt(base, c)["quality"]["sparse_density_ratio_actual"] for c in ["B1_sparse_light", "B2_sparse_mid", "B3_sparse_heavy"]]
                if not density[0] > density[1] > density[2]:
                    raise ValueError("Density severities are not ordered")
            replayed += 1
    report["bitwise_replayed_samples"] = replayed
    write_json(output / "checks/prototype_acceptance.json", report)
    render_file(output / "processed/prototype_100.h5", output / "visualization/prototype", 20, by_condition=True)
    log("Prototype passed: all 100 rows replayed exactly; density ordering, connected holes, labels and shapes checked")


def mesh_task(args):
    mesh_path, dataset, config = args
    mesh_path = Path(mesh_path)
    output = Path(config["output_root"])
    object_id = mesh_path.stem if dataset == "pu1k" else mesh_path.parent.parent.name + "/" + mesh_path.stem
    prefix = f"{dataset}_test/{object_id}"
    cache_name = dataset + "_" + hashlib.sha256(prefix.encode()).hexdigest()[:20] + ".npz"
    cache_path = output / "cache/mesh_patches" / cache_name
    metadata_path = cache_path.with_suffix(".json")
    stat = mesh_path.stat()
    expected = config_digest(config)
    if cache_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("config_digest") == expected and metadata.get("source_size") == stat.st_size and metadata.get("source_mtime_ns") == stat.st_mtime_ns:
            with np.load(cache_path) as cached:
                arrays = {key: cached[key] for key in cached.files}
        else:
            raise ValueError("Stale mesh patch cache")
    else:
        vertices, triangles, areas, mesh_info = read_off(mesh_path)
        mesh_sha = file_hash(mesh_path)
        rng = np.random.default_rng(stable_seed(prefix, "surface"))
        dense, face_ids, barycentric = sample_surface(vertices, triangles, areas, config["mesh_surface_sample_count"], rng)
        tree = cKDTree(dense)
        patches = config["pu1k_patches_per_test_object"] if dataset == "pu1k" else config["modelnet_patches_per_test_object"]
        anchors = [int(rng.integers(len(dense)))]
        # Clean-surface farthest anchors before any corruption; no FPS after corruption.
        minimum = np.full(len(dense), np.inf)
        for _ in range(1, patches):
            minimum = np.minimum(minimum, np.linalg.norm(dense - dense[anchors[-1]], axis=1))
            anchors.append(int(np.argmax(minimum)))
        arrays = {}
        for patch_index, anchor in enumerate(anchors):
            _, ids = tree.query(dense[anchor], k=config["mesh_local_sample_count"], workers=1)
            ids = rng.permutation(ids)
            gt_ids, clean_ids, pool_ids = ids[:1024], ids[1024:1280], ids[1280:]
            arrays[f"gt_{patch_index}"] = dense[gt_ids]
            arrays[f"clean_{patch_index}"] = dense[clean_ids]
            arrays[f"pool_{patch_index}"] = dense[pool_ids]
            arrays[f"gt_face_index_{patch_index}"] = face_ids[gt_ids]
            arrays[f"gt_barycentric_{patch_index}"] = barycentric[gt_ids]
            arrays[f"anchor_{patch_index}"] = dense[anchor]
        np.savez_compressed(cache_path, **arrays)
        metadata = {"mesh_path": str(mesh_path), "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                    "mesh_sha256": mesh_sha, "config_digest": expected, "patch_count": patches, "object_id": object_id,
                    "sampling": "area-weighted triangles; sqrt-barycentric; independent disjoint GT/input candidate subsets",
                    "patch_extent": f"Euclidean {config['mesh_local_sample_count']}-NN of {config['mesh_surface_sample_count']} clean surface samples",
                    "local_surface_limitation": "Spatial patches can include nearby layers; hole connectivity is on a sample kNN graph, not certified mesh geodesics.",
                    **mesh_info}
        write_json(metadata_path, metadata)
    conditions = list(CONDITIONS) if dataset == "pu1k" else ["C2_hole_10", "E1_mixed"]
    results = []
    for patch_index in range(metadata["patch_count"]):
        base = make_base(arrays[f"clean_{patch_index}"], arrays[f"gt_{patch_index}"], arrays[f"pool_{patch_index}"],
                         base_id=f"{prefix}/patch_{patch_index:02d}", object_id=f"{dataset}/{object_id}", object_known=True,
                         metadata={"mesh_path": str(mesh_path), "cache": cache_path.name, "patch_index": patch_index,
                                   "mesh_sha256": metadata["mesh_sha256"], "area_measure": "uniform_surface_sample_fraction_MC_not_exact_mesh_area"})
        results.append({c: corrupt(base, c) for c in conditions})
    return results


def write_mesh_tests(config, dataset):
    output = Path(config["output_root"])
    pu, mn, _ = sources(config)
    paths = sorted((pu / "test/original_meshes").glob("*.off")) if dataset == "pu1k" else sorted(mn.glob("*/test/*.off"))
    conditions = list(CONDITIONS) if dataset == "pu1k" else ["C2_hole_10", "E1_mixed"]
    patches = config["pu1k_patches_per_test_object"] if dataset == "pu1k" else config["modelnet_patches_per_test_object"]
    length = len(paths) * patches
    folder = output / "processed" / ("pu1k_test_patch" if dataset == "pu1k" else "modelnet40_test_patch")
    writers = {c: H5Writer(folder / (c + ".h5"), length, config, f"{dataset} independent test objects; local patch") for c in conditions}
    if all(w.skip for w in writers.values()):
        log(f"Already complete: {dataset} mesh tests")
        return
    tick = time.monotonic()
    # Limit concurrent large-mesh readers; up to a million triangles per original mesh.
    with ThreadPoolExecutor(min(config["workers"], 4)) as executor:
        for start in range(0, len(paths), 16):
            selected = paths[start:start + 16]
            groups = list(executor.map(mesh_task, [(p, dataset, config) for p in selected]))
            for c, writer in writers.items():
                writer.append([patch[c] for obj in groups for patch in obj])
            if time.monotonic() - tick > 12 or start + len(selected) == len(paths):
                log(f"{dataset} mesh tests: {start + len(selected):,}/{len(paths):,} objects, {len(conditions)} conditions")
                tick = time.monotonic()
    for writer in writers.values():
        writer.close()


def object_task(args):
    name, pu = args
    folder = Path(pu) / "test/input_2048"
    clean = np.loadtxt(folder / "input_2048" / (name + ".xyz"))
    gt = np.loadtxt(folder / "gt_8192" / (name + ".xyz"))
    if clean.shape != (2048, 3) or gt.shape != (8192, 3):
        raise ValueError("Unexpected official whole-object shape")
    base = make_base(clean, gt, base_id=f"pu1k_test_object/{name}", object_id=f"pu1k/{name}", object_known=True,
                     metadata={"area_measure": "unique_discrete_reference_fraction_GT_fraction_reported_separately", "source": "official_2048_8192_pair"},
                     deduplicate_reference=True)
    return {condition: corrupt(base, condition) for condition in CONDITIONS}


def write_object_tests(config):
    output = Path(config["output_root"])
    pu, _, _ = sources(config)
    names = sorted(p.stem for p in (pu / "test/input_2048/input_2048").glob("*.xyz"))
    writers = {c: H5Writer(output / "processed/pu1k_test_object" / (c + ".h5"), len(names), config,
                           "PU1K whole object; official 2048->8192 base; separate from patch results") for c in CONDITIONS}
    if all(w.skip for w in writers.values()):
        return
    with ThreadPoolExecutor(min(config["workers"], 4)) as executor:
        for start in range(0, len(names), 8):
            groups = list(executor.map(object_task, [(name, pu) for name in names[start:start + 8]]))
            for c, writer in writers.items():
                writer.append([group[c] for group in groups])
            log(f"PU1K whole-object tests: {min(start + 8, len(names))}/{len(names)} objects")
    for writer in writers.values():
        writer.close()


def build(config):
    output = initialize(config)
    acceptance = output / "checks/prototype_acceptance.json"
    if not acceptance.exists() or read_json(acceptance).get("status") != "passed":
        raise ValueError("Run and accept the prototype before full generation")
    visual = output / "checks/prototype_visual_review.json"
    if not visual.exists() or read_json(visual).get("status") != "passed" or read_json(visual).get("config_digest") != config_digest(config):
        raise ValueError("Prototype visual review must be recorded before full generation")
    _, _, source_h5 = sources(config)
    if file_hash(source_h5) != read_json(output / "checks/source_audit.json")["source_h5_sha256"]:
        raise ValueError("Raw training HDF5 changed after audit")
    splits = read_json(output / "metadata/dataset_splits.json")
    write_training_set(config, "pu1k_train_mixed.h5", splits["train_indices"], mode="train")
    write_training_set(config, "pu1k_val_fixed.h5", splits["val_indices"], mode="val")
    write_mesh_tests(config, "pu1k")
    write_mesh_tests(config, "modelnet40")
    if config["include_whole_object_tests"]:
        write_object_tests(config)
    validate_all(config)
    preview_all(config)
    summarize(config)


def validate_file(path, config):
    errors = []
    stats = defaultdict(list)
    condition_counts = Counter()
    unknown_objects = 0
    with h5py.File(path, "r") as f:
        count = len(f["input_xyz"])
        if not f.attrs.get("complete", False):
            errors.append("incomplete file")
        if f["input_xyz"].shape[1:] not in [(256, 3), (2048, 3)]:
            errors.append("unexpected input shape")
        if f["gt_xyz"].shape[1] != f["input_xyz"].shape[1] * 4:
            errors.append("upsampling ratio is not four")
        ids = f["sample_id"].asstr()[:]
        if len(set(ids)) != count:
            errors.append("duplicate sample IDs")
        unknown_objects = int((f["object_id_known"][:] == 0).sum())
        conditions = f["condition"].asstr()[:]
        condition_counts.update(conditions)
        for start in range(0, count, 256):
            stop = min(count, start + 256)
            x, gt, clean = f["input_xyz"][start:stop], f["gt_xyz"][start:stop], f["clean_input_xyz"][start:stop]
            mask = f["noise_mask_input"][start:stop].astype(bool)
            if not (np.isfinite(x).all() and np.isfinite(gt).all() and np.isfinite(clean).all()):
                errors.append(f"nonfinite at {start}")
            delta = x - clean
            if np.any(delta[~mask] != 0):
                errors.append(f"unlabelled noise at {start}")
            recorded_fraction = f["quality/noise_point_fraction"][start:stop]
            if not np.allclose(mask.mean(axis=1), recorded_fraction, atol=1e-10):
                errors.append(f"noise mask proportion mismatch at {start}")
            actual_std = np.array([d[m].std() if m.any() else 0.0 for d, m in zip(delta, mask)])
            if not np.allclose(actual_std, f["quality/noise_sigma_empirical"][start:stop], atol=1e-9):
                errors.append(f"noise statistics differ from actual displacement at {start}")
            if not np.allclose(f["hole_mask_gt"][start:stop].mean(axis=1), f["quality/hole_fraction_gt"][start:stop], atol=1e-10):
                errors.append(f"hole mask proportion mismatch at {start}")
            kinds = f["corruption_type"][start:stop]
            if np.any(f["hole_mask_gt"][start:stop][~np.isin(kinds, [2, 4])]):
                errors.append(f"unexpected hole labels at {start}")
            if np.any(f["sparse_mask_gt"][start:stop][~np.isin(kinds, [1, 4])]):
                errors.append(f"unexpected sparse labels at {start}")
            calculated_center = x.astype("f8").mean(axis=1)
            calculated_scale = np.linalg.norm(x.astype("f8") - calculated_center[:, None], axis=2).max(axis=1)
            if not np.allclose(calculated_center, f["inference_center"][start:stop], atol=1e-6) or not np.allclose(calculated_scale, f["inference_scale"][start:stop], rtol=1e-6):
                errors.append(f"input-only inference normalization mismatch at {start}")
            for key in f["quality"]:
                values = f["quality"][key][start:stop]
                stats[key].extend(values.tolist())
            # Recompute hashes of every GT, not just the stored producer assertion.
            hashes = f["gt_sha256"].asstr()[start:stop]
            if any(array_hash(a) != b for a, b in zip(gt, hashes)):
                errors.append(f"GT integrity mismatch at {start}")
        if max(stats["hole_intrusions"], default=0) > 0:
            errors.append("hole refill detected")
        for components, fraction in zip(stats["hole_components"], stats["hole_fraction_reference"]):
            if fraction > 0 and components != 1:
                errors.append("disconnected recorded hole")
        actual, target = np.array(stats["sparse_density_ratio_actual"]), np.array(stats["sparse_density_ratio_target"])
        if np.max(np.abs(actual - target)) > 0.06:
            errors.append("nonuniform density ratio outside tolerance")
        aggregate_noise = {}
        sigma = np.asarray(stats["noise_sigma"])
        empirical = np.asarray(stats["noise_sigma_empirical"])
        for s in sorted(set(sigma) - {0.0}):
            values = empirical[sigma == s] / s
            aggregate_noise[str(s)] = {"rows": len(values), "mean_empirical_over_target": float(values.mean()), "p05_p50_p95": np.percentile(values, [5, 50, 95]).tolist()}
            # Small prototypes have only 20 realizations; aggregate tolerance reflects sampling variance.
            if not 0.90 <= values.mean() <= 1.10:
                errors.append("aggregate noise standard deviation mismatch")
    report = {"path": str(path), "status": "passed" if not errors else "failed", "rows": count,
              "condition_counts": dict(condition_counts), "unknown_object_id_rows": unknown_objects,
              "quality_statistics": {k: {"min": min(v), "mean": float(np.mean(v)), "max": max(v)} for k, v in stats.items()},
              "noise_statistics": aggregate_noise, "errors": sorted(set(errors))}
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    return report


def validate_all(config):
    output = Path(config["output_root"])
    log("Validation: scanning all output rows, metadata, masks and GT hashes")
    reports = []
    files = sorted((output / "processed").rglob("*.h5"))
    manifest_path = output / "metadata/manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["file", "row", "sample_id", "base_id", "object_id", "object_id_known", "condition", "source_index", "seed", "gt_sha256"])
        for path in files:
            if ".partial." in path.name:
                raise ValueError(f"Unfinished dataset: {path}")
            report = validate_file(path, config)
            reports.append(report)
            with h5py.File(path, "r") as f:
                texts = {k: f[k].asstr()[:] for k in ["sample_id", "base_id", "object_id", "condition", "gt_sha256"]}
                known, source_ids, seeds = f["object_id_known"][:], f["source_index"][:], f["seed"][:]
                for i in range(report["rows"]):
                    csv_writer.writerow([path.relative_to(output).as_posix(), i, texts["sample_id"][i], texts["base_id"][i], texts["object_id"][i], int(known[i]), texts["condition"][i], int(source_ids[i]), int(seeds[i]), texts["gt_sha256"][i]])
            log(f"Validated {path.relative_to(output)}: {report['rows']:,} rows")
    with h5py.File(output / "processed/pu1k_train_mixed.h5", "r") as train, h5py.File(output / "processed/pu1k_val_fixed.h5", "r") as val:
        overlap = set(train["source_index"][:]) & set(val["source_index"][:])
        if overlap:
            raise ValueError("Training/validation source-row leakage")
    paired = []
    for folder in ["pu1k_test_patch", "modelnet40_test_patch", "pu1k_test_object"]:
        paths = sorted((output / "processed" / folder).glob("*.h5"))
        if not paths:
            continue
        with h5py.File(paths[0], "r") as first:
            ids, hashes = first["base_id"].asstr()[:], first["gt_sha256"].asstr()[:]
            centers, scales = first["center"][:], first["scale"][:]
        for path in paths[1:]:
            with h5py.File(path, "r") as f:
                if not (np.array_equal(ids, f["base_id"].asstr()[:]) and np.array_equal(hashes, f["gt_sha256"].asstr()[:]) and np.array_equal(centers, f["center"][:]) and np.array_equal(scales, f["scale"][:])):
                    raise ValueError("GT or canonical coordinate frame differs across conditions")
        paired.append({"group": folder, "base_samples": len(ids), "conditions": len(paths), "identical_gt_and_canonical_transform": True})
    write_json(output / "checks/full_validation.json", {"status": "passed", "files": reports, "paired_conditions": paired,
                                                         "train_val_source_row_overlap": 0,
                                                         "original_object_overlap_in_train_val": "unknown; cannot be guaranteed without source object IDs"})
    _, _, source_h5 = sources(config)
    if file_hash(source_h5) != read_json(output / "checks/source_audit.json")["source_h5_sha256"]:
        raise ValueError("Raw HDF5 no longer matches audited SHA256")
    write_json(output / "checks/code_fingerprints.json", {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    return reports


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def render_alignment(config, indices):
    plt = _matplotlib()
    output = Path(config["output_root"]) / "visualization/source_alignment"
    output.mkdir(parents=True, exist_ok=True)
    _, _, path = sources(config)
    metrics = []
    with h5py.File(path, "r") as source:
        for sheet in range(0, len(indices), 5):
            fig = plt.figure(figsize=(15, 15))
            for row, index in enumerate(indices[sheet:sheet + 5]):
                raw_in, raw_gt = source["poisson_256"][index], source["poisson_1024"][index]
                inp, gt, center, scale = core.normalize_pair(raw_in, raw_gt)
                distance = cKDTree(gt).query(inp)[0]
                metrics.append({"source_index": index, "nearest_input_to_gt_median_normalized": float(np.median(distance)), "center": center.tolist(), "scale": scale})
                for column, (a, b, title) in enumerate([(raw_in, raw_gt, "Source pair"), (inp, gt, "Same transform: 3D"), (inp, gt, "Same transform: XY")]):
                    ax = fig.add_subplot(5, 3, row * 3 + column + 1, projection="3d" if column < 2 else None)
                    if column < 2:
                        ax.scatter(*b.T, s=1, c="#aab7c4", alpha=0.5)
                        ax.scatter(*a.T, s=4, c="#0077bb")
                        _equal_3d(ax, b)
                    else:
                        ax.scatter(b[:, 0], b[:, 1], s=2, c="#aab7c4", alpha=0.5)
                        ax.scatter(a[:, 0], a[:, 1], s=5, c="#0077bb")
                        ax.set_aspect("equal", adjustable="datalim")
                    ax.set_title(f"row {index}: {title}", fontsize=9)
            fig.tight_layout()
            fig.savefig(output / f"alignment_{sheet // 5 + 1:02d}.png", dpi=120)
            plt.close(fig)
    write_json(Path(config["output_root"]) / "checks/alignment_samples.json", metrics)


def _equal_3d(ax, points):
    low, high = points.min(0), points.max(0)
    center = (low + high) / 2
    radius = max(float((high - low).max()) / 2, 1e-3) * 1.06
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def render_file(path, destination, count=20, by_condition=False):
    plt = _matplotlib()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        groups = {}
        if by_condition:
            values = f["condition"].asstr()[:]
            groups = {c: np.flatnonzero(values == c)[:count] for c in sorted(set(values))}
        else:
            groups[Path(path).stem] = np.linspace(0, len(f["input_xyz"]) - 1, min(count, len(f["input_xyz"])), dtype=int)
        for condition, ids in groups.items():
            for sheet in range(0, len(ids), 5):
                target = destination / f"{condition}_{sheet // 5 + 1:02d}.png"
                if target.exists():
                    continue
                selected = ids[sheet:sheet + 5]
                fig = plt.figure(figsize=(15, 3.0 * len(selected)))
                for r, index in enumerate(selected):
                    inp, gt, clean = f["input_xyz"][index], f["gt_xyz"][index], f["clean_input_xyz"][index]
                    hole = f["hole_mask_gt"][index].astype(bool)
                    sparse = f["sparse_mask_gt"][index].astype(bool)
                    noise = f["noise_mask_input"][index].astype(bool)
                    # Thin the 8192-point object references for legible pictures only.
                    show = np.arange(len(gt))[::max(1, len(gt) // 2048)]
                    colors = np.full(len(gt), "#aab7c4", dtype=object)
                    colors[sparse] = "#e0ad35"
                    colors[hole] = "#df4655"
                    for col in range(3):
                        ax = fig.add_subplot(len(selected), 3, r * 3 + col + 1, projection="3d" if col == 0 else None)
                        if col == 0:
                            ax.scatter(*gt[show].T, c=colors[show].tolist(), s=3, alpha=0.5)
                            ax.scatter(*inp[~noise].T, c="#0077bb", s=6)
                            ax.scatter(*inp[noise].T, c="#8b3ac6", s=7)
                            _equal_3d(ax, gt)
                        else:
                            a, b = (0, 1) if col == 1 else (0, 2)
                            ax.scatter(gt[show, a], gt[show, b], c=colors[show].tolist(), s=3, alpha=0.5)
                            ax.scatter(inp[~noise, a], inp[~noise, b], c="#0077bb", s=7)
                            ax.scatter(inp[noise, a], inp[noise, b], c="#8b3ac6", s=8)
                            if noise.any():
                                ax.quiver(clean[noise, a], clean[noise, b], inp[noise, a] - clean[noise, a], inp[noise, b] - clean[noise, b], angles="xy", scale_units="xy", scale=1, width=0.002, color="#8b3ac6", alpha=0.6)
                            ax.set_aspect("equal", adjustable="datalim")
                        ax.set_title(f"{condition} | row {index} | {['3D overlay', 'XY', 'XZ'][col]}", fontsize=8)
                fig.suptitle("Blue=input; gray=clean GT; red=hole GT; gold=sparse GT; purple=noisy input", fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, 0.98))
                fig.savefig(target, dpi=115)
                plt.close(fig)


def preview_all(config):
    output = Path(config["output_root"])
    log("Rendering fixed multi-view inspection samples for every condition")
    for folder in ["pu1k_test_patch", "modelnet40_test_patch", "pu1k_test_object"]:
        for path in sorted((output / "processed" / folder).glob("*.h5")):
            render_file(path, output / "visualization" / folder, config["preview_count_per_group"])
            log(f"Previewed {folder}/{path.stem}")
    for name in ["pu1k_train_mixed.h5", "pu1k_val_fixed.h5"]:
        render_file(output / "processed" / name, output / "visualization" / Path(name).stem, 20, by_condition=True)


def summarize(config):
    output = Path(config["output_root"])
    validation = read_json(output / "checks/full_validation.json")
    rows = []
    for report in validation["files"]:
        path = Path(report["path"])
        rows.append({"file": path.relative_to(output).as_posix(), "rows": report["rows"], "bytes": path.stat().st_size, "status": report["status"]})
    summary = {"status": "complete", "version": VERSION, "config_digest": config_digest(config), "datasets": rows,
               "total_rows_including_prototype": sum(r["rows"] for r in rows),
               "h5_total_GiB": sum(r["bytes"] for r in rows) / 1024**3,
               "preview_png_count": len(list((output / "visualization").rglob("*.png"))),
               "limitations": ["Training object IDs unknown; validation is patch-level with possible same-object overlap.",
                               "Training and official-object hole ratios are discrete reference-point fractions, not physical surface areas.",
                               "Mesh test hole ratios are Monte Carlo surface fractions; graph connectivity is not exact mesh-geodesic connectivity.",
                               "Mesh local patches may include spatially nearby surface layers.",
                               "ModelNet40 test only; no network training performed."]}
    write_json(output / "checks/completion_summary.json", summary)
    text = ["# 自适应点云重采样数据：已生成", "", f"版本：{VERSION}。所有正式文件通过完整性验收。", "",
            "原始数据仅引用，不复制、不修改。生成配置、划分及样本清单见 metadata/。", "",
            "| 文件 | 样本数 |", "|---|---:|"]
    text.extend(f"| {r['file']} | {r['rows']:,} |" for r in rows)
    text.extend(["", "## 使用", "", "使用代码目录 adaptive_resampling/dataset.py 的 load_sample 或 H5PatchDataset。默认只根据实际退化输入计算模型归一化参数；预测需要执行 inverse_prediction 后再评价。", "",
                 "区域标签和 center/scale 是监督或坐标恢复元信息，不作为普通推理的网络输入。", "",
                 "## 已知限制", "", *["- " + s for s in summary["limitations"]], "",
                 "## 验收证据", "", "- checks/source_audit.json：源数据检查和版本。", "- checks/prototype_acceptance.json：100例原型逐例重放。", "- checks/full_validation.json：全部输出逐行检查。", "- checks/completion_summary.json：数据量及限制。", "- visualization/：每条件20例，含三维、XY和XZ视图。", ""])
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")
    log(f"Complete: {len(rows)} HDF5 files, {summary['h5_total_GiB']:.2f} GiB, {summary['preview_png_count']} preview sheets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "prototype", "build", "validate", "preview", "summary"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = DEFAULT.copy()
    if args.config:
        config.update(read_json(args.config))
    if args.output:
        config["output_root"] = str(args.output)
    if args.workers:
        config["workers"] = args.workers
    initialize(config)
    functions = {"audit": audit, "prototype": prototype, "build": build, "validate": validate_all, "preview": preview_all, "summary": summarize}
    functions[args.stage](config)


if __name__ == "__main__":
    main()
