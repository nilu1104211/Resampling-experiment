#!/usr/bin/env python3
"""Bounded, reproducible V3 subset evaluation. Copy this file to the repo root.

Default: 80 PU1K whole objects + 80 ModelNet40 whole objects, 7 conditions,
2 models = 2240 predictions, all 2048 -> 8192 points. ModelNet40 uses a frozen
test-only supplement; original V3 data is unchanged. PUFM++ shares ONE existing
schedule or one separately labelled 256-training-patch calibration.
Only the standard library is needed for --phase plan.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

VERSION = "subset-v3-whole-object80"
DEFAULT_SPLITS = ["pu1k_object", "modelnet40_object"]
OBJECT_SPLITS = {"pu1k_object", "modelnet40_object"}
CONDITIONS = ["A0_normal", "B2_sparse_mid", "C2_hole_10", "D2_noise_005",
              "D7_outliers_02", "E1_mixed", "E2_mixed_outliers"]
CHECKPOINTS = {
    "pufm": "server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt",
    "pufmpp": "server_runs/patch32/pufmpp_s2_b8_w0_fused/last.pt",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(seed, *parts):
    return signature([seed, *parts])


def choose_subset(manifest, split, objects, patches, seed):
    if objects < 1 or (patches is not None and patches < 1):
        raise ValueError("Object and patch counts must be positive")
    entries = manifest["groups"][split]
    if objects > len(entries):
        raise ValueError(f"Requested {objects} objects, but {split} has {len(entries)}")
    offsets, total = [], 0
    for entry in entries:
        offsets.append(total)
        total += int(entry["rows"])
    # PU1K patch and whole-object tests must use the very same objects.
    object_group = {"pu1k_object": "pu1k_patch", "modelnet40_object": "modelnet40_patch"}.get(split, split)
    order = sorted(range(len(entries)), key=lambda i: rank(seed, "object", object_group, entries[i]["object_id"]))
    selected = []
    for number in order[:objects]:
        entry = entries[number]
        if patches is not None and patches > int(entry["rows"]):
            raise ValueError(f"{entry['object_id']} in {split} has only {entry['rows']} samples; cannot select {patches}")
        rows = sorted(range(int(entry["rows"])), key=lambda i: rank(seed, "patch", split, entry["object_id"], i))
        rows = sorted(rows if patches is None else rows[:patches])
        if not rows:
            raise ValueError(f"Object has no samples: {entry['object_id']}")
        selected.append({"object_id": entry["object_id"], "entry_index": number,
                         "patch_indices": rows, "dataset_indices": [offsets[number] + i for i in rows]})
    selected.sort(key=lambda item: item["object_id"])
    return {"policy": "sha256-ranked-object-and-patch-v2", "seed": seed, "object_sampling_group": object_group,
            "split": split, "available_objects": len(entries), "available_samples_per_condition": total,
            "requested_patches_per_object": "all_available" if patches is None else patches, "objects": selected,
            "samples_per_condition": sum(len(item["patch_indices"]) for item in selected)}


def flatten(selection):
    return [(index, item["object_id"]) for item in selection["objects"] for index in item["dataset_indices"]]


def make_plan(args):
    repo = args.repo.resolve()
    root = (args.root or Path(os.environ.get("V3_DATA_ROOT", repo / "adaptive_resampling_data_v3"))).resolve()
    manifest_path = root / "metadata/base_manifest.json"
    manifest = read_json(manifest_path)
    if args.split == "modelnet40_object":
        # Preserve the original test-object identities and their stable indices.
        # Actual whole-cloud coordinates come ONLY from the separate supplement.
        manifest = {**manifest, "groups": {**manifest["groups"], "modelnet40_object": [
            {**entry, "rows": 1} for entry in manifest["groups"]["modelnet40_patch"]]}}
    selection = choose_subset(manifest, args.split, args.objects, args.patches_per_object, args.seed)
    if len(args.conditions) != len(set(args.conditions)) or len(args.models) != len(set(args.models)):
        raise ValueError("Duplicate conditions or models are not allowed")
    # v3_core contains only protocol definitions; parse literals without importing dependencies.
    import ast
    tree = ast.parse((repo / "adaptive_resampling/v3_core.py").read_text(encoding="utf-8"))
    available = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"CONDITIONS", "VALIDATION_CONDITIONS"}:
                    available[target.id] = ([ast.literal_eval(key) for key in node.value.keys]
                                            if target.id == "CONDITIONS" else ast.literal_eval(node.value))
    allowed = available["VALIDATION_CONDITIONS"] if args.split == "validation" else available["CONDITIONS"]
    if not args.conditions or any(condition not in allowed for condition in args.conditions):
        raise ValueError(f"Conditions must belong to {args.split}: {list(allowed)}")
    training_count = sum(min(32, int(entry["rows"])) for entry in manifest["groups"]["train"])
    if args.calibration_samples < 8 or args.calibration_samples > training_count:
        raise ValueError(f"--calibration-samples must be between 8 and {training_count}")
    calibration_indices = sorted(sorted(range(training_count), key=lambda i: rank(args.seed, "calibration", i))[:args.calibration_samples])
    comparison = repo / "compare-change"
    ours = repo / "our/PUFM-main"
    if not comparison.is_dir():
        comparison = repo / "network_adaptation/compare-change"
    if not ours.is_dir():
        ours = repo / "adaptive_network/PUFM-main"
    per_model = selection["samples_per_condition"] * len(args.conditions)
    plan = {"runner_version": VERSION, "evaluation_scope": "fixed_object_subset",
            "inference_runtime": {"fps_backend": getattr(args, "fps_backend", "auto"),
                                  "patch_batch_size": getattr(args, "patch_batch_size", 1)},
            "repo": str(repo), "data_root": str(root), "output": str(args.output.resolve()),
            "comparison_root": str(comparison), "pufm_root": str(ours),
            "seed": args.seed, "split": args.split, "conditions": args.conditions, "models": args.models,
            "selection": selection, "selection_sha256": signature(selection),
            "predictions_per_model": per_model, "total_predictions": per_model * len(args.models),
            "p2f": args.p2f, "device": args.device,
            "checkpoints": {model: str((args.pufm_checkpoint if model == "pufm" else args.pufmpp_checkpoint).resolve()) for model in args.models},
            "calibration": {"scope": "external_schedule" if args.schedule else "fixed_training_subset",
                            "external_schedule": str(args.schedule.resolve()) if args.schedule else None,
                            "samples": args.calibration_samples, "training_samples_available": training_count,
                            "dataset_indices": calibration_indices, "grid_positions": 51, "batch_size": 8,
                            "assignment_workers": 8, "workers": 0, "flow_steps": 6, "manifold_step": 0.01,
                            "matching": "once_per_selected_sample_then_cached"},
            "metadata_sha256": {name: file_hash(root / "metadata" / name) for name in
                                ("base_manifest.json", "generation_config.json", "dataset_splits.json")}}
    if args.split == "modelnet40_object":
        supplement = (getattr(args, "whole_data_root", None) or root / "supplements/modelnet40_object80_seed21").resolve()
        path = supplement / "manifest.json"
        plan["whole_data_root"] = str(supplement)
        plan["supplement_manifest_sha256"] = file_hash(path) if path.is_file() else None
    return plan


def supplement_path(root, relative):
    path = (Path(root) / relative).resolve()
    path.relative_to(Path(root).resolve())
    return path


def validate_supplement(plan):
    if plan["split"] != "modelnet40_object":
        return
    import ast
    root = Path(plan["whole_data_root"])
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"ModelNet40 whole-object supplement is missing: {path}. Extract modelnet40_object80_seed21.zip into the V3 data root first.")
    package = read_json(path)
    if file_hash(path) != plan["supplement_manifest_sha256"]:
        raise ValueError("Supplement manifest changed after the evaluation plan was created")
    if package.get("schema") != "frozen-v3-whole-objects-v1" or package.get("input_points") != 2048 or package.get("target_points") != 8192:
        raise ValueError("Supplement is not a 2048 -> 8192 whole-object test package")
    if package.get("source_manifest_sha256") != plan["metadata_sha256"]["base_manifest.json"] or package.get("source_splits_sha256") != plan["metadata_sha256"]["dataset_splits.json"]:
        raise ValueError("Supplement belongs to different original V3 object splits")
    source = signature({name: (Path(plan["repo"]) / "adaptive_resampling" / name).read_text(encoding="utf-8").strip()
                        for name in ("core.py", "v3_core.py")})
    if package.get("protocol_source_sha256") != source:
        raise ValueError("Supplement's V3 sampling/corruption protocol differs from the repository")
    if any(condition not in package["conditions"] for condition in plan["conditions"]):
        raise ValueError("Requested conditions are not frozen in the whole-object supplement")
    records = {item["object_id"]: item for item in package["records"]}
    if len(records) != len(package["records"]):
        raise ValueError("Supplement has duplicate object identities")
    hashes = {}
    for selected in plan["selection"]["objects"]:
        if selected["object_id"] not in records:
            raise ValueError(f"Whole-object supplement does not contain selected object {selected['object_id']}; keep the packaged seed/budget or generate a matching supplement")
        record = records[selected["object_id"]]
        source = supplement_path(root, record["file"])
        digest = file_hash(source)
        if digest != record["sha256"]:
            raise ValueError(f"Whole-object data failed SHA256 verification: {source}")
        hashes[record["file"]] = digest
    plan["supplement_files_sha256"] = hashes


def evaluation_dataset(plan, condition, *, include_supervision=False, include_reference_metadata=False):
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    if plan["split"] != "modelnet40_object":
        return AdaptiveResamplingDataset(plan["data_root"], plan["split"], condition=condition,
                                         include_supervision=include_supervision,
                                         include_reference_metadata=include_reference_metadata)
    import numpy as np
    from adaptive_resampling.v3_paths import SourcePaths
    package_root = Path(plan["whole_data_root"])
    package = read_json(package_root / "manifest.json")
    records = {item["object_id"]: item for item in package["records"]}
    entries = read_json(Path(plan["data_root"]) / "metadata/base_manifest.json")["groups"]["modelnet40_patch"]
    condition_index = package["conditions"].index(condition)

    class FrozenWholeDataset(AdaptiveResamplingDataset):
        # Reuse the repository's exact input-only normalization, inverse-transform
        # metadata, label transforms, and reference handling through __getitem__.
        def __init__(self):
            self.root = Path(plan["data_root"])
            self.split, self.training, self.epoch = "modelnet40_object", False, 0
            self.normalize, self.rotate, self.features = True, False, False
            self.include_supervision = include_supervision
            self.reference_metadata = include_reference_metadata
            self.source_paths = SourcePaths(self.root) if include_reference_metadata else None
        def __len__(self):
            return len(entries)
        def source_index(self, index):
            return int(index)
        def raw_sample(self, index, epoch=None):
            record = records[entries[int(index)]["object_id"]]
            with np.load(supplement_path(package_root, record["file"]), allow_pickle=False) as archive:
                shared = set(archive["__shared_fields__"].tolist())
                if archive["__conditions__"].tolist() != package["conditions"]:
                    raise ValueError("Whole-object shard has different condition ordering")
                row = {}
                for name in archive.files:
                    if name.startswith("__"):
                        continue
                    value = archive[name] if name in shared else archive[name][condition_index]
                    row[name] = str(value.item()) if value.ndim == 0 and value.dtype.kind in "US" else value.copy()
            if isinstance(row.get("quality"), str):
                row["quality"] = json.loads(row["quality"])
            if row["object_id"] != record["object_id"] or row["condition"] != condition:
                raise ValueError("Whole-object shard identity mismatch")
            return row
        def close(self):
            pass
    return FrozenWholeDataset()


def make_suite(args):
    splits = DEFAULT_SPLITS if args.split == "all" else [args.split]
    output = args.output.resolve()
    plans = []
    for split in splits:
        child = SimpleNamespace(**vars(args))
        child.split = split
        child.output = output / split
        # An object-level record is one full cloud, not a local patch budget.
        child.patches_per_object = 1 if split in OBJECT_SPLITS else args.patches_per_object
        plan = make_plan(child)
        plan["calibration_output"] = str(output / "shared_calibration")
        plans.append(plan)
    by_split = {plan["split"]: plan for plan in plans}
    if {"pu1k_patch", "pu1k_object"} <= by_split.keys():
        patch_ids = [item["object_id"] for item in by_split["pu1k_patch"]["selection"]["objects"]]
        whole_ids = [item["object_id"] for item in by_split["pu1k_object"]["selection"]["objects"]]
        if patch_ids != whole_ids:
            raise ValueError("PU1K patch and whole-object selections do not match")
    return {"runner_version": VERSION, "evaluation_scope": "fixed_object_subset",
            "output": str(output), "splits": splits, "plans": plans,
            "predictions_per_model": sum(plan["predictions_per_model"] for plan in plans),
            "total_predictions": sum(plan["total_predictions"] for plan in plans),
            "calibration_runs": int("pufmpp" in args.models and args.schedule is None),
            "modelnet_whole_object_available": any(plan["split"] == "modelnet40_object" for plan in plans) and
                                               all(plan.get("supplement_manifest_sha256") is not None
                                                   for plan in plans if plan["split"] == "modelnet40_object")}


def print_suite(suite):
    for plan in suite["plans"]:
        rows_per_object = sorted({len(item["patch_indices"]) for item in plan["selection"]["objects"]})
        print(f"{plan['split']}: {len(plan['selection']['objects'])} objects x {rows_per_object} samples/object x "
              f"{len(plan['conditions'])} conditions x {len(plan['models'])} models = {plan['total_predictions']} predictions; "
              f"points: {'2048 -> 8192' if plan['split'] in OBJECT_SPLITS else '256 -> 1024'}", flush=True)
    print(f"TOTAL SUBSET PREDICTIONS: {suite['total_predictions']}; PUFM++ calibration runs: {suite['calibration_runs']}", flush=True)
    if any(plan["split"] == "modelnet40_object" for plan in suite["plans"]):
        print(f"ModelNet40 frozen whole-object supplement present: {suite['modelnet_whole_object_available']}", flush=True)


def check_checkpoint(saved, model):
    if saved.get("protocol") != "V3" or saved.get("epoch") != 29:
        raise ValueError(f"{model} must be a completed V3 30-epoch stage endpoint")
    if model == "pufmpp":
        if saved.get("args", {}).get("stage") != "s2" or saved.get("result", {}).get("partial_debug_run", True):
            raise ValueError("PUFM++ requires a non-debug S2 checkpoint")
    else:
        config = saved.get("configuration", {})
        if saved.get("stage") != "baseline" or saved.get("partial_debug_run", True):
            raise ValueError("PUFM requires a non-debug baseline checkpoint")
        if config.get("use_rgr", True) or config.get("condition", True) or config.get("allocation") != "midpoint":
            raise ValueError("Selected PUFM checkpoint is not the plain baseline")


def configure_paths(plan):
    os.environ["V3_CODE_ROOT"] = plan["repo"]
    os.environ["V3_DATA_ROOT"] = plan["data_root"]
    os.environ["PYTHONFAULTHANDLER"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.path.insert(0, plan["repo"])


def preflight(plan):
    configure_paths(plan)
    validate_supplement(plan)
    import torch
    import numpy
    import scipy
    import h5py
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    from adaptive_resampling.v3_training import require_training_profile
    train = AdaptiveResamplingDataset(plan["data_root"], "train", include_supervision=False, train_patches_per_object=32)
    profile = train.training_profile
    train.close()
    if profile["samples"] != plan["calibration"]["training_samples_available"]:
        raise ValueError("Training dataset size differs from the sampling plan")
    plan["training_profile"] = profile
    plan["checkpoint_sha256"] = {}
    for model, name in plan["checkpoints"].items():
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {model} checkpoint: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        check_checkpoint(saved, model)
        require_training_profile(saved, profile, context="Subset evaluation")
        del saved
        gc.collect()
        plan["checkpoint_sha256"][model] = file_hash(path)
        print(f"Checked {model}: completed, non-debug 30-epoch stage", flush=True)
    plan["environment"] = {"python": sys.version, "torch": str(torch.__version__),
                           "torch_cuda": torch.version.cuda, "numpy": numpy.__version__,
                           "scipy": scipy.__version__, "h5py": h5py.__version__}
    sources = [Path(__file__), Path(plan["comparison_root"]) / "v3_common.py",
               Path(plan["comparison_root"]) / "Enhanced_PUFM-main/v3_adapter.py",
               Path(plan["pufm_root"]) / "adaptive_v3.py", Path(plan["pufm_root"]) / "gain_teacher.py"]
    for folder in (Path(plan["repo"]) / "adaptive_resampling", Path(plan["pufm_root"]) / "models",
                   Path(plan["comparison_root"]) / "Enhanced_PUFM-main/models"):
        sources.extend(folder.rglob("*.py"))
    plan["source_sha256"] = signature({str(path): file_hash(path) for path in sorted(set(sources))})
    plan["raw_root"] = os.environ.get("V3_RAW_ROOT") if plan["p2f"] else None
    if plan["calibration"]["external_schedule"] and "pufmpp" in plan["models"]:
        path = Path(plan["calibration"]["external_schedule"])
        schedule = torch.load(path, map_location="cpu", weights_only=False)
        verify_schedule(schedule, plan, external_path=path)
        plan["calibration"]["external_sha256"] = file_hash(path)
    # Inspect only the selected evaluation records; original meshes are optional.
    ids, mesh_hashes = {}, {}
    expected = flatten(plan["selection"])
    for condition in plan["conditions"]:
        ds = evaluation_dataset(plan, condition, include_reference_metadata=plan["p2f"])
        try:
            ids[condition] = []
            for index, object_id in expected:
                sample = ds[index]
                if sample["object_id"] != object_id or sample["condition"] != condition:
                    raise ValueError("Sample does not match the selected object/condition")
                ids[condition].append(str(sample["sample_id"]))
                if plan["p2f"]:
                    reference = sample["reference"]
                    path = Path(reference["mesh_path"])
                    if str(path) not in mesh_hashes:
                        digest = file_hash(path)
                        if digest != reference["mesh_sha256"]:
                            raise ValueError(f"Mesh does not match the V3 reference: {path}")
                        mesh_hashes[str(path)] = digest
        finally:
            ds.close()
    plan["selected_sample_ids"] = ids
    plan["mesh_sha256"] = mesh_hashes
    return plan


def verify_schedule(saved, plan, external_path=None):
    import numpy as np
    expected = plan["checkpoint_sha256"]["pufmpp"]
    if saved.get("calibration_scope") == "fixed_training_subset":
        if saved.get("calibration_context") != signature({"checkpoint": expected, "calibration": plan["calibration"],
                                                          "profile": plan["training_profile"], "seed": plan["seed"],
                                                          "source": plan["source_sha256"]}):
            raise ValueError("Subset schedule belongs to different weights/data/calibration settings")
    elif external_path is not None:
        if saved.get("partial_debug_run", True):
            raise ValueError("An external debug schedule is not accepted")
        manifest = external_path.parent / "evaluation_manifest.json"
        if not manifest.is_file():
            raise ValueError("External schedule needs its evaluation_manifest.json to verify its checkpoint")
        owner = read_json(manifest)
        if owner.get("checkpoints", {}).get("pufmpp", {}).get("sha256") != expected or owner.get("training_profile") != plan["training_profile"]:
            raise ValueError("External schedule is not bound to the selected checkpoint/training data")
    else:
        raise ValueError("Schedule has no supported provenance")
    if "grid" not in saved or "cdf" not in saved:
        raise ValueError("Schedule must contain its grid and CDF")
    grid, cdf = np.asarray(saved["grid"]), np.asarray(saved["cdf"])
    if grid.ndim != 1 or grid.shape != cdf.shape or len(grid) < 2 or not np.isfinite(grid).all() or not np.isfinite(cdf).all():
        raise ValueError("Invalid schedule arrays")
    if abs(grid[0]) > 1e-6 or abs(grid[-1] - 1) > 1e-6 or abs(cdf[0]) > 1e-6 or abs(cdf[-1] - 1) > 1e-6 or (np.diff(grid) <= 0).any() or (np.diff(cdf) < 0).any():
        raise ValueError("Invalid schedule range or ordering")
    times = np.interp(np.linspace(0, 1, 7), cdf, grid)
    if not (np.diff(times) > 0).all():
        raise ValueError("The six-step schedule must increase strictly")
    return times.tolist()


def receipt_complete(marker, key, outputs):
    if not marker.exists():
        return False
    previous = read_json(marker)
    if previous.get("signature") != key:
        raise ValueError(f"Cached task has different inputs: {marker}. Choose a new --output")
    for path in outputs:
        if not path.is_file():
            return False
        if previous.get("outputs", {}).get(str(path)) != file_hash(path):
            raise ValueError(f"Cached output was modified: {path}. Choose a new --output")
    return True


def finish_receipt(marker, key, outputs):
    write_json(marker, {"signature": key, "outputs": {str(path): file_hash(path) for path in outputs}})


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_network(plan, model):
    import torch
    checkpoint = plan["checkpoints"][model]
    if model == "pufm":
        folder = Path(plan["pufm_root"])
        sys.path.insert(0, str(folder))
        entry = import_file("subset_pufm_entry", folder / "adaptive_v3.py")
        values = {key: None for key in ("preset", "regions", "capacity", "minimum", "flow_steps", "swap_step",
                                       "swap_iterations", "allocation", "backbone", "resume")}
        values.update({key: False for key in ("no_virtual", "no_reliability", "no_recovery", "no_uncertainty", "allow_debug")})
        args = SimpleNamespace(**values, checkpoint=checkpoint, command="predict", stage="baseline", device=plan["device"])
        entry.seed_all(plan["seed"])
        network, saved = entry.create_model(args)
        del saved
        gc.collect()
        network.eval()
        def predict(points, count):
            return network.integrate(network.prepare(points, count))
        return network, predict, None
    folder = Path(plan["comparison_root"])
    sys.path.insert(0, str(folder / "Enhanced_PUFM-main"))
    sys.path.insert(0, str(folder))
    common = import_file("v3_common", folder / "v3_common.py")
    # Construction for explicit subset inference. Its times are always replaced
    # with the verified schedule before predict is called; no uniform ablation.
    args = SimpleNamespace(command="schedule", schedule=None, uniform_schedule=False,
                            flow_steps=6, manifold_step=0.01, device=plan["device"], model_name="pufm++", stage="s2",
                            patch_batch_size=plan.get("inference_runtime", {}).get("patch_batch_size", 1))
    common.configure_assignment_workers(8)
    common.set_seed(plan["seed"])
    network = common.load_adapter(folder / "Enhanced_PUFM-main", args)
    saved = common.load_weights(network, checkpoint, args)
    del saved
    gc.collect()
    network.eval()
    return network, network.predict, common


def calibration_curve(adapter, batches, grid, device):
    import torch
    errors = []
    started = time.perf_counter()
    with torch.inference_mode():
        for number, position in enumerate(grid):
            total, count = 0.0, 0
            for initial, target in batches:
                x0, x1 = initial.to(device), target.to(device)
                t = torch.full((len(x0),), float(position), device=device)
                query = (1 - float(position)) * x0 + float(position) * x1
                _, latent = adapter.velocity(query, t)
                velocity, _ = adapter.velocity(query, t, latent)
                total += float((velocity.transpose(1, 2) - (x1 - x0)).square().mean().item()) * len(x0)
                count += len(x0)
            errors.append(total / count)
            print(json.dumps({"calibration_position": number + 1, "positions": len(grid),
                              "schedule_t": float(position), "mse": errors[-1],
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    return errors


def ensure_schedule(plan, adapter, common):
    import numpy as np
    import torch
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    external = plan["calibration"]["external_schedule"]
    if external:
        saved = torch.load(external, map_location="cpu", weights_only=False)
        return verify_schedule(saved, plan, Path(external)), file_hash(external)
    root = Path(plan.get("calibration_output", plan["output"]))
    root.mkdir(parents=True, exist_ok=True)
    path, marker = root / "subset_schedule.pt", root / "status/calibration.done.json"
    context = signature({"checkpoint": plan["checkpoint_sha256"]["pufmpp"], "calibration": plan["calibration"],
                         "profile": plan["training_profile"], "seed": plan["seed"], "source": plan["source_sha256"]})
    if receipt_complete(marker, context, [path]):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        print("Reusing completed subset calibration", flush=True)
        return verify_schedule(saved, plan), file_hash(path)
    ds = AdaptiveResamplingDataset(plan["data_root"], "train", training_mode="mixed", include_supervision=False,
                                  train_patches_per_object=32)
    indices = plan["calibration"]["dataset_indices"]
    batches, sample_ids = [], []
    generator = torch.Generator(device=plan["device"]).manual_seed(plan["seed"])
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(indices), 8):
                samples = [ds[(0, i)] for i in indices[start:start + 8]]
                x = torch.from_numpy(np.stack([s["input_xyz"] for s in samples])).to(plan["device"])
                gt = torch.from_numpy(np.stack([s["gt_xyz"] for s in samples])).to(plan["device"])
                initial = common.midpoint(x) + 0.02 * torch.randn(gt.shape, generator=generator, device=plan["device"])
                target = common.assignment_target(initial, gt)
                batches.append((initial.cpu(), target.cpu()))
                sample_ids.extend(str(s["sample_id"]) for s in samples)
                print(f"Cached calibration matching {len(sample_ids)}/{len(indices)}", flush=True)
    finally:
        ds.close()
    grid = np.linspace(0, 1, 51)
    errors = calibration_curve(adapter, batches, grid, plan["device"])
    density = np.maximum(np.asarray(errors), 1e-8)
    cdf = np.r_[0, np.cumsum((density[:-1] + density[1:]) * 0.5 * np.diff(grid))]
    cdf /= cdf[-1]
    saved = {"protocol": "V3", "split": "train", "partial_debug_run": True,
             "calibration_scope": "fixed_training_subset", "calibration_context": context,
             "sample_ids": sample_ids, "dataset_indices": indices, "grid": grid, "cdf": cdf, "loss": errors,
             "t_sched": torch.tensor(np.interp(np.linspace(0, 1, 7), cdf, grid), dtype=torch.float32),
             "seconds": time.perf_counter() - started}
    temporary = path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)
    finish_receipt(marker, context, [path])
    return verify_schedule(saved, plan), file_hash(path)


def aggregate(rows, context):
    import numpy as np
    grouped = {}
    columns = sorted(set().union(*(row.keys() for row in rows)) - {"sample_id", "object_id"})
    for row in rows:
        grouped.setdefault(row["object_id"], []).append(row)
    objects = []
    for object_id, members in sorted(grouped.items()):
        result = {"object_id": object_id, "patches": len(members)}
        for column in columns:
            values = [r[column] for r in members if column in r]
            if values:
                result[column] = float(np.mean(values))
        objects.append(result)
    rng, metrics = np.random.default_rng(20261021), {}
    for column in columns:
        values = np.asarray([item[column] for item in objects if column in item])
        bootstrap = values[rng.integers(len(values), size=(1000, len(values)))].mean(1)
        metrics[column] = {"mean_over_objects": float(values.mean()), "valid_objects": len(values),
                           "std_over_objects": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                           "bootstrap_95CI": np.percentile(bootstrap, [2.5, 97.5]).tolist()}
    return objects, {**context, "partial_debug_evaluation": True, "evaluation_scope": "fixed_object_subset",
                     "samples": len(rows), "objects": len(objects), "prediction_coordinates": "model",
                     "aggregation": "mean patches within each selected object, then mean selected objects",
                     "metrics": metrics}


def write_csv(path, rows, leading):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = leading + sorted(set().union(*(row.keys() for row in rows)) - set(leading))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_predictions(plan, model, condition, prediction, directory, schedule_hash):
    import numpy as np
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    from adaptive_resampling.v3_metrics import evaluate_sample
    with np.load(prediction, allow_pickle=False) as archive:
        ids, values = archive["sample_id"].astype(str), archive["prediction_xyz"]
    expected = plan["selected_sample_ids"][condition]
    count = 8192 if plan["split"] in OBJECT_SPLITS else 1024
    if ids.tolist() != expected or len(set(ids)) != len(ids) or values.shape != (len(expected), count, 3) or not np.isfinite(values).all():
        raise ValueError("Predictions do not match the fixed selection or expected point budget")
    rows = []
    ds = evaluation_dataset(plan, condition, include_supervision=True, include_reference_metadata=plan["p2f"])
    started = time.perf_counter()
    try:
        for position, (index, object_id) in enumerate(flatten(plan["selection"])):
            sample = ds[index]
            if str(sample["sample_id"]) != ids[position] or sample["object_id"] != object_id:
                raise ValueError("Metric sample ID/object pairing mismatch")
            metrics = evaluate_sample(values[position], sample, coordinates="model", include_p2f=plan["p2f"])
            rows.append({"sample_id": ids[position], "object_id": object_id, **metrics})
            if (position + 1) % 8 == 0 or position + 1 == len(ids):
                print(f"{model} {condition}: evaluated {position + 1}/{len(ids)}", flush=True)
    finally:
        ds.close()
    objects, summary = aggregate(rows, {"model": model, "split": plan["split"], "condition": condition,
                                        "selection_sha256": plan["selection_sha256"], "schedule_sha256": schedule_hash,
                                        "calibration_scope": plan["calibration"]["scope"] if model == "pufmpp" else "not_applicable",
                                        "metric_wall_seconds": time.perf_counter() - started})
    write_csv(directory / "per_sample.csv", rows, ["sample_id", "object_id"])
    write_csv(directory / "per_object.csv", objects, ["object_id", "patches"])
    write_json(directory / "summary.json", summary)


def worker(plan, model):
    configure_paths(plan)
    os.environ["V3_FPS_BACKEND"] = plan.get("inference_runtime", {}).get("fps_backend", "auto")
    import numpy as np
    import torch
    from adaptive_resampling.v3_pointops import fps_backend_status
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    network, predict, common = load_network(plan, model)
    schedule_hash = None
    if model == "pufmpp":
        network.times, schedule_hash = ensure_schedule(plan, network, common)
        print(f"PUFM++ six-step times: {network.times}", flush=True)
    root = Path(plan["output"])
    context_key = signature(plan)
    count = 8192 if plan["split"] in OBJECT_SPLITS else 1024
    for condition in plan["conditions"]:
        path = root / "predictions" / f"{model}_{condition}.npz"
        info = path.with_suffix(".json")
        marker = root / "status" / f"{model}_{condition}_predict.done.json"
        key = signature([context_key, model, condition, "predict", schedule_hash])
        if not receipt_complete(marker, key, [path, info]):
            ds = evaluation_dataset(plan, condition)
            values, seconds, ids = [], [], []
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    for position, (index, object_id) in enumerate(flatten(plan["selection"])):
                        sample = ds[index]
                        expected = plan["selected_sample_ids"][condition][position]
                        if str(sample["sample_id"]) != expected or sample["object_id"] != object_id:
                            raise ValueError("Prediction sample differs from the fixed selection")
                        seed = int(rank(plan["seed"], "inference", expected)[:8], 16)
                        torch.manual_seed(seed)
                        x = torch.from_numpy(sample["input_xyz"])[None].to(plan["device"])
                        if plan["device"].startswith("cuda"):
                            torch.cuda.synchronize()
                        tick = time.perf_counter()
                        y = predict(x, count)
                        if y.shape != (1, count, 3) or not torch.isfinite(y).all():
                            raise ValueError("Invalid prediction shape or nonfinite values")
                        if plan["device"].startswith("cuda"):
                            torch.cuda.synchronize()
                        seconds.append(time.perf_counter() - tick)
                        if position == 0:
                            print(f"{model} FPS runtime: {fps_backend_status()}", flush=True)
                        values.append(y[0].cpu().numpy())
                        ids.append(expected)
                        if (position + 1) % 8 == 0 or position + 1 == len(flatten(plan["selection"])):
                            print(f"{model} {condition}: predicted {position + 1}/{len(flatten(plan['selection']))}; elapsed {time.perf_counter() - started:.1f}s", flush=True)
            finally:
                ds.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, sample_id=np.asarray(ids, dtype=str), prediction_xyz=np.asarray(values, dtype="f4"))
            temporary.replace(path)
            write_json(info, {"protocol": "V3", "partial_debug_run": True, "evaluation_scope": "fixed_object_subset",
                              "model": model, "split": plan["split"], "condition": condition, "samples": len(ids),
                              "selection_sha256": plan["selection_sha256"], "schedule_sha256": schedule_hash,
                              "calibration_scope": plan["calibration"]["scope"] if model == "pufmpp" else "not_applicable",
                              "coordinates": "model", "inference_seconds": seconds,
                               "fps_runtime": fps_backend_status(),
                              "prediction_wall_seconds": time.perf_counter() - started})
            finish_receipt(marker, key, [path, info])
        else:
            print(f"Reusing {model} {condition} predictions", flush=True)
        directory = root / ("metrics_p2f" if plan["p2f"] else "metrics") / model / condition
        outputs = [directory / name for name in ("summary.json", "per_sample.csv", "per_object.csv")]
        marker = root / "status" / f"{model}_{condition}_metrics.done.json"
        key = signature([context_key, model, condition, "metrics", schedule_hash, file_hash(path)])
        if not receipt_complete(marker, key, outputs):
            evaluate_predictions(plan, model, condition, path, directory, schedule_hash)
            finish_receipt(marker, key, outputs)
        else:
            print(f"Reusing {model} {condition} metrics", flush=True)


def run_child(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                stamped = time.strftime("[%Y-%m-%d %H:%M:%S] ") + line
                print(stamped, end="", flush=True)
                handle.write(stamped)
                handle.flush()
            if process.wait():
                raise RuntimeError(f"Evaluation failed; see {log}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise


def summarize(plan):
    root = Path(plan["output"])
    rows = []
    for model in plan["models"]:
        for condition in plan["conditions"]:
            path = root / ("metrics_p2f" if plan["p2f"] else "metrics") / model / condition / "summary.json"
            item = read_json(path)
            if item.get("selection_sha256") != plan["selection_sha256"] or item.get("samples") != plan["selection"]["samples_per_condition"]:
                raise ValueError("Summary does not match the planned subset")
            rows.append({"model": model, "split": plan["split"], "condition": condition,
                         "evaluation_scope": "fixed_object_subset", "objects": item["objects"], "samples": item["samples"],
                         "calibration_scope": item["calibration_scope"],
                         **{name: metric["mean_over_objects"] for name, metric in item["metrics"].items()}})
    output = root / "subset_summary.csv"
    write_csv(output, rows, ["model", "split", "condition", "evaluation_scope", "objects", "samples", "calibration_scope"])
    print(f"Completed SUBSET evaluation. Results: {output}", flush=True)


def summarize_suite(suite):
    rows = []
    for plan in suite["plans"]:
        summarize(plan)
        with (Path(plan["output"]) / "subset_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    output = Path(suite["output"]) / "subset_summary.csv"
    write_csv(output, rows, ["model", "split", "condition", "evaluation_scope", "objects", "samples", "calibration_scope"])
    print(f"Completed all requested SUBSET splits. Combined results: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("server_runs/whole_object_eval80_20261030"))
    parser.add_argument("--phase", choices=["plan", "check", "run"], default="run")
    parser.add_argument("--split", choices=["all", "validation", "pu1k_patch", "modelnet40_patch", *DEFAULT_SPLITS], default="all",
                        help="all: ONLY whole objects from PU1K and ModelNet40; no separate patch/validation tasks")
    parser.add_argument("--objects", type=int, default=80)
    parser.add_argument("--whole-data-root", type=Path, help="ModelNet40 frozen whole-object supplement directory")
    parser.add_argument("--patches-per-object", type=int,
                        help="Legacy patch-only tasks; whole-object tasks always have one record per object")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--models", nargs="+", choices=["pufm", "pufmpp"], default=["pufm", "pufmpp"])
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--schedule", type=Path, help="Existing full schedule with its original evaluation_manifest.json")
    parser.add_argument("--p2f", action="store_true", help="Also compute point-to-mesh distances on all SELECTED objects")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps-backend", choices=["auto", "cuda", "torch"], default="auto")
    parser.add_argument("--patch-batch-size", type=int, default=1,
                        help="PUFM++ patch batching; keep 1 for validated serial numerical equivalence")
    parser.add_argument("--pufm-checkpoint", type=Path, default=Path(CHECKPOINTS["pufm"]))
    parser.add_argument("--pufmpp-checkpoint", type=Path, default=Path(CHECKPOINTS["pufmpp"]))
    parser.add_argument("--worker", choices=["pufm", "pufmpp"], help=argparse.SUPPRESS)
    parser.add_argument("--job-manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.patch_batch_size < 1:
        parser.error("--patch-batch-size must be positive")
    if args.worker:
        worker(read_json(args.job_manifest), args.worker)
        return
    # Resolve repository-relative arguments consistently even when invoked elsewhere.
    args.repo = args.repo.resolve()
    for name in ("output", "pufm_checkpoint", "pufmpp_checkpoint", "root", "schedule", "whole_data_root"):
        path = getattr(args, name)
        if path is not None and not path.is_absolute():
            setattr(args, name, args.repo / path)
    suite = make_suite(args)
    print_suite(suite)
    if args.phase == "plan":
        print(json.dumps(suite, ensure_ascii=False, indent=2))
        return
    suite["plans"] = [preflight(plan) for plan in suite["plans"]]
    if args.phase == "check":
        print("All requested subset checkpoint/data checks passed. No GPU inference/calibration was run.", flush=True)
        return
    root = Path(suite["output"])
    manifest = root / "suite_evaluation_manifest.json"
    if manifest.exists() and read_json(manifest) != suite:
        raise ValueError("Output belongs to different splits/subset/weights/code/settings. Choose a new --output")
    # Check all per-split owners before writing anything or starting a GPU job.
    for plan in suite["plans"]:
        owner = Path(plan["output"]) / "subset_evaluation_manifest.json"
        if owner.exists() and read_json(owner) != plan:
            raise ValueError(f"Split output belongs to a different configuration: {owner}. Choose a new --output")
    root.mkdir(parents=True, exist_ok=True)
    write_json(manifest, suite)
    write_json(root / "subset_selections.json", {plan["split"]: plan["selection"] for plan in suite["plans"]})
    for plan in suite["plans"]:
        split_root = Path(plan["output"])
        owner = split_root / "subset_evaluation_manifest.json"
        write_json(owner, plan)
        write_json(split_root / "subset_selection.json", plan["selection"])
        for model in plan["models"]:
            run_child([sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--worker", model,
                       "--job-manifest", str(owner)], split_root / "logs" / f"{model}.log")
    summarize_suite(suite)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STOPPED: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
