#!/usr/bin/env python3
"""Evaluate the trained PUFM baseline and PUFM++ S2 with the existing V3 CLIs.

Run from the Linux repository after activating scdata-v3 and sourcing env_v3.sh.
No training, checkpoint rewriting, package installation, or file deletion is done.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


VALIDATION = ["A0_normal", "B2_sparse_mid", "C2_hole_10", "D2_noise_005",
              "D7_outliers_02", "E1_mixed", "E2_mixed_outliers"]
CONDITIONS = ["A0_normal", "B1_sparse_light", "B2_sparse_mid", "B3_sparse_heavy",
              "C1_hole_05", "C2_hole_10", "C3_hole_20", "D1_noise_002",
              "D2_noise_005", "D3_noise_010", "D4_global_gaussian_005",
              "D5_local10_gaussian_005", "D6_local50_gaussian_005", "D7_outliers_02",
              "D8_laplace_005_unseen", "E1_mixed", "E2_mixed_outliers"]
TEST_SPLITS = ["pu1k_patch", "pu1k_object", "modelnet40_patch"]


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def conditions_for(split):
    return VALIDATION if split == "validation" else CONDITIONS


def phase_splits(phase):
    if phase in {"check", "calibrate"}:
        return []
    if phase in {"test", "all"}:
        return ["validation"] + TEST_SPLITS
    return [phase]


def checkpoint_details(saved, model, path):
    if saved.get("protocol") != "V3" or saved.get("epoch", -1) != 29:
        raise ValueError(f"Expected the completed 30-epoch stage endpoint in {path}")
    if model == "pufmpp":
        if saved.get("args", {}).get("stage") != "s2" or saved.get("result", {}).get("partial_debug_run", True):
            raise ValueError("PUFM++ requires a complete, non-debug S2 checkpoint.")
        return {"stage": "s2", "completed_stage_epochs": saved["epoch"] + 1}
    config = saved.get("configuration", {})
    if saved.get("stage") != "baseline" or saved.get("partial_debug_run", True):
        raise ValueError("PUFM requires a complete, non-debug baseline checkpoint.")
    if config.get("use_rgr", True) or config.get("allocation") != "midpoint" or config.get("condition", True):
        raise ValueError("Selected PUFM checkpoint is not the plain baseline configuration.")
    return {"stage": "baseline", "completed_stage_epochs": saved["epoch"] + 1,
            "configuration": config, "parent_checkpoint": saved.get("args", {}).get("checkpoint")}


def model_command(repo, model, checkpoint, data_root, split, condition, output, schedule, seed):
    if model == "pufmpp":
        command = [sys.executable, "-u", "-B", str(repo / "compare-change/Enhanced_PUFM-main/v3.py"),
                   "predict", "--stage", "s2", "--flow-steps", "6", "--manifold-step", "0.01",
                   "--schedule", str(schedule)]
    else:
        # No preset override: load the exact saved baseline configuration.
        command = [sys.executable, "-u", "-B", str(repo / "our/PUFM-main/adaptive_v3.py"), "predict"]
    return command + ["--device", "cuda", "--workers", "0", "--seed", str(seed),
                      "--root", str(data_root), "--checkpoint", str(checkpoint),
                      "--split", split, "--condition", condition,
                      "--log-every", "100", "--output", str(output)]


def calibration_command(repo, checkpoint, data_root, schedule, seed):
    return [sys.executable, "-u", "-B", str(repo / "compare-change/Enhanced_PUFM-main/v3.py"),
            "schedule", "--stage", "s2", "--device", "cuda", "--workers", "0",
            "--batch-size", "8", "--assignment-workers", "8", "--seed", str(seed),
            "--training-mode", "mixed", "--train-patches-per-object", "32",
            "--root", str(data_root), "--checkpoint", str(checkpoint),
            "--schedule-grid", "51", "--flow-steps", "6", "--output", str(schedule)]


def run_job(command, log, marker, outputs, context, dry_run=False):
    signature = hashlib.sha256(json.dumps([context, command], sort_keys=True).encode()).hexdigest()
    print("\n$ " + shlex.join(command), flush=True)
    if dry_run:
        return
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise RuntimeError(f"Existing job has different inputs: {marker}. Choose a new --output directory.")
        if all(path.is_file() and path.stat().st_size > 0 for path in outputs):
            print("Already completed; reusing the recorded result.", flush=True)
            return
    log.parent.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\n$ " + shlex.join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f"Command failed with exit code {code}; full log: {log}")
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        raise RuntimeError(f"The command did not produce all expected outputs: {outputs}")
    write_json(marker, {"signature": signature, "command": command})


def summarize(root, p2f=False):
    metric_folder = "metrics_p2f" if p2f else "metrics"
    rows = []
    for path in sorted(root.glob(f"*/{metric_folder}/*/*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("partial_debug_evaluation"):
            raise ValueError(f"Partial evaluation cannot enter the full comparison: {path}")
        row = {"model": path.parent.parent.name, "split": item["split"],
               "condition": item["condition"], "objects": item["objects"], "samples": item["samples"]}
        row.update({name: value["mean_over_objects"] for name, value in item["metrics"].items()})
        rows.append(row)
    if not rows:
        return
    suffix = "_p2f" if p2f else ""
    fields = ["model", "split", "condition", "objects", "samples"]
    fields += sorted(set().union(*(row.keys() for row in rows)) - set(fields))
    with (root / f"summary_by_condition{suffix}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    scores = []
    for model in ("pufm", "pufmpp"):
        selected = {r["condition"]: r["cd_l1"] for r in rows if r["model"] == model and r["split"] == "validation"}
        if set(selected) == set(VALIDATION):
            values = {family: [value for condition, value in selected.items() if condition.startswith(family)]
                      for family in "ABCDE"}
            means = {family: sum(value) / len(value) for family, value in values.items()}
            scores.append({"model": model, "cd_l1_five_class_equal": sum(means.values()) / 5,
                           "class_cd_l1": means})
    write_json(root / f"validation_scores{suffix}.json", {"rule": "Equal A/B/C/D/E weights; mean conditions within D/E",
                                                       "scores": scores})
    print(f"Updated comparison table: {root / ('summary_by_condition' + suffix + '.csv')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["check", "calibrate", "validation", *TEST_SPLITS, "test", "all"],
                        default="validation", help="test/all run full validation first, then the three test splits")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, help="V3 data root; otherwise V3_DATA_ROOT or repo/adaptive_resampling_data_v3")
    parser.add_argument("--output", type=Path, default=Path("server_runs/evaluation_20261030"))
    parser.add_argument("--pufm-checkpoint", type=Path,
                        default=Path("server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt"))
    parser.add_argument("--pufmpp-checkpoint", type=Path,
                        default=Path("server_runs/patch32/pufmpp_s2_b8_w0_fused/last.pt"))
    parser.add_argument("--p2f", action="store_true", help="Requires all original meshes; reuses existing predictions")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without opening checkpoints or running models")
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    os.environ["V3_CODE_ROOT"] = str(repo)
    data_root = (args.root or Path(os.environ.get("V3_DATA_ROOT", str(repo / "adaptive_resampling_data_v3")))).resolve()
    os.environ["V3_DATA_ROOT"] = str(data_root)
    os.environ["PYTHONFAULTHANDLER"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    root = args.output.resolve()
    checkpoints = {"pufm": args.pufm_checkpoint.resolve(), "pufmpp": args.pufmpp_checkpoint.resolve()}
    splits = phase_splits(args.phase)
    schedule = root / "pufmpp_s2_schedule.pt"
    context = {"runner_revision": 1, "seed": args.seed, "data_root": str(data_root)}
    if not args.dry_run:
        import torch
        sys.path.insert(0, str(repo))
        from adaptive_resampling.v3_core import CONDITIONS as actual_conditions, VALIDATION_CONDITIONS
        from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
        from adaptive_resampling.v3_training import require_training_profile
        import numpy
        import scipy
        import h5py
        import ssl
        context["environment"] = {"python": sys.version, "torch": str(torch.__version__),
                                  "torch_cuda": torch.version.cuda, "numpy": numpy.__version__,
                                  "scipy": scipy.__version__, "h5py": h5py.__version__,
                                  "hdf5": h5py.version.hdf5_version, "openssl": ssl.OPENSSL_VERSION}
        if list(actual_conditions) != CONDITIONS or list(VALIDATION_CONDITIONS) != VALIDATION:
            raise ValueError("Repository condition definitions differ from this runner; update the runner first.")
        train = AdaptiveResamplingDataset(data_root, "train", include_supervision=False, train_patches_per_object=32)
        profile = train.training_profile
        train.close()
        context["training_profile"] = profile
        context["checkpoints"] = {}
        for model, path in checkpoints.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {model} checkpoint: {path}. Use --{model}-checkpoint to select its actual location.")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            details = checkpoint_details(saved, model, path)
            require_training_profile(saved, profile, context="Evaluation")
            del saved
            gc.collect()
            context["checkpoints"][model] = {"path": str(path), "sha256": digest_file(path), **details}
            print(json.dumps({"model": model, **details}, ensure_ascii=False), flush=True)
        sources = [repo / "compare-change/v3_common.py", repo / "compare-change/Enhanced_PUFM-main/v3_adapter.py",
                   repo / "our/PUFM-main/adaptive_v3.py", repo / "tools/evaluate_v3_predictions.py"]
        for folder in ("adaptive_resampling", "our/PUFM-main/models", "compare-change/Enhanced_PUFM-main/models"):
            sources.extend((repo / folder).rglob("*.py"))
        source_hashes = {str(p.relative_to(repo)): digest_file(p) for p in sorted(set(sources))}
        context["source_sha256"] = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
        for name in ("base_manifest.json", "generation_config.json"):
            context[name + "_sha256"] = digest_file(data_root / "metadata" / name)
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "evaluation_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != context:
            raise ValueError("This output directory belongs to different weights/data/code. Select a new --output directory.")
        write_json(manifest, context)
        total_bytes, largest = 0, 0
        for split in splits:
            sample_set = AdaptiveResamplingDataset(data_root, split, condition=conditions_for(split)[0], include_supervision=False)
            count = len(sample_set)
            sample_set.close()
            point_count = 8192 if split == "pu1k_object" else 1024
            size = count * point_count * 3 * 4
            total_bytes += size * len(conditions_for(split)) * 2
            largest = max(largest, size)
            print(f"{split}: {count} samples per condition; {len(conditions_for(split))} conditions per model", flush=True)
        free = shutil.disk_usage(root).free
        print(f"Free disk: {free / 2**30:.2f} GiB; raw XYZ for requested phases: {total_bytes / 2**30:.2f} GiB (plus IDs, logs, CSV and diagnostics).", flush=True)
        if free < largest * 2 + 512 * 2**20:
            raise RuntimeError("Insufficient disk headroom for the next prediction and result files.")
        if args.p2f:
            from adaptive_resampling.v3_paths import SourcePaths
            paths = SourcePaths(data_root)
            mesh_splits = json.loads((data_root / "metadata/dataset_splits.json").read_text(encoding="utf-8"))
            missing = []
            for key in {"validation" if s == "validation" else "modelnet40_test" if s == "modelnet40_patch" else "pu1k_test" for s in splits}:
                missing.extend(paths.resolve(r["path"]) for r in mesh_splits[key] if not Path(paths.resolve(r["path"])).is_file())
            if missing:
                raise FileNotFoundError(f"P2F needs {len(missing)} missing original mesh files; examples: {missing[:3]}. Set V3_RAW_ROOT and upload all required meshes.")
        if args.phase == "check":
            print("Checkpoint/data checks passed. These are fixed-budget endpoint weights, not automatically selected best weights.", flush=True)
            return
    calibration_done = False
    if args.phase == "calibrate":
        run_job(calibration_command(repo, checkpoints["pufmpp"], data_root, schedule, args.seed),
                root / "logs/schedule.log", root / "schedule.done.json", [schedule], context, args.dry_run)
        return
    for split in splits:
        for model in ("pufm", "pufmpp"):
            if model == "pufmpp" and not calibration_done:
                print("PUFM++ calibration scans the full selected training set at 51 time positions; no weight updates.", flush=True)
                run_job(calibration_command(repo, checkpoints[model], data_root, schedule, args.seed),
                        root / "logs/schedule.log", root / "schedule.done.json", [schedule], context, args.dry_run)
                calibration_done = True
            for condition in conditions_for(split):
                prediction = root / split / "predictions" / f"{model}_{condition}.npz"
                job = f"{model}_{condition}"
                outputs = [prediction, prediction.with_suffix(".json")]
                run_job(model_command(repo, model, checkpoints[model], data_root, split, condition,
                                      prediction, schedule, args.seed),
                        root / split / "logs" / f"{job}_predict.log",
                        root / split / "status" / f"{job}_predict.done.json", outputs, context, args.dry_run)
                metric_folder = "metrics_p2f" if args.p2f else "metrics"
                metrics = root / split / metric_folder / model / condition
                command = [sys.executable, "-u", "-B", str(repo / "tools/evaluate_v3_predictions.py"),
                           str(prediction), "--root", str(data_root), "--split", split,
                           "--condition", condition, "--coordinates", "model", "--output", str(metrics)]
                if args.p2f:
                    command.append("--p2f")
                run_job(command, root / split / "logs" / f"{job}_{metric_folder}.log",
                        root / split / "status" / f"{job}_{metric_folder}.done.json",
                        [metrics / "summary.json", metrics / "per_object.csv", metrics / "per_sample.csv"],
                        context, args.dry_run)
                if not args.dry_run:
                    summarize(root, args.p2f)
    if not args.dry_run:
        print(f"Completed requested phases. Results: {root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STOPPED: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
