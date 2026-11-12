"""Run the fixed Our2 diagnosis queue against PUFM-60 and the original v3 data."""

import argparse
from collections import defaultdict
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from types import SimpleNamespace

from our2_p0 import (
    CONDITIONS, REPO, SHA60, assert_input, baseline_reference, dataset_for, frozen_plan,
    historical_inputs, historical_rows, input_record, load_config, read_json, read_rows, run,
    sample_indices, write_rows,
)

import numpy as np
import torch

from adaptive_resampling.v3_metrics import evaluate_sample
from our2_diagnosis_inference import measure, predict_paths, proxy_rows, render_example, save_example
from our2_diagnosis_report import report
from our2_diagnosis_training import TRAIN_GROUPS, load_trained, probe, train_group
from our2_v56 import EXAMPLE_CONDITIONS


STAGES = {"B0": "baseline", "M1": "m1_refine", "M2": "m2_hard", "Joint": "joint_finetune"}
WHOLE = ("pu1k_object", "modelnet40_object")


class Queue:
    def __init__(self, output):
        self.output = output
        self.path = output / "queue.json"
        self.state = read_json(self.path) if self.path.exists() else {"tasks": {}}
        self.failures = {}

    def save(self, status):
        run.write_json(self.path, {**self.state, "status": status})
        run.write_json(self.output / "failures.json", self.failures)

    def task(self, name, action, dependencies=()):
        tasks = self.state["tasks"]
        prior = tasks.get(name, {})
        folder = self.output / name
        if any(tasks.get(dep, {}).get("status") != "completed" for dep in dependencies):
            self.failures[name] = {"reason": "DEPENDENCY_UNAVAILABLE", "dependencies": list(dependencies)}
            tasks[name] = {"status": "blocked"}
        elif name != "reports" and prior.get("status") == "completed" and prior.get("dependencies") == {
            dep: run.signature(tasks[dep]) for dep in dependencies
        } and all((self.output / file).is_file() and run.file_hash(self.output / file) == digest
                  for file, digest in prior["files"].items()):
            print(f"SKIP {name}: verified cache", flush=True)
            return
        elif prior.get("status") in ("completed", "invalid") and name.startswith("matched_training/"):
            self.failures[name] = {"reason": "TRAINING_CACHE_CHANGED", "path": str(folder)}
            tasks[name] = {**prior, "status": "invalid"}
        else:
            tasks[name] = {"status": "running"}
            self.save("running")
            print(f"START {name}", flush=True)
            try:
                result = action(folder)
                if not result:
                    raise ValueError("No artifacts produced")
                files = {str(path.relative_to(self.output)): run.file_hash(path) for path in result}
                tasks[name] = {"status": "completed", "files": files,
                               "dependencies": {dep: run.signature(tasks[dep]) for dep in dependencies}}
            except Exception as exc:
                self.failures[name] = {"reason": type(exc).__name__, "message": str(exc),
                                       "traceback": traceback.format_exc()}
                tasks[name] = {"status": "failed"}
                print(self.failures[name]["traceback"], flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self.save("running")


def checkpoint_path(args, name):
    if name == "B0":
        return args.baseline_checkpoint
    if name == "B1":
        return args.p0_results / "v1_extra5/train/best.pt"
    return args.training_run / STAGES[name] / "best.pt"


def load(args, name, training_stage=None):
    stage = training_stage or STAGES.get(name, "baseline")
    path = args.training_run / stage / "best.pt" if training_stage else checkpoint_path(args, name)
    config = load_config(args, stage)
    options = SimpleNamespace(**vars(args), command="predict", resume=None, allow_debug=False)
    options.checkpoint = None if name == "B0" else path
    if name == "B1":
        config = read_json(path.parent / "config.json")
    if args.debug and options.checkpoint:
        # A local smoke run can use a different Torch/FPS runtime; all weight/data/source checks still apply.
        from models.dual import DualPUFM
        saved = run.read_checkpoint(path)
        expected = {"baseline_sha256": SHA60, "metadata_sha256": run.data_identity(args.root)}
        source = saved["identity"]["code_sha256"]
        current = run.code_identity()
        source_matches = source.keys() == current.keys() and all(
            digest == current[key] or digest == hashlib.sha256(
                (REPO / key).read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest() for key, digest in source.items()
        )
        if not source_matches or saved["config"] != config or not saved["training_complete"] or saved["partial_debug_run"] or any(
            saved["identity"][key] != value for key, value in expected.items()
        ):
            raise ValueError(f"Incompatible checkpoint {path}")
        model = DualPUFM(config).to(args.device)
        model.load_state_dict(saved["model"], strict=True)
    else:
        model, _ = run.make_model(options, config)
    if options.checkpoint:
        selected = read_json(path.parent / "best.json")["epoch"]
        if selected != run.read_checkpoint(path)["completed_epochs"]:
            raise ValueError(f"Best epoch does not agree with the checkpoint: {path}")
    return model.eval().requires_grad_(False)


def git(*argv):
    return subprocess.check_output(["git", *argv], cwd=REPO, text=True, encoding="utf-8").strip()


def manifest(args):
    paths = {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)}
    inputs = {}
    for name in (*STAGES, "B1"):
        path = checkpoint_path(args, name)
        inputs[name] = {"path": str(path), "sha256": run.file_hash(path) if path.is_file() else None}
        best = path.parent / "best.json"
        if name != "B0" and best.is_file():
            inputs[name]["epoch"] = read_json(best)["epoch"]
    for stage in ("m1_hole", "m2_soft", "joint_condition"):
        path = args.training_run / stage / "best.pt"
        inputs[stage] = {"path": str(path), "sha256": run.file_hash(path) if path.is_file() else None}
    files = [*Path(__file__).parent.glob("our2_diagnosis_*.py"), Path(__file__)]
    files += [REPO / "tools" / name for name in ("our2_p0.py", "our2_v56.py", "our2_v6_seeds.py")]
    history = [args.p0_results / "v0.json", args.p0_results / "v1_extra5/train/last.pt",
               args.v56_root / "identity.json",
               args.v56_root / "v5/patch/validation/per_sample.csv",
               args.v56_root / "v6/patch_learned/validation/per_sample.csv",
               *(args.p0_results / "v2_v3" / split / "summary.json" for split in ("validation", *WHOLE)),
               args.training_run / "m1_rel/validation/epoch_0000/per_sample.csv",
               *(args.previous_eval / split / "baseline" / name for split in WHOLE
                 for name in ("frozen_plan.json", "per_sample.csv"))]
    result = {"paths": paths, "seed": 21, "epochs": args.epochs, "debug": args.debug,
              "whole_regression": args.whole_regression, "checkpoints": inputs,
              "git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
              "runtime": run.runtime_identity(args), "data_metadata_sha256": run.data_identity(args.root),
              "training_source_sha256": run.code_identity(),
              "diagnosis_source_sha256": {p.name: run.file_hash(p) for p in sorted(files)},
              "historical_sha256": {str(p): run.file_hash(p) for p in history if p.is_file()},
              "budget": {"lambda": .25, "rho_max": .05, "random_seeds": [21, 22, 23]},
              "coordinates": {"forward": "v3 network-normalized XYZ from observed input",
                              "final_metrics": "v3 canonical XYZ", "mechanism_metrics": "network-normalized XYZ"}}
    if inputs["B0"]["sha256"] != SHA60:
        raise ValueError("Wrong PUFM-60 checkpoint SHA256")
    marker = args.output / "manifest.json"
    if marker.exists():
        old = read_json(marker)
        volatile = {"git_head", "git_dirty"}
        if {k: v for k, v in old.items() if k not in volatile} != {
            k: v for k, v in result.items() if k not in volatile
        }:
            raise ValueError("Existing run has a different source/data/runtime/config fingerprint. Use a new --output.")
        return old
    run.write_json(marker, result)
    return result


def observed_for(args, condition, sample, reference):
    points = torch.as_tensor(sample["input_xyz"], device=args.device)[None].float()
    record = input_record(condition, sample, points)
    assert_input(record, reference)
    if sample["gt_xyz"].shape != (4 * points.shape[1], 3) or not np.isfinite(sample["gt_xyz"]).all():
        raise ValueError(f"Invalid GT point budget for {sample['sample_id']}")
    key = {"condition": condition, "sample_id": record[1], "object_id": sample["object_id"],
           "input_sha256": record[2], "gt_sha256": run.tensor_hash(torch.from_numpy(sample["gt_xyz"]))}
    return points, key


@torch.no_grad()
def baseline_identity(args, split, folder):
    model = load(args, "B0")
    native = baseline_reference(args, model)
    plan = frozen_plan(args, split, folder)
    reference = historical_rows(args, split)
    inputs = {key: row["input_sha256"] for key, row in reference.items()}
    rows = []
    for condition in CONDITIONS:
        ds = dataset_for(args, plan, condition)
        try:
            sample = ds[sample_indices(ds, plan)[0]]
            points, key = observed_for(args, condition, sample, inputs)
            prediction = model(points)
            torch.testing.assert_close(native(points), prediction, rtol=0, atol=0)
            digest = run.tensor_hash(prediction)
            same = digest == reference[condition, key["sample_id"]]["prediction_sha256"]
            if not same and not args.debug:
                raise ValueError(f"B0 does not match historical prediction: {split}/{condition}")
            rows.append({**key, "prediction_sha256": digest, "historical_hash_match": same})
        finally:
            ds.close()
    path = folder / "baseline_equivalence.json"
    run.write_json(path, {"status": "passed", "cases": rows, "debug": bool(args.debug)})
    return [path]


def write_evidence(folder, evidence):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in evidence.items():
        if rows:
            path = folder / f"{name}.csv"
            write_rows(path, rows)
            paths.append(path)
    return paths


@torch.no_grad()
def frozen(args, mode, split, conditions, folder, fixed=False, regression=False):
    name = {"m1": "M1", "m2": "B0", "joint": "Joint"}[mode]
    model = load(args, name)
    head = load(args, "M2").m2 if mode == "m2" else None
    if (mode == "m1" and model.m2 is not None) or (mode == "m2" and model.m1 is not None):
        raise ValueError("Independent paths have an unexpected module")
    plan = frozen_plan(args, split, folder)
    objects = {item["object_id"] for item in plan["selection"]["objects"][:2]} if fixed else set()
    reference = historical_inputs(args, split)
    archive = {}
    old_dir = {"m1": "v5/patch", "m2": "v6/patch_learned"}.get(mode)
    if old_dir and split == "validation" and not args.debug:
        path = args.v56_root / old_dir / split / "per_sample.csv"
        if path.exists():
            archive = {(r["sample_id"], r["group"]): r for r in read_rows(path)}
    evidence = defaultdict(list)
    saved = []
    groups = {"m1": ("M1-full", "M1-off"), "m2": ("D0", "D1-L", "D1-U"),
              "joint": ("J0", "J3", "J4")}[mode] if regression else None
    scope = "whole_regression" if regression else "fixed_whole" if fixed else "patch"
    for condition in conditions:
        ds = dataset_for(args, plan, condition)
        seen = set()
        try:
            indices = sample_indices(ds, plan)
            if args.debug and not fixed:
                indices = indices[:args.debug]
            for index in indices:
                sample = ds[index]
                if fixed and sample["object_id"] not in objects:
                    continue
                points, key = observed_for(args, condition, sample, reference)
                key.update(split=split, scope=scope,
                           flow=name, checkpoint_sha256=checkpoint_sha(args, name))
                initial, context, traces = predict_paths(mode, model, points, head, sample["sample_id"], groups)
                primary = {"m1": "M1-full", "m2": "D0", "joint": "J3"}[mode]
                if not seen:
                    torch.testing.assert_close(model(points), traces[primary][-1], rtol=0, atol=0)
                rows = measure(sample, mode, initial, context, traces, key, evidence, full_trace=not regression)
                for row in rows:
                    previous = archive.get((row["sample_id"], row["group"]))
                    if previous:
                        if (previous["input_sha256"] != row["input_sha256"]
                                or previous["prediction_sha256"] != row["prediction_sha256"]):
                            raise ValueError(f"{row['group']} historical V5/V6 identity mismatch")
                        row["historical_hash_match"] = True
                evidence["per_sample"].extend(rows)
                if not seen and mode == "m2" and scope == "patch":
                    evidence["proxy_gain"].extend(proxy_rows(context[0], sample, key))
                if not regression and (fixed or not seen) and condition in EXAMPLE_CONDITIONS:
                    stem = f"{condition}_{run.signature(sample['object_id'])[:10]}"
                    example, figure = folder / "examples" / f"{stem}.npz", folder / "figures" / f"{stem}.png"
                    save_example(example, sample, initial, context, traces, mode)
                    render_example(example, figure, mode)
                    saved += [example, figure]
                seen.add(sample["sample_id"])
        finally:
            ds.close()
        expected = 2 if fixed else args.debug or sum(c == condition for c, _ in reference)
        if len(seen) != expected:
            raise ValueError(f"Unexpected sample count {split}/{condition}: {len(seen)} != {expected}")
    return write_evidence(folder, evidence) + saved


def checkpoint_sha(args, name):
    return args.checkpoint_info[name]["sha256"]


@torch.no_grad()
def evaluate_models(args, names, split, folder):
    plan = frozen_plan(args, split, folder)
    reference = historical_inputs(args, split)
    rows = []
    for name in names:
        if name in TRAIN_GROUPS:
            training = args.output / "matched_training" / name
            metadata = read_json(training / "matched.json")
            head = load(args, "M2").m2 if name.startswith("D1") else None
            model = load_trained(args, name, Path(metadata.get("source", training)), head)
            digest = metadata["last_sha256"]
        else:
            model = load(args, name)
            digest = checkpoint_sha(args, name)
        for condition in CONDITIONS:
            ds = dataset_for(args, plan, condition)
            try:
                indices = sample_indices(ds, plan)
                for index in indices[:args.debug or None]:
                    sample = ds[index]
                    points, key = observed_for(args, condition, sample, reference)
                    output = model(points)
                    rows.append({**key, "group": name, "split": split, "flow": name,
                                 "scope": "patch" if split == "validation" else "whole_regression",
                                 "checkpoint_sha256": digest, "prediction_sha256": run.tensor_hash(output),
                                 **evaluate_sample(output[0], sample)})
            finally:
                ds.close()
    return write_evidence(folder, {"per_sample": rows})


def training(args, name, folder):
    head = load(args, "M2").m2 if name.startswith("D1") else None
    if name == "T0" and not args.debug:
        from our2_diagnosis_training import options_for
        options, cfg = options_for(args, name, folder)
        source = checkpoint_path(args, "B1").parent
        metadata = read_json(source / "manifest.json") if (source / "manifest.json").is_file() else {}
        if (source / "last.pt").is_file() and metadata["config"] == cfg and metadata["fused_optimizer"] == options.fused_optimizer:
            last = run.read_checkpoint(source / "last.pt")
            load(args, "B1")
            ds = run.training_dataset(args, cfg)
            try:
                if (last["config"] != cfg or last["training_profile"] != ds.training_profile
                        or last["identity"]["runtime"] != run.runtime_identity(args)
                        or last["identity"]["code_sha256"] != run.code_identity()
                        or last["identity"]["metadata_sha256"] != run.data_identity(args.root)
                        or last["identity"]["baseline_sha256"] != SHA60 or last["partial_debug_run"]
                        or not last["training_complete"] or last["completed_epochs"] != cfg["epochs"]):
                    raise ValueError("B1 last checkpoint does not match the common training protocol")
                info = {"config": cfg, "updates": sum(r["optimizer_steps"] for r in last["history"]),
                        "last_sha256": run.file_hash(source / "last.pt"),
                        "best_sha256": run.file_hash(source / "best.pt"),
                        "source": str(source), "initial_flow_sha256": SHA60, "reused": True}
            finally:
                ds.close()
            path = folder / "matched.json"
            run.write_json(path, info)
            return [path]
    info = train_group(args, name, folder, head)
    path = folder / "matched.json"
    run.write_json(path, info)
    return [path, folder / "last.pt", folder / "best.pt", folder / "train.jsonl", folder / "best.json"]


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {"root": "adaptive_resampling_data_v3", "whole-data-root": "modelnet40_object80_seed21",
                "baseline-checkpoint": "server_runs/patch32/pufm_common_b8_w0_a8_fused/epoch_0060.pt",
                "training-run": "server_runs/our2_modules_from60_seed21_b8",
                "previous-eval": "server_runs/our2_whole_from60_seed21",
                "p0-results": "server_runs/our2_p0_from60_seed21", "v56-root": "server_runs/our2_v56_from60_seed21",
                "output": "server_runs/our2_full_diagnosis_from60_seed21"}
    for key, path in defaults.items():
        parser.add_argument(f"--{key}", type=Path, default=REPO / path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps-backend", choices=("cuda", "torch"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--whole-regression", action="store_true", help="Evaluate key groups on the fixed 80+80 objects")
    parser.add_argument("--debug", choices=(0, 1, 2), type=int, default=0,
                        help="Local Patch inputs per condition and batches per training arm")
    args = parser.parse_args()
    if args.cpu_threads < 1 or (args.debug and args.whole_regression):
        parser.error("Invalid thread count or regression requested with --debug")
    for key in defaults:
        attr = key.replace("-", "_")
        setattr(args, attr, getattr(args, attr).resolve())
    if args.debug:
        args.output = args.output / f"debug_{args.debug}"
    args.epochs = 1 if args.debug else 5
    for protected in (args.root, args.whole_data_root, args.training_run, args.previous_eval,
                      args.p0_results, args.v56_root, REPO / "our2"):
        if args.output.is_relative_to(protected) or protected.is_relative_to(args.output):
            parser.error(f"Choose a separate output directory outside {protected}")
    return args


def main():
    args = parse()
    run.configure_runtime(args)
    run.seed_all(21)
    provenance = manifest(args)
    args.checkpoint_info = provenance["checkpoints"]
    queue = Queue(args.output)
    baseline = "identity/validation"
    queue.task(baseline, partial(baseline_identity, args, "validation"))
    for mode in ("m1", "m2", "joint"):
        for condition in CONDITIONS:
            queue.task(f"evaluation/patch/{mode}/{condition}",
                       partial(frozen, args, mode, "validation", (condition,)), (baseline,))
    for name in ("B1", "M2"):
        queue.task(f"evaluation/patch/{name}", partial(evaluate_models, args, (name,), "validation"), (baseline,))
    if not args.debug:
        for split in WHOLE:
            check = f"identity/{split}"
            queue.task(check, partial(baseline_identity, args, split))
            for mode in ("m1", "m2", "joint"):
                queue.task(f"evaluation/fixed_whole/{split}/{mode}",
                           partial(frozen, args, mode, split, EXAMPLE_CONDITIONS, fixed=True), (check,))
    for stage in ("m1_refine", "m2_soft", "m2_hard", "joint_finetune"):
        def do_probe(folder, stage=stage):
            rows = probe(args, stage, load(args, stage, training_stage=stage))
            return write_evidence(folder, {"gradient_training": rows})
        queue.task(f"train_probes/{stage}", do_probe, (baseline,))
    for name in TRAIN_GROUPS:
        if name == "D0-FT":
            continue  # T0 shares its exact configuration and is also the D0-FT control.
        stage = f"matched_training/{name}"
        queue.task(stage, partial(training, args, name), (baseline,))
        queue.task(f"evaluation/patch/{name}", partial(evaluate_models, args, (name,), "validation"), (stage,))
    if args.whole_regression:
        for split in WHOLE:
            check = f"identity/{split}"
            for mode in ("m1", "m2", "joint"):
                queue.task(f"evaluation/whole_regression/{split}/{mode}",
                           partial(frozen, args, mode, split, CONDITIONS, regression=True), (check,))
            for name in TRAIN_GROUPS:
                if name == "D0-FT":
                    continue
                queue.task(f"evaluation/whole_regression/{split}/{name}",
                           partial(evaluate_models, args, (name,), split), (check, f"matched_training/{name}"))
    queue.task("reports", lambda folder: report(args, queue, folder),
               ())  # Reports remain available even when an independent diagnosis failed.
    status = "partial" if queue.failures else "completed"
    queue.save(status)
    print(f"OUR2 DIAGNOSIS {status.upper()}: {args.output}", flush=True)
    return 1 if queue.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
