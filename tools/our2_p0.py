"""Run the first Our2 validation batch without changing the trained model sources."""

import argparse
import ast
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from our2 import run
from our2.config import DEFAULT, validate_config
from run_subset_eval import CONDITIONS, evaluation_dataset, flatten, make_plan, validate_supplement

SHA60 = "a5051af124fba4112bb118196e2f353cfbc2f43ec8385ef1b2f1cb8cb1e71599"
SPLITS = ("validation", "pu1k_object", "modelnet40_object")
REGRESSION_SPLITS = SPLITS[1:]
MAIN_METRICS = (
    "cd_l1", "cd_l2_squared", "f_score", "hausdorff_l2", "precision", "recall",
    "hole_coverage_mean_l2", "hole_coverage_fraction",
    "sparse_coverage_mean_l2", "sparse_coverage_fraction", "gt_coverage_mean_l2", "prediction_seconds",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(args, stage):
    config = {**DEFAULT, **read_json(args.training_run / "configs" / f"{stage}.json")}
    validate_config(config)
    if config["stage"] != stage or config["baseline_sha256"] != SHA60:
        raise ValueError(f"{stage} does not belong to the PUFM-60 run")
    return config


def identity(args):
    queue = read_json(args.training_run / "queue.json")
    if queue["status"] != "completed" or queue["identity"]["baseline_sha256"] != SHA60:
        raise ValueError("Expected the completed Our2 queue from PUFM-60")
    provenance = {
        "baseline_sha256": run.file_hash(args.baseline_checkpoint),
        "metadata_sha256": run.data_identity(args.root),
        "training_source_sha256": run.code_identity(),
        "runtime": run.runtime_identity(args),
    }
    if provenance["baseline_sha256"] != SHA60:
        raise ValueError("Expected the original epoch_0060.pt, SHA256 " + SHA60)
    for key, original in (("metadata_sha256", "data_sha256"),
                          ("training_source_sha256", "training_source_sha256")):
        if provenance[key] != queue["identity"][original]:
            raise ValueError(f"{key} differs from the original training queue")
    manifest = read_json(args.training_run / "m1_refine" / "manifest.json")
    if provenance["runtime"] != manifest["runtime"]:
        raise ValueError("Use the recorded training environment, GPU, FPS backend and CPU threads")
    provenance["diagnostic_source_sha256"] = {
        path.name: run.file_hash(path)
        for path in (Path(__file__), Path(__file__).with_name("our2_p0_m2.py"))
    }
    provenance["reference_source_sha256"] = {
        name: run.file_hash(REPO / "baseline" / "PUFM-main" / "models" / name)
        for name in ("diffusion.py", "utils.py")
    }
    provenance["config_sha256"] = {
        stage: run.file_hash(args.training_run / "configs" / f"{stage}.json")
        for stage in ("baseline", "m1_refine", "m2_hard", "joint_finetune")
    }
    provenance["checkpoint_sha256"] = {
        stage: run.file_hash(args.training_run / stage / "best.pt")
        for stage in ("m1_refine", "m2_hard", "joint_finetune")
    }
    return provenance


def load_model(args, stage):
    options = SimpleNamespace(**vars(args))
    options.command, options.resume, options.allow_debug = "predict", None, False
    options.checkpoint = None if stage == "baseline" else args.training_run / stage / "best.pt"
    model, _ = run.make_model(options, load_config(args, stage))
    return model.eval().requires_grad_(False)


def frozen_plan(args, split, output):
    if split == "validation":
        return None
    options = SimpleNamespace(
        repo=REPO, root=args.root, output=output, split=split, objects=80, patches_per_object=None,
        seed=21, conditions=CONDITIONS, models=["pufm"], calibration_samples=256,
        fps_backend=args.fps_backend, patch_batch_size=1, schedule=None, p2f=False,
        device=args.device, pufm_checkpoint=args.baseline_checkpoint,
        whole_data_root=args.whole_data_root,
    )
    plan = make_plan(options)
    validate_supplement(plan)
    historical = read_json(args.previous_eval / split / "baseline" / "frozen_plan.json")
    for key in ("selection_sha256", "supplement_manifest_sha256"):
        if plan.get(key) != historical.get(key):
            raise ValueError(f"{split}: frozen {key} differs from the previous evaluation")
    return plan


def dataset_for(args, plan, condition):
    if plan is not None:
        return evaluation_dataset(plan, condition, include_supervision=True)
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    return AdaptiveResamplingDataset(
        args.root, "validation", condition=condition, include_supervision=True, augment_rotation=False,
    )


def sample_indices(dataset, plan):
    return list(range(len(dataset))) if plan is None else [i for i, _ in flatten(plan["selection"])]


def input_record(condition, sample, observed):
    return [condition, sample["sample_id"], run.tensor_hash(observed)]


def historical_rows(args, split):
    if split == "validation":
        path = args.training_run / "m1_rel" / "validation" / "epoch_0000" / "per_sample.csv"
    else:
        path = args.previous_eval / split / "baseline" / "per_sample.csv"
    rows = read_rows(path)
    indexed = {(row["condition"], row["sample_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate historical samples in {path}")
    return indexed


def historical_inputs(args, split):
    return {key: row["input_sha256"] for key, row in historical_rows(args, split).items()}


def assert_input(record, reference):
    condition, sample_id, digest = record
    if reference.get((condition, sample_id)) != digest:
        raise ValueError(f"Frozen input changed: {condition} / {sample_id}")


def baseline_reference(args, wrapper):
    """Build the original flow from its own source with the shared portable point ops."""
    from models.utils import midpoint_interpolate
    source = REPO / "baseline" / "PUFM-main" / "models"
    copied = REPO / "our2" / "PUFM-main" / "models"
    if ast.dump(ast.parse((source / "diffusion.py").read_text(encoding="utf-8"))) != ast.dump(
        ast.parse((copied / "diffusion.py").read_text(encoding="utf-8"))
    ):
        raise ValueError("Original and Our2 flow source differ")

    def midpoint_tree(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return ast.dump(next(node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name == "midpoint_interpolate"))

    if midpoint_tree(source / "utils.py") != midpoint_tree(copied / "utils.py"):
        raise ValueError("Original and portable Midpoint functions differ")
    spec = importlib.util.spec_from_file_location("our2_p0_original_flow", source / "diffusion.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    flow = reference.PUFM_w_attn(SimpleNamespace()).to(args.device).eval().requires_grad_(False)
    flow.load_state_dict(wrapper.flow.state_dict(), strict=True)

    def predict(observed):
        initial = midpoint_interpolate(SimpleNamespace(up_rate=4), observed.transpose(1, 2).contiguous())
        query = initial
        for step in range(5):
            query = query + flow(query, initial, query.new_full((len(query),), step / 5)) / 5
        return query.transpose(1, 2)

    return predict


def check_branches(args, observed):
    import torch
    m1 = load_model(args, "m1_refine")
    state = m1.prepare(observed)
    if m1.m2 is not None or "budget" in state:
        raise ValueError("M1-only unexpectedly applies a regional budget")
    del m1, state
    m2 = load_model(args, "m2_hard")
    state = m2.prepare(observed)
    if m2.m1 is not None or state["m1"] is not None or state["budget"]["hard"].sum().item() != 4 * observed.shape[1]:
        raise ValueError("M2-only budget or M1 independence check failed")
    del m2, state
    joint = load_model(args, "joint_finetune")
    original_switch = joint.config["joint_condition"]
    if not original_switch:
        raise ValueError("Expected the trained Joint conditioning path to be enabled")
    enabled = joint.prepare(observed)
    joint.config["joint_condition"] = False
    disabled = joint.prepare(observed)
    geometry, m1 = enabled["geo"], enabled["m1"]
    policy = joint.config["budget_policy"]
    torch.testing.assert_close(enabled["logits"], joint.m2(geometry, m1, True, policy))
    torch.testing.assert_close(disabled["logits"], joint.m2(geometry, None, False, policy))
    return float((enabled["logits"] - disabled["logits"]).abs().max())


def v0(args, provenance):
    import torch
    from adaptive_resampling.v3_metrics import evaluate_sample

    wrapper = load_model(args, "baseline")
    direct = baseline_reference(args, wrapper)
    cases = []
    with torch.inference_mode():
        for split in SPLITS:
            plan = frozen_plan(args, split, args.output / "v0")
            historical = historical_rows(args, split)
            reference = {key: row["input_sha256"] for key, row in historical.items()}
            for condition in CONDITIONS:
                dataset = dataset_for(args, plan, condition)
                try:
                    sample = dataset[sample_indices(dataset, plan)[0]]
                    observed = torch.as_tensor(sample["input_xyz"], device=args.device)[None].float()
                    record = input_record(condition, sample, observed)
                    assert_input(record, reference)
                    expected = 1024 if split == "validation" else 8192
                    native, wrapped = direct(observed), wrapper(observed)
                    if tuple(wrapped.shape) != (1, expected, 3):
                        raise ValueError(f"Unexpected output budget: {split} {wrapped.shape}")
                    torch.testing.assert_close(native, wrapped, atol=1e-6, rtol=1e-6)
                    native_metrics = evaluate_sample(native[0], sample)
                    wrapped_metrics = evaluate_sample(wrapped[0], sample)
                    deltas = {key: wrapped_metrics[key] - native_metrics[key]
                              for key in ("cd_l1", "f_score", "hausdorff_l2")}
                    if max(map(abs, deltas.values())) > 1e-5:
                        raise ValueError(f"V0 metric mismatch: {record[:2]} {deltas}")
                    wrapper_hash = run.tensor_hash(wrapped)
                    cases.append({
                        "split": split, "condition": condition, "sample_id": sample["sample_id"],
                        "input_sha256": record[-1], "output_points": expected,
                        "native_sha256": run.tensor_hash(native), "wrapper_sha256": wrapper_hash,
                        "historical_prediction_sha256_match": wrapper_hash == historical[tuple(record[:2])]["prediction_sha256"],
                        "max_abs_coordinate_delta": float((native - wrapped).abs().max()),
                        "metric_deltas": deltas,
                    })
                    if split == "validation" and condition == "A0_normal":
                        branch_delta = check_branches(args, observed)
                finally:
                    dataset.close()
            print(f"V0 passed {split} ({len(CONDITIONS)} conditions)", flush=True)
    report = {
        "status": "passed", "identity": provenance,
        "reference": "original PUFM flow source, same portable point ops and Midpoint",
        "baseline_model_sha256": run.model_hash(wrapper), "cases": cases,
        "joint_condition_logit_max_abs_delta": branch_delta,
    }
    run.write_json(args.output / "v0.json", report)


def compare_v1(args, split):
    previous = args.previous_eval / split
    new = args.output / "v1_extra5" / split
    folders = {"B0": previous / "baseline", "B1": new, "M1": previous / "m1_refine"}
    summaries = {name: read_json(path / "summary.json") for name, path in folders.items()}
    sample_hashes = {summary["input_selection_sha256"] for summary in summaries.values()}
    if len(sample_hashes) != 1 or any(s["samples"] != 560 or s["baseline_sha256"] != SHA60
                                     for s in summaries.values()):
        raise ValueError(f"V1 inputs or baseline differ: {split}")
    for key in ("metadata_sha256", "code_sha256", "runtime"):
        if any(s[key] != summaries["B0"][key] for s in summaries.values()):
            raise ValueError(f"V1 {key} differs: {split}")
    if summaries["M1"]["checkpoint_sha256"] != run.file_hash(args.training_run / "m1_refine" / "best.pt"):
        raise ValueError(f"V1 historical M1 prediction uses another checkpoint: {split}")

    def metrics(values):
        result = {key: float(values[key]) for key in MAIN_METRICS if values.get(key) not in (None, "")}
        # Repository CD-L1 is the average of the two directional means.
        result["pred_to_gt_mean_l2"] = 2 * result["cd_l1"] - result["gt_coverage_mean_l2"]
        return result

    paired = []
    for candidate, control in (("B1", "B0"), ("M1", "B1")):
        def by_object(name):
            rows = read_rows(folders[name] / "per_object.csv")
            indexed = {(r["condition"], r["object_id"]): r for r in rows}
            if len(rows) != len(indexed) or len(rows) != 560:
                raise ValueError(f"V1 object table has missing or duplicate rows: {name} {split}")
            return indexed

        left, right = by_object(candidate), by_object(control)
        if left.keys() != right.keys():
            raise ValueError(f"V1 object pairs differ: {split} {candidate}/{control}")
        for condition, object_id in sorted(left):
            a, b = metrics(left[condition, object_id]), metrics(right[condition, object_id])
            paired.append({
                "comparison": f"{candidate}-{control}", "condition": condition, "object_id": object_id,
                **{key: value - b[key] for key, value in a.items() if key in b},
            })
    write_rows(new / "v1_paired_deltas.csv", paired)
    table = []
    for group, summary in summaries.items():
        for scope, values in [("macro", summary["macro"]), *summary["conditions"].items()]:
            table.append({"group": group, "condition": scope, **metrics(values)})
    write_rows(new / "v1_comparison.csv", table)


def v1(args):
    baseline, m1 = (load_config(args, name) for name in ("baseline", "m1_refine"))
    if (m1["batch_size"], m1["seed"], m1["epochs"], m1["data_epoch_offset"], m1["lr"]) != (8, 21, 5, 70, 1e-6):
        raise ValueError("M1 refine configuration does not match the requested B1 protocol")
    config = {**m1, "stage": "baseline", "enable_m1": False, "enable_m2": False}
    validate_config(config)
    if not config["train_flow"] or not config["restore_baseline_optimizer"] or baseline["seed"] != m1["seed"]:
        raise ValueError("B1 must fine-tune the original flow with its Adam state")
    m1_manifest = read_json(args.training_run / "m1_refine" / "manifest.json")
    if m1_manifest["config"] != m1:
        raise ValueError("M1 training manifest and queue configuration disagree")
    output = args.output / "v1_extra5"
    config_path = output / "config.json"
    if config_path.is_file() and read_json(config_path) != config:
        raise ValueError("B1 output already belongs to another configuration")
    run.write_json(config_path, config)
    command = [sys.executable, "-u", "-B", str(REPO / "our2" / "run.py")]
    common = [
        "--config", str(config_path), "--root", str(args.root), "--baseline-checkpoint", str(args.baseline_checkpoint),
        "--whole-data-root", str(args.whole_data_root), "--device", args.device,
        "--fps-backend", args.fps_backend, "--cpu-threads", str(args.cpu_threads),
    ]
    train = output / "train"
    last, best = train / "last.pt", train / "best.pt"
    completed = (
        last.is_file() and best.is_file() and
        run.read_checkpoint(last)["training_complete"] and run.read_checkpoint(best)["training_complete"]
    )
    if not completed:
        train_v1(args, config, train, last if last.is_file() else None, m1_manifest["fused_optimizer"])
    history = [json.loads(line) for line in (train / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    if [r["epoch"] for r in history] != list(range(1, 6)) or any(
        r["optimizer_steps"] != 3188 or r["seen_patches"] != 25504 for r in history
    ):
        raise ValueError("B1 did not match M1's five full training epochs and update budget")
    record_flow_updates(args, config, best, last)
    for split in REGRESSION_SPLITS:
        frozen_plan(args, split, output / split)
        subprocess.run(command + ["predict", *common, "--checkpoint", str(best), "--split", split,
                                  "--output", str(output / split)], cwd=REPO, check=True)
        compare_v1(args, split)
        print(f"V1 completed {split}", flush=True)


def train_v1(args, config, output, resume, fused_optimizer):
    """Run the original trainer and annotate its epoch records at checkpoint time."""
    from unittest.mock import patch
    import torch
    from adaptive_resampling.v3_transport import configure_assignment_workers

    options = SimpleNamespace(**vars(args))
    options.command, options.output, options.resume = "train", output, resume
    options.checkpoint = options.init = options.m1_checkpoint = options.m2_checkpoint = None
    options.fused_optimizer, options.allow_debug, options.limit_batches = fused_optimizer, False, 0
    configure_assignment_workers(config["assignment_workers"])
    run.seed_all(config["seed"])
    model, saved = run.make_model(options, config)
    run.seed_all(config["seed"])
    previous = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    recorded_epoch = saved["completed_epochs"] if saved is not None else 0
    gradient_count, gradient_sum, gradient_max = 0, 0.0, 0.0
    clip, save = torch.nn.utils.clip_grad_norm_, run.save_checkpoint

    def trace_clip(*positional, **keywords):
        nonlocal gradient_count, gradient_sum, gradient_max
        norm = clip(*positional, **keywords)
        value = float(norm.detach())
        gradient_count += 1
        gradient_sum += value
        gradient_max = max(gradient_max, value)
        return norm

    def trace_save(path, checkpoint):
        nonlocal recorded_epoch, gradient_count, gradient_sum, gradient_max
        epoch = checkpoint["completed_epochs"]
        if epoch > recorded_epoch:
            record = checkpoint["history"][-1]
            if epoch != recorded_epoch + 1 or gradient_count != record["optimizer_steps"]:
                raise ValueError("Unexpected flow gradient count in B1 training")
            squared = 0.0
            for name, prior in previous.items():
                current = checkpoint["model"][name].detach().cpu().clone()
                squared += float((current.double() - prior.double()).square().sum())
                previous[name] = current
            record.update(flow_gradient_preclip_l2_mean=gradient_sum / gradient_count,
                          flow_gradient_preclip_l2_max=gradient_max, flow_epoch_update_l2=squared ** .5)
            recorded_epoch = epoch
            gradient_count, gradient_sum, gradient_max = 0, 0.0, 0.0
        save(path, checkpoint)

    with patch.object(torch.nn.utils, "clip_grad_norm_", trace_clip), patch.object(run, "save_checkpoint", trace_save):
        run.train(model, saved, options, config)
    del model, saved
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


def record_flow_updates(args, config, best, last):
    from models.dual import DualPUFM, load_baseline
    model = DualPUFM(config)
    load_baseline(model, run.read_checkpoint(args.baseline_checkpoint))
    initial = dict(model.named_parameters())
    report = {}
    for path in (best, last):
        checkpoint = run.read_checkpoint(path)
        squared, maximum = 0.0, 0.0
        for name, parameter in initial.items():
            delta = checkpoint["model"][name].double() - parameter.detach().double()
            squared += float(delta.square().sum())
            maximum = max(maximum, float(delta.abs().max()))
        report[path.stem] = {
            "epoch": checkpoint["completed_epochs"], "flow_update_l2": squared ** .5,
            "flow_update_max_abs": maximum,
        }
    run.write_json(best.parent / "flow_updates.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("v0", "v1", "v2-v3"))
    parser.add_argument("--root", type=Path, default=REPO / "adaptive_resampling_data_v3")
    parser.add_argument("--whole-data-root", type=Path, default=REPO / "modelnet40_object80_seed21")
    parser.add_argument("--baseline-checkpoint", type=Path,
                        default=REPO / "server_runs/patch32/pufm_common_b8_w0_a8_fused/epoch_0060.pt")
    parser.add_argument("--training-run", type=Path, default=REPO / "server_runs/our2_modules_from60_seed21_b8")
    parser.add_argument("--previous-eval", type=Path, default=REPO / "server_runs/our2_whole_from60_seed21")
    parser.add_argument("--output", type=Path, default=REPO / "server_runs/our2_p0_from60_seed21")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps-backend", choices=("cuda", "torch"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--limit-samples", type=int, default=0, help="V2-V3 debug samples per condition")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.limit_samples < 0 or (args.limit_samples and args.phase != "v2-v3"):
        parser.error("Invalid CPU threads or debug limit")
    for key in ("root", "whole_data_root", "baseline_checkpoint", "training_run", "previous_eval", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.is_relative_to(REPO / "our2"):
        parser.error("Keep outputs outside our2")
    run.configure_runtime(args)
    run.seed_all(21)
    provenance = identity(args)
    if args.phase == "v0":
        v0(args, provenance)
    else:
        gate = read_json(args.output / "v0.json")
        if gate["status"] != "passed" or gate["identity"] != provenance:
            raise ValueError("Run V0 successfully with the same source, data, checkpoints and runtime first")
        if args.phase == "v1":
            v1(args)
        else:
            # Keep the same module instance when launched as a script.
            sys.modules["our2_p0"] = sys.modules[__name__]
            from our2_p0_m2 import diagnose
            diagnose(args, provenance)


if __name__ == "__main__":
    main()
