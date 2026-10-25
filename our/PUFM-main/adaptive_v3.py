"""Reproducible staged RGR/MGB training, train-only teachers, and GT-free inference."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_ROOT = Path(os.environ.get("V3_CODE_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(CODE_ROOT))
from adaptive_resampling.v3_dataset import DEFAULT_ROOT, AdaptiveResamplingDataset, EpochSampler, OfficialPU1KDataset
from adaptive_resampling.v3_transport import configure_assignment_workers
from adaptive_resampling.v3_optimizer import make_optimizer, load_optimizer_state
from adaptive_resampling.v3_training import DEFAULT_TRAIN_PATCHES_PER_OBJECT, require_training_profile
from models.adaptive_wrapper import AdaptivePUFM, AdaptiveConfig
from models.adaptive_losses import flow_loss, recovery_losses, gain_loss, retention_loss
from models.adaptive_ops import cd_l1
from gain_teacher import paired_labels, write_cache, generator_signature, stable_seed


def preset(name):
    options = {"baseline": dict(use_rgr=False, use_virtual=False, allocation="midpoint"),
               "rgr": dict(use_rgr=True, use_virtual=False, allocation="midpoint"),
               "rule": dict(use_rgr=False, use_virtual=False, allocation="density"),
               "static": dict(use_rgr=False, use_virtual=False, allocation="static", gain_form="static"),
               "mgb": dict(use_rgr=False, use_virtual=False, allocation="gain"),
               "rgr_static": dict(use_rgr=True, allocation="static", gain_form="static"),
               "full": dict(use_rgr=True, allocation="gain"),
               "full_condition": dict(use_rgr=True, allocation="gain", condition=True)}
    return AdaptiveConfig(**options[name])


def dataset(args, split, supervision=False):
    if split == "official":
        return OfficialPU1KDataset(args.root)
    return AdaptiveResamplingDataset(args.root, split, condition=None if split == "train" else args.condition,
                                     training_mode=args.training_mode, include_supervision=supervision,
                                     augment_rotation=split == "train",
                                     train_patches_per_object=getattr(args, "train_patches_per_object", DEFAULT_TRAIN_PATCHES_PER_OBJECT))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_read(path):
    return torch.load(path, map_location="cpu", weights_only=False) if path else None


def create_model(args):
    saved = checkpoint_read(args.resume or args.checkpoint)
    if args.resume and "configuration" not in saved:
        raise ValueError("Resume requires an adaptive V3 checkpoint")
    if args.resume or (args.command not in {"train"} and args.preset is None and saved and "configuration" in saved):
        config = AdaptiveConfig(**saved["configuration"])
    else:
        native_weights = saved is not None and "configuration" not in saved
        config = preset(args.preset or ("baseline" if native_weights or args.stage == "baseline" and args.command == "train" else "full"))
    for key in ["regions", "capacity", "minimum", "flow_steps", "swap_step", "swap_iterations"]:
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.allocation:
        config.allocation = args.allocation
    if args.backbone:
        config.backbone = args.backbone
    if args.no_virtual:
        config.use_virtual = False
    if args.no_reliability:
        config.use_reliability = False
    if args.no_recovery:
        config.use_recovery = False
    if args.no_uncertainty:
        config.use_uncertainty = False
    model = AdaptivePUFM(config).to(args.device)
    if saved is not None:
        state = saved.get("model", saved.get("state_dict", saved))
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        if any(key.startswith("flow.") for key in state):
            model.load_state_dict(state, strict=True)
        else:
            model.flow.load_state_dict(state, strict=True)
    if args.resume and (asdict(config) != asdict(AdaptiveConfig(**saved["configuration"])) or saved.get("stage") != args.stage):
        raise ValueError("Resume must preserve model configuration and stage; use --checkpoint for a new stage")
    if args.command in {"predict", "build-teacher", "evaluate-gain"}:
        if not saved and not args.allow_debug:
            raise ValueError("This command requires trained --checkpoint weights")
        if saved and saved.get("partial_debug_run") and not args.allow_debug:
            raise ValueError("Debug-only weights require explicit --allow-debug and cannot be reported as trained results")
        if saved and "configuration" in saved and not {"baseline", "joint"}.intersection(saved.get("trained_stages", [])) and not args.allow_debug:
            raise ValueError("The generator has no trained baseline/joint stage")
    return model, saved


def tensor_batch(batch, device):
    return batch["input_xyz"].to(device).float(), batch["gt_xyz"].to(device).float(), {
        key: value.to(device) for key, value in batch.get("supervision", {}).items()}


def save_checkpoint(path, model, optimizer, epoch, args, stages, debug=False):
    torch.save({"protocol": "V3", "configuration": asdict(model.config), "model": model.state_dict(),
                "training_profile": args.training_profile,
                "optimizer": optimizer.state_dict(), "epoch": epoch, "stage": args.stage,
                "trained_stages": stages, "args": vars(args), "partial_debug_run": debug,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, path)


def train(model, saved, args):
    if args.stage in {"rgr", "joint"} and not saved and not args.allow_debug:
        raise ValueError("Recovery/joint stages need --checkpoint; untrained prototypes require --allow-debug")
    if args.stage == "baseline" and (model.config.use_rgr or model.config.allocation != "midpoint" or model.config.condition):
        raise ValueError("Baseline stage must use --preset baseline")
    if args.stage == "rgr" and not model.config.use_rgr:
        raise ValueError("RGR stage needs an RGR-enabled preset")
    if args.stage == "joint" and model.config.allocation == "midpoint":
        raise ValueError("Joint budget training needs an adaptive allocation preset")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [model.flow] if args.stage == "baseline" else [model.rgr] if args.stage == "rgr" else [model.flow, model.rgr, model.conditioner, model.gain]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    ds = dataset(args, "train", supervision=model.config.use_rgr)
    args.training_profile = ds.training_profile
    print(json.dumps({"training_profile": ds.training_profile}), flush=True)
    sampler = EpochSampler(ds)
    loader = DataLoader(ds, sampler=sampler, batch_size=args.batch_size, num_workers=args.workers,
                        persistent_workers=args.workers > 0)
    optimizer = make_optimizer(torch.optim.Adam, [p for p in model.parameters() if p.requires_grad],
                               lr=args.lr, fused=args.fused_optimizer)
    start = 0
    stages = list(saved.get("trained_stages", [])) if saved else []
    if args.resume:
        if saved.get("args", {}).get("training_mode") != args.training_mode:
            raise ValueError("Resume cannot change the training data mode")
        require_training_profile(saved, ds.training_profile)
        load_optimizer_state(optimizer, saved["optimizer"], fused=args.fused_optimizer,
                             disable_fused=getattr(args, "no_fused_optimizer", False))
        start = saved["epoch"] + 1
        torch.set_rng_state(saved["rng"]["torch"].cpu())
        np.random.set_state(saved["rng"]["numpy"])
        random.setstate(saved["rng"]["python"])
        if torch.cuda.is_available() and saved["rng"]["cuda"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["rng"]["cuda"]])
    if args.stage not in stages:
        stages.append(args.stage)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        if args.stage == "rgr":
            model.flow.eval()
        values = []
        for step, batch in enumerate(loader):
            points, gt, supervision = tensor_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            state = model.encode(points) if args.stage == "rgr" else model.prepare(points)
            losses = {}
            if args.stage != "rgr":
                losses["flow"] = flow_loss(model, state, gt)
            if model.config.use_rgr:
                losses.update(recovery_losses(model, state, gt, supervision))
            loss = losses.get("flow", points.new_zeros(())) + args.rgr_weight * losses.get("rgr", points.new_zeros(()))
            if args.stage == "joint" and step % args.online_teacher_every == 0:
                records = paired_labels(model, state, gt, batch["prediction_scale"].to(args.device), pairs=1,
                                        base_method=model.config.allocation, seed=stable_seed(args.seed, epoch, step))
                record = records[0]
                auxiliary, _ = gain_loss(model, state["regions"]["gain_features"], record["budget"], record["receiver"],
                                         record["donor"], record["delta"], record["improvement"])
                loss = loss + args.gain_weight * auxiliary
                losses["gain"] = auxiliary
            if args.stage == "joint" and args.retain_weight:
                prediction = model.integrate(state)
                retain = retention_loss(prediction, state)
                loss = loss + args.retain_weight * retain
                losses["retain"] = retain
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Non-finite parameter gradient")
            optimizer.step()
            values.append(float(loss))
            if step % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": float(loss),
                                  **{key: float(value) for key, value in losses.items()}}), flush=True)
            if args.max_batches and step + 1 >= args.max_batches:
                break
        report = {"epoch": epoch, "stage": args.stage, "loss": float(np.mean(values)), "batches": len(values),
                  "partial_debug_run": bool(args.max_batches)}
        debug = bool(args.max_batches or args.allow_debug or args.stage == "rgr" and saved and saved.get("partial_debug_run"))
        save_checkpoint(out / "last.pt", model, optimizer, epoch, args, stages, debug=debug)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(out / f"epoch_{epoch + 1:04d}.pt", model, optimizer, epoch, args, stages, debug=debug)
        with (out / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
    ds.close()


def build_teacher(model, args):
    if args.split != "train":
        raise ValueError("Teacher construction is restricted to split=train")
    ds = dataset(args, "train")
    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(ds), size=min(args.samples, len(ds)), replace=False).tolist())
    try:
        return write_cache(model, ds, args.output, indices, args.teacher_epochs,
                           pairs=args.pairs, seed=args.seed, debug=args.allow_debug)
    finally:
        ds.close()


def train_gain(model, saved, args):
    if not args.teacher:
        raise ValueError("train-gain requires --teacher NPZ")
    metadata = json.loads(Path(args.teacher).with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("split") != "train" or metadata.get("generator_sha256") != generator_signature(model):
        raise ValueError("Teacher cache is from the wrong split or a different generator/RGR/configuration")
    ds = dataset(args, "train")
    args.training_profile = ds.training_profile
    ds.close()
    require_training_profile(metadata, args.training_profile, context="Teacher cache")
    if metadata.get("training_mode") != args.training_mode:
        raise ValueError("Teacher cache training mode differs from the requested training mode")
    if args.resume:
        require_training_profile(saved, args.training_profile)
    if metadata.get("partial_debug_run") and not args.allow_debug:
        raise ValueError("Debug teacher cannot be used without --allow-debug")
    archive = np.load(args.teacher, allow_pickle=False)
    args.teacher_sha256 = hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest()
    if args.resume and saved.get("args", {}).get("teacher_sha256") != args.teacher_sha256:
        raise ValueError("Resume cannot silently change the gain-teacher cache")
    features = torch.from_numpy(archive["features"]).float()
    budget = torch.from_numpy(archive["budget"]).long()
    receiver, donor = [torch.from_numpy(archive[key]).long() for key in ["receiver", "donor"]]
    delta = torch.from_numpy(archive["delta"]).long()
    improvement = torch.from_numpy(archive["improvement"]).float()
    loader = DataLoader(torch.utils.data.TensorDataset(features, budget, receiver, donor, delta, improvement),
                        batch_size=args.gain_batch_size, shuffle=True)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.gain.parameters():
        p.requires_grad_(True)
    model.eval()
    optimizer = make_optimizer(torch.optim.Adam, model.gain.parameters(), lr=args.lr, fused=args.fused_optimizer)
    start = 0
    if args.resume:
        load_optimizer_state(optimizer, saved["optimizer"], fused=args.fused_optimizer,
                             disable_fused=getattr(args, "no_fused_optimizer", False))
        start = saved["epoch"] + 1
        torch.set_rng_state(saved["rng"]["torch"].cpu())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        losses = []
        for step, items in enumerate(loader):
            f, m, a, b, delta, improvement = [x.to(args.device) for x in items]
            optimizer.zero_grad(set_to_none=True)
            loss = f.new_zeros(())
            for size in delta.unique():
                chosen = delta == size
                term, _ = gain_loss(model, f[chosen], m[chosen], a[chosen], b[chosen], int(size), improvement[chosen])
                loss = loss + chosen.float().mean() * term
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.gain.parameters() if p.grad is not None):
                raise FloatingPointError("Invalid GainNet gradient")
            optimizer.step()
            losses.append(float(loss))
            if args.max_batches and step + 1 >= args.max_batches:
                break
        stages = list(saved.get("trained_stages", [])) if saved else []
        if "gain" not in stages:
            stages.append("gain")
        args.stage = "gain"
        save_checkpoint(out / "last.pt", model, optimizer, epoch, args, stages,
                        debug=bool(args.allow_debug or args.max_batches))
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(out / f"epoch_{epoch + 1:04d}.pt", model, optimizer, epoch, args, stages,
                            debug=bool(args.allow_debug or args.max_batches))
        with (out / "gain.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "teacher_rows": len(features)}) + "\n")
        print(json.dumps({"gain_epoch": epoch, "loss": float(np.mean(losses)), "teacher_rows": len(features)}), flush=True)
    archive.close()


@torch.no_grad()
def predict(model, saved, args):
    if args.split == "train":
        raise ValueError("Prediction benchmark needs a fixed evaluation split")
    if args.split != "official" and not args.condition:
        raise ValueError("Provide a fixed --condition for the common evaluator")
    model.eval()
    ds = dataset(args, args.split)
    total = min(args.limit, len(ds)) if args.limit else len(ds)
    count = 8192 if args.split in {"pu1k_object", "official"} else 1024
    destination = Path(args.output)
    if destination.suffix != ".npz":
        raise ValueError("Prediction output must be .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".prediction.npy")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype="f4", shape=(total, count, 3))
    ids, rows = [], []
    diagnostic_file = destination.with_suffix(".diagnostics.jsonl")
    try:
        with diagnostic_file.open("w", encoding="utf-8") as handle:
            for i in range(total):
                sample = ds[i]
                x = torch.from_numpy(sample["input_xyz"])[None].to(args.device)
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                state = model.prepare(x, count)
                prediction = model.integrate(state)
                if prediction.shape != (1, count, 3) or not torch.isfinite(prediction).all():
                    raise FloatingPointError("Invalid prediction shape or values")
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                diagnostic = model.diagnostics(state, prediction)
                values[i] = prediction[0].cpu().numpy()
                ids.append(sample["sample_id"])
                row = {"sample_id": sample["sample_id"], "seconds": seconds,
                       **{key: value[0].detach().cpu().tolist() for key, value in diagnostic.items()}}
                handle.write(json.dumps(row) + "\n")
                if i % args.log_every == 0:
                    print(f"Predicted {i + 1}/{total}", flush=True)
        values.flush()
        np.savez(destination, sample_id=np.asarray(ids, dtype=str), prediction_xyz=values)
    finally:
        del values
        temporary.unlink(missing_ok=True)
        if hasattr(ds, "close"):
            ds.close()
    result = {"protocol": "V3", "samples": total, "points": count, "split": args.split, "condition": args.condition,
              "coordinates": "model", "configuration": asdict(model.config), "checkpoint": args.checkpoint,
              "partial_debug_run": bool(args.allow_debug or total < len(ds)), "diagnostics": str(diagnostic_file)}
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def smoke(model, args):
    ds = dataset(args, "train", supervision=True)
    noise_epoch = (3 - ds.source_index(0)) % 5
    sample = ds[(noise_epoch, 0)]  # Keep the noise-family check after patch subsetting.
    x = torch.from_numpy(sample["input_xyz"])[None].to(args.device)
    gt = torch.from_numpy(sample["gt_xyz"])[None].to(args.device)
    supervision = {key: torch.from_numpy(value)[None].to(args.device) for key, value in sample["supervision"].items()}
    model.train()
    state = model.initialize(model.encode(x, for_teacher=True))
    loss = flow_loss(model, state, gt) + recovery_losses(model, state, gt, supervision)["rgr"]
    label = paired_labels(model, state, gt, x.new_tensor([sample["prediction_scale"]]), pairs=1)[0]
    gain, _ = gain_loss(model, state["regions"]["gain_features"], label["budget"], label["receiver"], label["donor"], label["delta"], label["improvement"])
    loss = loss + gain
    loss.backward()
    gradient = {}
    for name in ["flow", "rgr", "gain", "conditioner"]:
        gradient[name] = sum(p.grad.abs().sum().item() for p in getattr(model, name).parameters() if p.grad is not None)
    assert gradient["flow"] > 0 and gradient["gain"] > 0
    if model.config.use_rgr:
        assert gradient["rgr"] > 0
    if model.config.condition:
        assert gradient["conditioner"] > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        y, diagnostic = model(x, return_diagnostics=True)
    assert y.shape == (1, 1024, 3) and torch.isfinite(y).all()
    result = {"status": "passed", "random_weights_smoke_only": True, "loss": float(loss), "gradient_l1": gradient,
              "input_points": 256, "prediction_points": 1024, "teacher_improvement": float(label["improvement"]),
              "configuration": asdict(model.config), "density_TV": diagnostic["density_total_variation"].tolist()}
    if args.smoke_object:
        obj = dataset(args, "pu1k_object")
        xo = torch.from_numpy(obj[0]["input_xyz"])[None].to(args.device)
        with torch.no_grad():
            yo = model(xo, 8192)
        assert yo.shape == (1, 8192, 3) and torch.isfinite(yo).all()
        result["object_prediction_points"] = 8192
        obj.close()
    ds.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@torch.no_grad()
def evaluate_gain(model, args):
    from scipy.stats import spearmanr
    if args.split != "validation":
        raise ValueError("Held-out gain diagnosis uses split=validation, never the training cache")
    model.eval()
    ds = dataset(args, "validation")
    predictions, actual = [], []
    selected, oracle = [], []
    for i in np.linspace(0, len(ds) - 1, min(args.samples, len(ds)), dtype=int):
        sample = ds[int(i)]
        x = torch.from_numpy(sample["input_xyz"])[None].to(args.device)
        gt = torch.from_numpy(sample["gt_xyz"])[None].to(args.device)
        records = paired_labels(model, model.encode(x, for_teacher=True), gt, x.new_tensor([sample["prediction_scale"]]),
                                pairs=args.pairs, seed=stable_seed(args.seed, i))
        local_pred, local_actual = [], []
        for row in records:
            pred = model.gain.transfer(row["features"], row["budget"], row["receiver"], row["donor"], row["delta"])
            local_pred.append(float(pred) / model.config.teacher_gain_scale)
            local_actual.append(float(row["improvement"]))
        predictions.extend(local_pred)
        actual.extend(local_actual)
        chosen = int(np.argmax(local_pred))
        selected.append(local_actual[chosen] if local_pred[chosen] > 0 else 0.)
        oracle.append(max(0., max(local_actual)))
    prediction, actual = np.asarray(predictions), np.asarray(actual)
    valid = np.abs(actual) > 1e-8
    correlation = float(spearmanr(prediction, actual).statistic) if len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(prediction) > 0 else None
    result = {"split": "validation", "frozen_weights": True, "swaps": len(actual), "spearman": correlation,
              "sign_accuracy_excluding_near_zero": float(np.mean((prediction[valid] > 0) == (actual[valid] > 0))) if valid.any() else None,
              "positive_fraction": float(np.mean(actual > 1e-8)), "mean_selected_actual_improvement": float(np.mean(selected)),
              "mean_sampled_oracle_improvement_diagnostic_only": float(np.mean(oracle)),
              "mean_regret_to_sampled_oracle": float(np.mean(np.asarray(oracle) - selected)),
              "prediction": prediction.tolist(), "actual": actual.tolist(), "partial_debug_run": args.allow_debug}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    ds.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "build-teacher", "train-gain", "predict", "smoke", "evaluate-gain"])
    parser.add_argument("--preset", choices=["baseline", "rgr", "rule", "static", "mgb", "rgr_static", "full", "full_condition"])
    parser.add_argument("--backbone", choices=["pufm", "pufm_w_attn"])
    parser.add_argument("--stage", choices=["baseline", "rgr", "joint"], default="baseline")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--split", default="train", choices=["train", "validation", "pu1k_patch", "pu1k_object", "modelnet40_patch", "official"])
    parser.add_argument("--condition")
    parser.add_argument("--training-mode", choices=["mixed", "clean"], default="mixed")
    parser.add_argument("--train-patches-per-object", type=int, default=DEFAULT_TRAIN_PATCHES_PER_OBJECT,
                        help="Fixed shared patches per training object (default: 32; 0: all stored patches)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gain-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--assignment-workers", type=int, default=1,
                        help="Parallel exact CPU matching within a batch; independent of --workers")
    optimizer_options = parser.add_mutually_exclusive_group()
    optimizer_options.add_argument("--fused-optimizer", action="store_true",
                                   help="Use CUDA fused Adam updates (optional floating-point rounding differences)")
    optimizer_options.add_argument("--no-fused-optimizer", action="store_true",
                                   help="Explicitly disable fused updates, including when resuming a fused checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rgr-weight", type=float, default=1.)
    parser.add_argument("--gain-weight", type=float, default=.1)
    parser.add_argument("--retain-weight", type=float, default=0.)
    parser.add_argument("--online-teacher-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--teacher")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-debug", action="store_true")
    parser.add_argument("--smoke-object", action="store_true")
    parser.add_argument("--no-virtual", action="store_true")
    parser.add_argument("--no-reliability", action="store_true")
    parser.add_argument("--no-recovery", action="store_true")
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument("--allocation", choices=["midpoint", "uniform", "density", "curvature", "static", "gain"])
    for key in ["regions", "capacity", "minimum", "flow-steps", "swap-step", "swap-iterations"]:
        parser.add_argument("--" + key, type=int)
    args = parser.parse_args()
    if args.command == "train-gain":
        args.stage = "gain"
    if args.checkpoint and args.resume:
        parser.error("Use either --checkpoint or --resume")
    if min(args.epochs, args.batch_size, args.gain_batch_size, args.online_teacher_every, args.save_every, args.samples, args.pairs) < 1:
        parser.error("Counts and frequencies must be positive")
    if args.assignment_workers < 1:
        parser.error("--assignment-workers must be positive")
    if args.train_patches_per_object < 0:
        parser.error("--train-patches-per-object must be nonnegative")
    configure_assignment_workers(args.assignment_workers)
    seed_all(args.seed)
    model, saved = create_model(args)
    if args.command == "train":
        train(model, saved, args)
        return
    if args.command == "train-gain":
        train_gain(model, saved, args)
        return
    operation = {"build-teacher": build_teacher, "predict": predict, "smoke": smoke, "evaluate-gain": evaluate_gain}[args.command]
    result = operation(model, saved, args) if args.command == "predict" else operation(model, args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
