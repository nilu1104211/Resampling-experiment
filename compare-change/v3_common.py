"""Shared V3 protocol and portable point operations for the comparison copies."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch


def bootstrap():
    configured = os.environ.get("V3_CODE_ROOT")
    code = (Path(configured) if configured else
            next((p for p in Path(__file__).resolve().parents if (p / "adaptive_resampling/v3_dataset.py").is_file()),
                 Path(__file__).resolve().parents[1]))
    if not (code / "adaptive_resampling/v3_dataset.py").is_file():
        raise FileNotFoundError("Set V3_CODE_ROOT to the directory containing adaptive_resampling")
    sys.path.insert(0, str(code))
    return code


bootstrap()
from adaptive_resampling.v3_dataset import DEFAULT_ROOT, AdaptiveResamplingDataset, EpochSampler, OfficialPU1KDataset
from adaptive_resampling.v3_transport import assignment_indices, configure_assignment_workers
from adaptive_resampling.v3_optimizer import make_optimizer, load_optimizer_state
from adaptive_resampling.v3_training import DEFAULT_TRAIN_PATCHES_PER_OBJECT, require_training_profile
from adaptive_resampling.v3_pointops import fps_indices


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def index_points(points, ids):
    """B,N,C and B,... -> B,...,C."""
    batch = torch.arange(len(points), device=points.device)
    return points[batch.reshape((-1,) + (1,) * (ids.ndim - 1)), ids]


def fps(points, count):
    return index_points(points, fps_indices(points, count))


def knn(points, centers, k):
    if not 0 < k <= points.shape[1]:
        raise ValueError("KNN count exceeds available observations")
    return torch.cdist(centers, points).topk(k, dim=-1, largest=False, sorted=True).indices


def midpoint(points, rate=4):
    """Original KNN midpoint pool + FPS, always based on the observed input."""
    neighbors = index_points(points, knn(points, points, min(2 * rate, points.shape[1])))
    candidates = ((points[:, :, None, :] + neighbors) * .5).flatten(1, 2)
    return fps(candidates, points.shape[1] * rate)


def chamfer(prediction, target):
    distance = torch.cdist(prediction, target).square()
    return (distance.min(-1).values.mean() + distance.min(-2).values.mean()) * .5


def assignment_target(source, target):
    """Exact equal-cardinality transport; GT is used only inside training losses."""
    return index_points(target, assignment_indices(source, target))


def transport_loss(prediction, target):
    return (prediction - assignment_target(prediction, target)).square().sum(-1).mean()


def validate_prediction(prediction, batch, count):
    if prediction.shape != (batch, count, 3) or not torch.isfinite(prediction).all():
        raise ValueError(f"Expected finite {(batch, count, 3)} prediction, got {prediction.shape}")
    return prediction


@torch.no_grad()
def patch_predict(points, output_count, predictor, *, patch_points=256, patch_rate=3, patch_batch_size=1):
    """GT-free inference: normalize input patches, restore coordinates, merge by FPS."""
    if points.shape[1] * 4 != output_count:
        raise ValueError("V3 comparison adapters implement 4x upsampling")
    if patch_batch_size < 1:
        raise ValueError("patch_batch_size must be positive")
    if points.shape[1] == patch_points:
        return validate_prediction(predictor(points), len(points), output_count)
    if points.shape[1] < patch_points:
        raise ValueError("A prediction patch needs at least 256 observed points")
    outputs = []
    for cloud in points.split(1):
        centers = fps(cloud, math.ceil(cloud.shape[1] / patch_points * patch_rate))
        ids = knn(cloud, centers, patch_points)
        patches = index_points(cloud, ids).squeeze(0)
        center = patches.mean(1, keepdim=True)
        scale = (patches - center).norm(dim=-1).amax(1).reshape(-1, 1, 1).clamp_min(1e-8)
        predictions = []
        normalized = (patches - center) / scale
        for i in range(0, len(patches), patch_batch_size):
            batch = normalized[i:i + patch_batch_size]
            pred = predictor(batch)
            validate_prediction(pred, len(batch), patch_points * 4)
            predictions.append(pred * scale[i:i + len(batch)] + center[i:i + len(batch)])
        merged = torch.cat(predictions, 0).reshape(1, -1, 3)
        outputs.append(fps(merged, output_count))
    return validate_prediction(torch.cat(outputs), len(points), output_count)


def dataset(args, split, *, supervision=False):
    if split == "official":
        return OfficialPU1KDataset(args.root)
    return AdaptiveResamplingDataset(
        args.root, split, condition=None if split == "train" else args.condition,
        training_mode=args.training_mode, include_supervision=supervision,
        augment_rotation=split == "train",
        train_patches_per_object=getattr(args, "train_patches_per_object", DEFAULT_TRAIN_PATCHES_PER_OBJECT),
        training_input_points=(1024 if split == "train" and getattr(args, "model_name", None) == "spu-imr"
                               and getattr(args, "spu_profile", None) == "native" else None),
    )


def load_adapter(folder, args):
    spec = importlib.util.spec_from_file_location("comparison_v3_adapter", Path(folder) / "v3_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter(args).to(args.device)


def load_weights(adapter, path, args, *, resume=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if args.model_name == "spu-imr" and checkpoint.get("protocol") == "V3":
        saved_args = {"spu_profile": "legacy-patch256", "spu_transport": "hungarian", **saved_args}
    if saved_args.get("model_name") not in {None, args.model_name}:
        raise ValueError("Checkpoint belongs to a different comparison network")
    for name in adapter.compatibility_keys:
        if name in saved_args and saved_args[name] != getattr(args, name):
            raise ValueError(f"Checkpoint {name}={saved_args[name]!r} differs from --{name.replace('_', '-')}={getattr(args, name)!r}")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint.get("model_dict", checkpoint)))
    # Accept native model weights without silently ignoring missing parameters.
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if set(state) == set(adapter.state_dict()):
        adapter.load_state_dict(state, strict=True)
    else:
        state = {k.removeprefix("model."): v for k, v in state.items()}
        adapter.model.load_state_dict(state, strict=True)
    return checkpoint


def tensor_batch(batch, device):
    # No supervision/reference/clean-input fields are ever passed to the network.
    return batch["input_xyz"].to(device).float(), batch["gt_xyz"].to(device).float()


def check_data(args):
    records = []
    for split, expected in [("train", (256, 1024)), ("validation", (256, 1024)),
                             ("pu1k_patch", (256, 1024)), ("modelnet40_patch", (256, 1024)),
                             ("pu1k_object", (2048, 8192)), ("official", (2048, 8192))]:
        ds = dataset(args, split)
        s = ds[0]
        n, m = expected
        assert s["input_xyz"].shape == (n, 3) and s["gt_xyz"].shape == (m, 3)
        assert np.isfinite(s["input_xyz"]).all() and np.isfinite(s["gt_xyz"]).all()
        assert "supervision" not in s
        records.append({"split": split, "samples": len(ds), "input": n, "output": m})
        if hasattr(ds, "close"):
            ds.close()
    return {"status": "passed", "model": args.model_name, "datasets": records}


def train(adapter, args):
    from torch.utils.data import DataLoader
    ds = dataset(args, "train")
    args.training_profile = ds.training_profile
    print(json.dumps({"training_profile": ds.training_profile}), flush=True)
    sampler = EpochSampler(ds)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        persistent_workers=args.workers > 0, drop_last=False)
    optimizer = make_optimizer(torch.optim.AdamW, adapter.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay, fused=args.fused_optimizer)
    scheduler = adapter.make_scheduler(optimizer, args) if hasattr(adapter, "make_scheduler") else None
    start = 0
    if args.resume:
        saved = load_weights(adapter, args.resume, args, resume=True)
        if saved.get("protocol") != "V3":
            raise ValueError("Resume requires a V3 training checkpoint")
        require_training_profile(saved, ds.training_profile)
        for key in ["training_mode"] + (["stage"] if args.model_name == "pufm++" else []):
            if saved.get("args", {}).get(key) != getattr(args, key):
                raise ValueError(f"Resume cannot change {key}; use --checkpoint for a new fine-tuning stage")
        load_optimizer_state(optimizer, saved["optimizer"], fused=args.fused_optimizer,
                             disable_fused=getattr(args, "no_fused_optimizer", False))
        if scheduler is not None:
            if saved.get("scheduler") is None:
                raise ValueError("Resume requires the saved native SPU scheduler; use a new stage for a changed recipe")
            for key in ("spu_schedule_epochs", "spu_warmup_epochs"):
                if saved.get("args", {}).get(key) != getattr(args, key):
                    raise ValueError(f"Resume cannot change {key}")
            scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"] + 1
        if "rng" in saved:
            random.setstate(saved["rng"]["python"])
            np.random.set_state(saved["rng"]["numpy"])
            torch.set_rng_state(saved["rng"]["torch"].cpu())
            if torch.cuda.is_available() and saved["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in saved["rng"]["cuda"]])
    elif args.checkpoint:
        load_weights(adapter, args.checkpoint, args)
    print(json.dumps({"runtime": {"start_epoch_zero_based": start, "loader_workers": args.workers,
                                  "assignment_workers": getattr(args, "assignment_workers", 1),
                                  "fused_optimizer": any(bool(g.get("fused")) for g in optimizer.param_groups)}}), flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, args.epochs):
        sampler.set_epoch(epoch)
        adapter.train()
        losses = []
        for step, batch in enumerate(loader):
            points, gt = tensor_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = adapter.training_loss(points, gt)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % args.log_every == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": loss.item()}), flush=True)
            if args.max_batches and step + 1 >= args.max_batches:
                break
        result = {"epoch": epoch, "loss": float(np.mean(losses)), "batches": len(losses),
                  "partial_debug_run": bool(args.max_batches)}
        if args.validation_samples:
            adapter.eval()
            val = dataset(args, "validation")
            scores = []
            with torch.no_grad():
                for i in range(min(args.validation_samples, len(val))):
                    s = val[i]
                    x = torch.from_numpy(s["input_xyz"])[None].to(args.device)
                    g = torch.from_numpy(s["gt_xyz"])[None].to(args.device)
                    scores.append(chamfer(adapter.predict(x, 1024), g).item())
            val.close()
            result["validation_cd_model_coordinates"] = float(np.mean(scores))
        if scheduler is not None:
            scheduler.step(epoch)
        torch.save({"protocol": "V3", "training_profile": ds.training_profile,
                    "model": adapter.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch, "args": vars(args), "training": adapter.training_description,
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                    "result": result}, out / "last.pt")
        with (out / "train.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    ds.close()


def predict(adapter, args):
    if not args.checkpoint:
        raise ValueError("predict requires --checkpoint; random weights are only allowed by smoke")
    load_weights(adapter, args.checkpoint, args)
    adapter.eval()
    ds = dataset(args, args.split)
    count = 8192 if args.split in {"pu1k_object", "official"} else 1024
    total = min(args.limit, len(ds)) if args.limit else len(ds)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids, diagnostics = [], []
    # Bounded RAM even for all 41,956 ModelNet40 records.
    temporary = out.with_suffix(".prediction.npy")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype="f4", shape=(total, count, 3))
    try:
        with torch.no_grad():
            for i in range(total):
                s = ds[i]
                x = torch.from_numpy(s["input_xyz"])[None].to(args.device)
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                y = validate_prediction(adapter.predict(x, count), 1, count)
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                values[i] = y[0].cpu().numpy()
                ids.append(s["sample_id"])
                diagnostics.append({"sample_id": s["sample_id"], "seconds": time.perf_counter() - start,
                                    **getattr(adapter, "last_diagnostics", {})})
                if i % args.log_every == 0:
                    print(f"Predicted {i + 1}/{total}", flush=True)
        values.flush()
        np.savez(out, sample_id=np.asarray(ids, dtype=str), prediction_xyz=values)
    finally:
        del values
        temporary.unlink(missing_ok=True)
        if hasattr(ds, "close"):
            ds.close()
    out.with_suffix(".json").write_text(json.dumps({
        "protocol": "V3", "coordinates": "model", "split": args.split, "condition": args.condition,
        "partial_debug_run": total < len(ds), "args": vars(args), "training": adapter.training_description,
        "inference": adapter.inference_description, "samples": diagnostics,
    }, indent=2), encoding="utf-8")
    return {"status": "passed", "samples": total, "output_points": count, "predictions": str(out)}


def smoke(adapter, args):
    ds = dataset(args, "train")
    s = ds[0]
    x = torch.from_numpy(s["input_xyz"])[None].to(args.device)
    gt = torch.from_numpy(s["gt_xyz"])[None].to(args.device)
    optimizer = make_optimizer(torch.optim.Adam, adapter.parameters(), lr=args.lr, fused=args.fused_optimizer)
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    loss = adapter.training_loss(x, gt)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    optimizer.step()
    adapter.eval()
    with torch.no_grad():
        count = x.shape[1] * 4
        y = validate_prediction(adapter.predict(x, count), 1, count)
    report = {"status": "passed", "model": args.model_name, "device": args.device,
              "loss": loss.item(), "gradient_parameters": len(gradients),
              "input_points": x.shape[1], "prediction_points": y.shape[1],
              "training": adapter.training_description, "inference": adapter.inference_description,
              "random_weights_smoke_only": True}
    if args.smoke_object:
        obj = dataset(args, "pu1k_object")
        xo = torch.from_numpy(obj[0]["input_xyz"])[None].to(args.device)
        with torch.no_grad():
            yo = validate_prediction(adapter.predict(xo, 8192), 1, 8192)
        report["object_prediction_points"] = yo.shape[1]
        obj.close()
    ds.close()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(folder, model_name):
    parser = argparse.ArgumentParser(description=f"{model_name}: V3 train/predict/check/smoke")
    parser.add_argument("command", choices=["check-data", "smoke", "train", "predict", "schedule"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--training-mode", choices=["mixed", "clean"], default="mixed")
    parser.add_argument("--train-patches-per-object", type=int, default=DEFAULT_TRAIN_PATCHES_PER_OBJECT,
                        help="Fixed shared patches per training object (default: 32; 0: all stored patches)")
    parser.add_argument("--split", choices=["validation", "pu1k_patch", "modelnet40_patch", "pu1k_object", "official"], default="pu1k_patch")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=1 if model_name == "pufm++" else 2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--assignment-workers", type=int, default=1,
                        help="Parallel exact CPU matching within a batch; independent of --workers")
    optimizer_options = parser.add_mutually_exclusive_group()
    optimizer_options.add_argument("--fused-optimizer", action="store_true",
                                   help="Use CUDA fused AdamW/Adam updates (optional floating-point rounding differences)")
    optimizer_options.add_argument("--no-fused-optimizer", action="store_true",
                                   help="Explicitly disable fused updates, including when resuming a fused checkpoint")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4 if model_name == "pufm++" else 1e-3 if model_name == "spu-imr" else 5e-4)
    parser.add_argument("--weight-decay", type=float, default=0 if model_name == "pufm++" else .05)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output", default=str(Path(folder) / "v3_runs" / "smoke.json"))
    parser.add_argument("--max-batches", type=int, default=0, help="Debug only; zero processes the full epoch")
    parser.add_argument("--validation-samples", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="Prediction subset for debugging")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--smoke-object", action="store_true")
    parser.add_argument("--stage", choices=["s1", "s2"], default="s1")
    parser.add_argument("--flow-steps", type=int, default=6)
    parser.add_argument("--schedule")
    parser.add_argument("--uniform-schedule", action="store_true", help="PUFM++ uniform-time ablation; formal default uses a calibrated schedule")
    parser.add_argument("--schedule-batches", type=int, default=0)
    parser.add_argument("--schedule-grid", type=int, default=51)
    parser.add_argument("--manifold-step", type=float, default=.01)
    parser.add_argument("--fps-backend", choices=["auto", "cuda", "torch"], default="auto")
    parser.add_argument("--patch-batch-size", type=int, default=1,
                        help="PUFM++ patch batch size; 1 preserves the original V3 arithmetic path")
    parser.add_argument("--spu-passes", type=int, default=8)
    parser.add_argument("--spu-profile", choices=["native", "legacy-patch256"], default="native")
    parser.add_argument("--spu-groups", type=int, help="Native: 256; legacy patch256: 64")
    parser.add_argument("--spu-transport", choices=["source", "hungarian"],
                        help="Native defaults to source EMD; Hungarian is an explicit backend adaptation")
    parser.add_argument("--spu-schedule-epochs", type=int, default=6000)
    parser.add_argument("--spu-warmup-epochs", type=int, default=50)
    parser.add_argument("--octree-depth", type=int, default=8)
    parser.add_argument("--octree-scale", type=float, default=.5)
    parser.add_argument("--octree-beam", type=int, default=4)
    args = parser.parse_args()
    args.model_name = model_name
    os.environ["V3_FPS_BACKEND"] = args.fps_backend
    if args.patch_batch_size < 1 or args.spu_schedule_epochs < 1 or not 0 <= args.spu_warmup_epochs < args.spu_schedule_epochs:
        parser.error("Invalid patch batch size or SPU schedule/warmup horizon")
    if args.command == "predict" and args.split != "official" and not args.condition:
        parser.error("predict requires one --condition so its NPZ matches the common evaluator")
    if args.command == "train" and args.output.endswith(".json"):
        args.output = str(Path(folder) / "v3_runs" / "train")
    if args.command == "predict" and not args.output.endswith(".npz"):
        parser.error("predict --output must end with .npz")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.flow_steps < 1 or args.spu_passes < 1:
        parser.error("Epochs, batch size, inference steps and passes must be positive")
    if args.assignment_workers < 1:
        parser.error("--assignment-workers must be positive")
    if args.train_patches_per_object < 0:
        parser.error("--train-patches-per-object must be nonnegative")
    configure_assignment_workers(args.assignment_workers)
    set_seed(args.seed)
    if args.command == "check-data":
        result = check_data(args)
    else:
        adapter = load_adapter(folder, args)
        if args.command == "train":
            train(adapter, args)
            return
        if args.command == "schedule":
            if not hasattr(adapter, "calibrate_schedule") or not args.checkpoint:
                parser.error("schedule is supported by PUFM++ and requires --checkpoint")
            load_weights(adapter, args.checkpoint, args)
            result = adapter.calibrate_schedule(dataset(args, "train"), args)
        elif args.command == "predict":
            result = predict(adapter, args)
        else:
            result = smoke(adapter, args)
    print(json.dumps(result, indent=2), flush=True)
