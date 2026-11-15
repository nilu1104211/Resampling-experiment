"""Shared staged protocol executed against each independent PUFM source copy."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "our" / "PUFM-main"))
from baseline_validation import BaselineValidation, ModelNet40HoldoutDataset, atomic_save, digest_file, validation_state
from pu1k_sparse_dataset import PU1KSparseDataset
from adaptive_resampling.v3_dataset import EpochSampler, OfficialPU1KDataset
from adaptive_resampling.v3_transport import configure_assignment_workers
from models.adaptive_ops import aligned_target, cd_l1
from geometry_m2.single_m2 import Config, SingleM2PUFM
from geometry_m2.geometry_features import local_regions, geometry_feature
from geometry_m2.allocation import allocate, initialize
from geometry_m2.teacher import build_teacher, read_teacher, generator_signature, code_signature, write_manifest


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@contextmanager
def output_lock(output):
    """OS releases the lock on process exit, including interrupted jobs."""
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    with (path/".writer.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def create_model(args, mode):
    saved = read_checkpoint(args.resume or args.checkpoint)
    own = saved.get("protocol") == "PUFM-single-M2-v1"
    if own and saved.get("stage") in {"gain", "adapt"} and saved["configuration"]["mode"] != mode:
        raise ValueError("Use the trained checkpoint's D1/D2/D3 entry point")
    config = Config(**saved["configuration"]) if own else Config(backbone=saved.get("configuration", {}).get("backbone", "pufm_w_attn"))
    config.mode = mode
    if args.flow_steps is not None:
        config.flow_steps = args.flow_steps
    model = SingleM2PUFM(config).to(args.device)
    state = saved.get("model", saved.get("state_dict", saved))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if own:
        model.load_state_dict(state, strict=True)
    elif any(key.startswith("flow.") for key in state):
        if saved.get("configuration", {}).get("use_rgr", False):
            raise ValueError("The common starting checkpoint must be the clean PUFM baseline")
        model.flow.load_state_dict({key[5:]: value for key, value in state.items() if key.startswith("flow.")}, strict=True)
    else:
        model.flow.load_state_dict(state, strict=True)
    if saved.get("partial_debug_run") and not (args.max_batches or args.validation_limit):
        raise ValueError("Debug checkpoints require a debug run")
    if args.resume and (not own or saved["stage"] != args.stage or saved["configuration"] != asdict(config)):
        raise ValueError("Resume stage and model configuration must match")
    if args.allocator and not args.resume:
        allocator = read_checkpoint(args.allocator)
        if allocator["configuration"]["mode"] != mode or allocator["stage"] != "gain":
            raise ValueError("Expected this variant's trained M2 checkpoint")
        if allocator["common_generator_sha256"] != generator_signature(model):
            raise ValueError("S5 must start from the same S2 generator used by S4")
        model.m2.load_state_dict({key[3:]: value for key, value in allocator["model"].items() if key.startswith("m2.")})
    return model, saved


def gain_objective(model, region, record, pair, device, rank_weight):
    features, _, valid = model.m2(region, model.config.mode, record["input"].shape[1])
    budget = record["budget"].to(device)
    receiver = record["receiver"][:, pair].to(device)
    donor = record["donor"][:, pair].to(device)
    predicted = model.m2.gain.transfer(features, budget, receiver, donor, record["delta"])
    target = record["improvement"][:, pair].to(device) / model.m2.gain_scale
    batch = torch.arange(len(predicted), device=device)
    usable = valid[batch, receiver] & valid[batch, donor]
    regression = F.smooth_l1_loss(predicted[usable], target[usable]) if usable.any() else predicted.sum()*0
    directional = usable & (target.abs() > model.m2.noise_threshold / model.m2.gain_scale)
    ranking = F.softplus(-target[directional].sign()*predicted[directional]).mean() if directional.any() else predicted.sum()*0
    return regression + rank_weight*ranking, predicted[usable], target[usable]


def apply_step(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite objective")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
    optimizer.step()


def flow_epoch(model, loader, sampler, optimizer, parameters, epoch, args):
    sampler.set_epoch(epoch)
    model.train()
    model.m2.eval()
    frozen = epoch < args.freeze_epochs
    model.flow.requires_grad_(not frozen)
    if frozen:
        model.flow.eval()
    losses = []
    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        points, target = batch["input_xyz"].to(args.device), batch["gt_xyz"].to(args.device)
        state = model.encode(points, with_m2=args.stage == "adapt")
        method = "gain" if args.stage == "adapt" and step % 2 == 0 else ("uniform", "density", "curvature", "random")[(step//2) % 4]
        state = initialize(state, allocate(model, state, method), model.config.minimum)
        x0 = state["initial"]
        x1 = aligned_target(x0, target)
        t = 1-torch.cos(torch.rand(len(points), device=args.device)*torch.pi/2)
        query = (1-t[:, None, None])*x0 + t[:, None, None]*x1
        loss = F.mse_loss(model.velocity(query, t, state), x1-x0)
        apply_step(loss, optimizer, parameters)
        losses.append(float(loss.detach()))
        if step % args.log_every == 0:
            print(json.dumps(dict(stage=args.stage, epoch=epoch+1, batch=step, loss=losses[-1], backbone_frozen=frozen)), flush=True)
        if args.max_batches and step+1 >= args.max_batches:
            break
    return dict(loss=float(np.mean(losses)), batches=len(losses), backbone_frozen=frozen)


def gain_epoch(model, manifest, optimizer, parameters, epoch, args):
    model.eval()
    model.m2.train()
    losses, predictions, targets = [], [], []
    order = np.random.default_rng(args.seed+epoch).permutation(len(manifest["shards"]))
    for step, index in enumerate(order):
        record = read_checkpoint(Path(args.teacher) / manifest["shards"][index]["file"])
        region = local_regions(record["input"].to(args.device), model.config.regions)
        if not torch.allclose(region["centers"].cpu(), record["centers"], atol=1e-6, rtol=1e-6):
            raise ValueError("Teacher region layout changed")
        optimizer.zero_grad(set_to_none=True)
        objectives = []
        for pair in range(record["receiver"].shape[1]):
            loss, prediction, target = gain_objective(model, region, record, pair, args.device, args.rank_weight)
            objectives.append(loss)
            predictions.extend(prediction.detach().cpu().tolist())
            targets.extend(target.cpu().tolist())
        loss = torch.stack(objectives).mean()
        apply_step(loss, optimizer, parameters)
        losses.append(float(loss.detach()))
        if step % args.log_every == 0:
            print(json.dumps(dict(stage="gain", epoch=epoch+1, shard=step, loss=losses[-1])), flush=True)
        if args.max_batches and step+1 >= args.max_batches:
            break
    p, t = np.asarray(predictions), np.asarray(targets)
    if not len(t):
        raise ValueError("No valid geometry pairs contributed to this gain epoch")
    usable = np.abs(t) > float(model.m2.noise_threshold/model.m2.gain_scale)
    correlation = float(spearmanr(p, t).statistic) if len(p)>1 and np.std(p)>0 and np.std(t)>0 else None
    return dict(loss=float(np.mean(losses)), batches=len(losses), train_teacher_spearman=correlation,
                train_teacher_direction_accuracy=float(np.mean(np.sign(p[usable])==np.sign(t[usable]))) if usable.any() else None,
                usable_teacher_pairs=len(t))


def capture_rng():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(rng):
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def train(model, saved, args, dataset):
    if args.stage in {"gain", "adapt"} and saved.get("stage") != "preadapt" and not args.resume:
        raise ValueError("S4 requires the common S2 preadapt checkpoint")
    if args.stage == "adapt" and not args.allocator and not args.resume:
        raise ValueError("S5 requires --allocator from this variant's S4 best.pt")
    manifest = None
    common_signature = saved.get("common_generator_sha256") if args.resume else generator_signature(model)
    if args.stage == "gain":
        manifest = read_teacher(args.teacher, model, args.source_sha256)
        stats = manifest["statistics"]
        model.m2.geometry_center.fill_(stats["geometry"][model.config.mode]["center"])
        model.m2.geometry_scale.fill_(stats["geometry"][model.config.mode]["scale"])
        model.m2.gain_scale.fill_(stats["gain_scale"])
        model.m2.noise_threshold.fill_(stats["noise_threshold"])
    model.requires_grad_(False)
    modules = [model.m2] if args.stage == "gain" else [model.local_encoder, model.conditioner, model.flow]
    for module in modules:
        module.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.stage == "gain":
        groups = [dict(params=parameters, lr=args.lr)]
    else:
        groups = [dict(params=list(model.local_encoder.parameters())+list(model.conditioner.parameters()), lr=args.lr),
                  dict(params=list(model.flow.parameters()), lr=args.flow_lr)]
    optimizer = torch.optim.Adam(groups)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(source_sha256=args.source_sha256, code_sha256=code_signature(), seed=args.seed,
                    stage=args.stage, lr=args.lr, flow_lr=args.flow_lr, freeze_epochs=args.freeze_epochs,
                    rank_weight=args.rank_weight, batch_size=args.batch_size,
                    max_batches=args.max_batches, validation_limit=args.validation_limit,
                    checkpoint_sha256=digest_file(args.checkpoint),
                    allocator_sha256=digest_file(args.allocator) if args.allocator else None,
                    teacher_manifest_sha256=digest_file(Path(args.teacher)/"manifest.json") if args.teacher else None)
    if args.resume and saved["run_identity"] != identity:
        raise ValueError("Resume data, source weights, code or training settings changed")
    selection = BaselineValidation(args, saved if args.resume else None)
    start = saved["epoch"]+1 if args.resume else 0
    sampler = EpochSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.workers,
                        persistent_workers=args.workers>0)
    if args.resume:
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
    for epoch in range(start, args.epochs):
        begin = time.monotonic()
        report = gain_epoch(model, manifest, optimizer, parameters, epoch, args) if args.stage == "gain" else flow_epoch(model, loader, sampler, optimizer, parameters, epoch, args)
        model.allocation = "gain" if args.stage in {"gain", "adapt"} else "uniform"
        validation, improved = selection.evaluate(model, epoch)
        report.update(epoch=epoch, epoch_one_based=epoch+1, stage=args.stage, mode=model.config.mode,
                      validation=validation, best_validation=selection.best, seconds=time.monotonic()-begin)
        payload = dict(protocol="PUFM-single-M2-v1", configuration=asdict(model.config), model=model.state_dict(),
                       optimizer=optimizer.state_dict(), epoch=epoch, stage=args.stage, rng=capture_rng(),
                       run_identity=identity, common_generator_sha256=common_signature,
                       best_validation=selection.best, training_profile=dataset.training_profile,
                       partial_debug_run=bool(args.max_batches or args.validation_limit), args=vars(args))
        atomic_save(payload, output/"last.pt")
        if improved:
            atomic_save(payload, output/"best.pt")
        with (output/"train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report)+"\n")
        print(json.dumps(report), flush=True)
    return output/"best.pt"


@torch.no_grad()
def screen(model, dataset, args):
    # Rule normalization uses training inputs only, before any validation call.
    values = []
    for index in range(min(256, len(dataset))):
        points = torch.from_numpy(dataset[(0, index)]["input_xyz"])[None].to(args.device)
        value, valid = geometry_feature(local_regions(points, model.config.regions), model.config.mode)
        values.append(value[valid])
    values = torch.cat(values)
    if not len(values):
        raise ValueError("No valid training geometry for rule screening")
    model.m2.geometry_center.copy_(values.median())
    model.m2.geometry_scale.copy_((values.quantile(.9)-values.quantile(.1)).clamp_min(1e-8))
    selection = BaselineValidation(args)
    result = {}
    for method in ("midpoint", "uniform", "density", "curvature", "rule"):
        model.allocation = method
        result[method] = selection.evaluate(model, 0)[0]
    write_manifest(Path(args.output)/"screen.json", result)


@torch.no_grad()
def drift(model, dataset, args):
    manifest = json.loads((Path(args.teacher)/"manifest.json").read_text())
    record = read_checkpoint(Path(args.teacher)/manifest["shards"][0]["file"])
    model.eval()
    points = record["input"].to(args.device)
    target = torch.from_numpy(np.stack([dataset[(0, row)]["gt_xyz"] for row in record["rows"]])).to(args.device)
    state = model.encode(points, with_m2=False)
    budget = record["budget"].to(args.device)
    before = cd_l1(model.integrate(initialize(state, budget, model.config.minimum)), target)
    changed = budget.clone()
    batches = torch.arange(len(points), device=args.device)
    changed[batches, record["receiver"][:, 0].to(args.device)] += record["delta"]
    changed[batches, record["donor"][:, 0].to(args.device)] -= record["delta"]
    gains = before-cd_l1(model.integrate(initialize(state, changed, model.config.minimum)), target)
    old = record["improvement"][:, 0]
    current = gains.cpu()
    usable = old.abs()>manifest["statistics"]["noise_threshold"]
    result = dict(samples=len(old), scope="first training teacher shard, diagnostic only",
                  old_generator_sha256=manifest["protocol"]["generator_sha256"],
                  new_generator_sha256=generator_signature(model),
                  mean_absolute_gain_change=float((current-old).abs().mean()),
                  sign_agreement=float((current[usable].sign()==old[usable].sign()).float().mean()) if usable.any() else None)
    write_manifest(Path(args.output)/"teacher_drift.json", result)


@torch.no_grad()
def predict(model, saved, args):
    model.eval()
    model.allocation = args.allocation or ("gain" if saved["stage"] in {"gain", "adapt"} else "uniform")
    dataset = OfficialPU1KDataset(args.root) if args.split == "official" else ModelNet40HoldoutDataset(args.root, "A0_normal", objects_per_class=args.modelnet40_objects_per_class or None)
    destination = Path(args.output)
    if destination.suffix != ".npz":
        raise ValueError("Prediction output must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 8192 if args.split == "official" else 1024
    size = min(args.limit or len(dataset), len(dataset))
    temporary = destination.with_suffix(".prediction.npy")
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype="f4", shape=(size, count, 3))
    ids = []
    try:
        with destination.with_suffix(".diagnostics.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(size):
                sample = dataset[index]
                points = torch.from_numpy(sample["input_xyz"])[None].to(args.device)
                state = model.prepare(points, count)
                prediction = model.integrate(state)
                if prediction.shape != (1, count, 3) or not torch.isfinite(prediction).all():
                    raise FloatingPointError("Invalid prediction")
                values[index] = prediction[0].cpu().numpy()
                ids.append(sample["sample_id"])
                handle.write(json.dumps(dict(sample_id=ids[-1], **{k:v[0].cpu().tolist() for k,v in model.diagnostics(state, prediction).items()}))+"\n")
                if index % args.log_every == 0:
                    print(f"Predicted {index+1}/{size}", flush=True)
        values.flush()
        np.savez(destination, sample_id=np.asarray(ids), prediction_xyz=values)
    finally:
        del values
        temporary.unlink(missing_ok=True)
        dataset.close()
    write_manifest(destination.with_suffix(".json"), dict(protocol="V3", coordinates="model", split=args.split,
                   condition="official_A0" if args.split=="official" else "A0_normal", samples=size, points=count,
                   modelnet40_objects_per_class=args.modelnet40_objects_per_class or None,
                   checkpoint_sha256=digest_file(args.checkpoint), configuration=asdict(model.config),
                   partial_debug_run=bool(saved.get("partial_debug_run") or args.limit)))


def main(mode):
    parser = argparse.ArgumentParser(description="Independent geometry M2 experiment")
    parser.add_argument("command", choices=("train", "teacher", "screen", "drift", "predict"))
    parser.add_argument("--stage", choices=("preadapt", "gain", "adapt"), default="preadapt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--allocator")
    parser.add_argument("--teacher")
    parser.add_argument("--root", required=True)
    parser.add_argument("--pu1k-h5")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--assignment-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--flow-lr", type=float, default=1e-5)
    parser.add_argument("--rank-weight", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--flow-steps", type=int)
    parser.add_argument("--teacher-samples", type=int, default=4096, help="0 uses all HDF5 rows")
    parser.add_argument("--teacher-pairs", type=int, default=4)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--split", choices=("official", "modelnet40_holdout"), default="modelnet40_holdout")
    parser.add_argument("--modelnet40-objects-per-class", type=int, default=5, help="0 selects all 2368 holdout objects")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allocation", choices=("uniform", "density", "curvature", "rule", "gain"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.teacher_pairs, args.teacher_batch_size, args.log_every) < 1:
        parser.error("Epochs, batch sizes, pairs and log interval must be positive")
    if min(args.freeze_epochs, args.teacher_samples, args.max_batches, args.validation_limit, args.workers, args.limit) < 0:
        parser.error("Counts cannot be negative")
    if args.command != "predict" and not args.pu1k_h5:
        parser.error("Training, screening and teachers require --pu1k-h5")
    if (args.stage=="gain" and args.command=="train" or args.command=="drift") and not args.teacher:
        parser.error("This stage requires --teacher")
    seed_all(args.seed)
    configure_assignment_workers(args.assignment_workers)
    model, saved = create_model(args, mode)
    if args.command == "predict":
        predict(model, saved, args)
        return
    Path(args.output).mkdir(parents=True, exist_ok=True)
    args.source_sha256 = digest_file(args.pu1k_h5)
    source = saved.get("args", {}).get("pu1k_sha256") or saved.get("run_identity", {}).get("source_sha256")
    if source and source != args.source_sha256:
        raise ValueError("Training HDF5 differs from the checkpoint source")
    dataset = PU1KSparseDataset(args.pu1k_h5, "train", source_sha256=args.source_sha256)
    try:
        with output_lock(args.output):
            if args.command == "train":
                train(model, saved, args, dataset)
            elif args.command == "teacher":
                if saved.get("stage") != "preadapt":
                    raise ValueError("Shared teacher requires common S2 preadapt weights")
                build_teacher(model, dataset, args)
            elif args.command == "screen":
                screen(model, dataset, args)
            else:
                drift(model, dataset, args)
    finally:
        dataset.close()
