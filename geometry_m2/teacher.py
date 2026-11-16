"""Resumable train-only counterfactual shards shared by all geometry variants."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from models.adaptive_ops import cd_l1, region_counts
from geometry_m2.allocation import CANDIDATE_POLICY, allocate, initialize
from geometry_m2.geometry_features import geometry_feature
from baseline_validation import atomic_save, digest_file


def generator_signature(model):
    config = asdict(model.config)
    for key in ("mode", "exchanges", "gain_threshold"):
        config.pop(key)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for key, value in sorted(model.state_dict().items()):
        if not key.startswith("m2."):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def code_signature():
    import models
    files = sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(models.__file__).parent.glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        if path.name != "__init__.py":
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@torch.no_grad()
def build_teacher(model, dataset, args):
    if dataset.split != "train":
        raise ValueError("Only original training HDF5 may provide gain labels")
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    count = min(args.teacher_samples or len(dataset), len(dataset))
    indices = np.random.default_rng(args.seed).permutation(len(dataset))[:count].tolist()
    protocol = dict(version=1, split="train", source_sha256=args.source_sha256,
                    generator_sha256=generator_signature(model), code_sha256=code_signature(),
                    config={k: v for k, v in asdict(model.config).items() if k != "mode"},
                    candidate_policy=CANDIDATE_POLICY,
                    metric="0.5 * (mean pred-to-GT L2 + mean GT-to-pred L2), input-normalized",
                    indices=indices, augmentation_epoch=0, pairs=args.teacher_pairs,
                    shard_size=args.teacher_batch_size, seed=args.seed,
                    torch=torch.__version__, fps_backend=os.environ.get("V3_FPS_BACKEND", "auto"),
                    pointops=os.environ.get("PUFM_POINTOPS", "auto"), device=args.device,
                    partial_debug_run=bool(args.max_batches or args.validation_limit))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["protocol"] != protocol:
            raise ValueError("Teacher cache identity changed. Choose a new teacher directory.")
    else:
        write_manifest(manifest_path, dict(protocol=protocol, complete=False, shards=[]))
    shards = []
    for start in range(0, count, args.teacher_batch_size):
        destination = output / f"shard_{start:06d}.pt"
        rows = indices[start:start+args.teacher_batch_size]
        if destination.exists():
            existing = torch.load(destination, map_location="cpu", weights_only=False)
            if existing["generator_sha256"] != protocol["generator_sha256"] or existing["rows"] != rows:
                raise ValueError("Teacher shard identity mismatch")
        else:
            samples = [dataset[(0, row)] for row in rows]
            points = torch.from_numpy(np.stack([s["input_xyz"] for s in samples])).to(args.device)
            target = torch.from_numpy(np.stack([s["gt_xyz"] for s in samples])).to(args.device)
            state = model.encode(points, with_m2=False)
            torch.manual_seed(args.seed + start)
            method = ("uniform", "density", "curvature", "random")[(start // args.teacher_batch_size) % 4]
            budget = allocate(model, state, method)
            base_state = initialize(state, budget, model.config.minimum)
            before_output = model.integrate(base_state)
            before = cd_l1(before_output, target)
            repeat_error = cd_l1(model.integrate(base_state), target) - before
            if not torch.isfinite(before).all() or not torch.isfinite(repeat_error).all():
                raise FloatingPointError("Non-finite teacher baseline inference")
            gt_present = region_counts(target, state["regions"]["centers"]) > 0
            valid = state["regions"]["valid"] & gt_present
            delta = model.config.delta
            rng = np.random.default_rng(args.seed + start)
            choices = []
            for batch in range(len(points)):
                legal = (budget[batch]+delta <= state["capacities"][batch])[:, None] & (budget[batch]-delta >= model.config.minimum)[None, :]
                legal &= valid[batch, :, None] & valid[batch, None, :]
                legal.fill_diagonal_(False)
                pairs = legal.nonzero().cpu().numpy()
                if not len(pairs):
                    raise ValueError(f"No valid train teacher exchange for row {rows[batch]}")
                choices.append(pairs[rng.choice(len(pairs), args.teacher_pairs, replace=len(pairs)<args.teacher_pairs)])
            receiver = torch.tensor(np.stack(choices)[..., 0], device=args.device)
            donor = torch.tensor(np.stack(choices)[..., 1], device=args.device)
            after = []
            batch_ids = torch.arange(len(points), device=args.device)
            for pair in range(args.teacher_pairs):
                changed = budget.clone()
                changed[batch_ids, receiver[:, pair]] += delta
                changed[batch_ids, donor[:, pair]] -= delta
                after.append(cd_l1(model.integrate(initialize(state, changed, model.config.minimum)), target))
            after = torch.stack(after, -1)
            if not torch.isfinite(after).all():
                raise FloatingPointError("Non-finite counterfactual inference")
            geometry = {mode: tuple(v.cpu() for v in geometry_feature(state["regions"], mode)) for mode in ("D1", "D2", "D3")}
            existing = dict(rows=rows, input=points.cpu(), input_id=[s["sample_id"] for s in samples],
                            gt_id=[s["sample_id"] for s in samples], object_id=[s["object_id"] for s in samples],
                            object_identity="HDF5 row only; source mesh identity unavailable",
                            input_sha256=[array_hash(s["input_xyz"]) for s in samples],
                            gt_sha256=[array_hash(s["gt_xyz"]) for s in samples],
                            centers=state["regions"]["centers"].cpu(), budget=budget.cpu(),
                            capacities=state["capacities"].cpu(), base_method=method,
                            receiver=receiver.cpu(), donor=donor.cpu(), delta=delta,
                            before=before.cpu(), after=after.cpu(), improvement=(before[:, None]-after).cpu(),
                            repeat_error=repeat_error.cpu(), geometry=geometry,
                            generator_sha256=protocol["generator_sha256"])
            atomic_save(existing, destination)
        shards.append(dict(file=destination.name, sha256=digest_file(destination), samples=len(rows)))
        print(json.dumps(dict(teacher_samples=min(start+len(rows), count), total=count)), flush=True)
    improvements, repeat, geometry_values = [], [], {mode: [] for mode in ("D1", "D2", "D3")}
    validity = {mode: [] for mode in geometry_values}
    for shard in shards:
        data = torch.load(output / shard["file"], map_location="cpu", weights_only=False)
        improvements.append(data["improvement"].flatten())
        repeat.append(data["repeat_error"].abs())
        for mode, (values, valid) in data["geometry"].items():
            geometry_values[mode].append(values[valid])
            validity[mode].append(valid.float().flatten())
    gains, fluctuation = torch.cat(improvements), torch.cat(repeat)
    stats = dict(gain_scale=max(float(gains.abs().quantile(.9)), 1e-8),
                 noise_threshold=max(float(fluctuation.max())*5, 1e-7),
                 positive_fraction=float((gains>0).float().mean()),
                 gain_quantiles=gains.quantile(torch.tensor([0., .1, .5, .9, 1.])).tolist(), geometry={})
    for mode in geometry_values:
        values = torch.cat(geometry_values[mode])
        if not len(values):
            raise ValueError(f"No valid {mode} training geometry for normalization")
        stats["geometry"][mode] = dict(center=float(values.median()),
                                       scale=max(float(values.quantile(.9)-values.quantile(.1)), 1e-8),
                                       valid_fraction=float(torch.cat(validity[mode]).mean()))
    manifest = dict(protocol=protocol, complete=True, shards=shards, statistics=stats)
    write_manifest(manifest_path, manifest)
    return manifest


def write_manifest(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_teacher(path, model, source_sha256):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    protocol = manifest["protocol"]
    if not manifest["complete"] or protocol["split"] != "train":
        raise ValueError("Incomplete or non-training teacher")
    if protocol["source_sha256"] != source_sha256 or protocol["generator_sha256"] != generator_signature(model):
        raise ValueError("Teacher training data or frozen generator mismatch")
    if protocol["code_sha256"] != code_signature():
        raise ValueError("Teacher code changed; use a new cache")
    runtime = dict(torch=torch.__version__, fps_backend=os.environ.get("V3_FPS_BACKEND", "auto"),
                   pointops=os.environ.get("PUFM_POINTOPS", "auto"))
    if any(protocol[key] != value for key, value in runtime.items()):
        raise ValueError("Teacher inference backend changed")
    for shard in manifest["shards"]:
        if digest_file(path / shard["file"]) != shard["sha256"]:
            raise ValueError("Teacher shard checksum mismatch")
    return manifest
