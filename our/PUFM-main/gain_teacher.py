"""Train-only, reproducible fixed-budget swaps after the actual PUFM generator."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
import torch
from models.adaptive_ops import cd_l1
from models.budget_allocator import allocate


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    return value


def generator_signature(model):
    """Gain-head training does not invalidate its frozen-generator cache."""
    config = asdict(model.config)
    for key in ["allocation", "gain_form", "swap_step", "swap_iterations", "teacher_gain_scale"]:
        config.pop(key)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    from models.pointops_compat import _native
    digest.update(("installed_cuda" if _native is not None and next(model.parameters()).is_cuda else "portable_torch").encode())
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(("flow.", "rgr.", "conditioner.")):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


@torch.no_grad()
def paired_labels(model, state, gt, canonical_scale, *, pairs=2, seed=21, base_method="uniform", delta=None):
    """Reuse all region pools and one baseline trajectory; alter ONLY A/B quotas."""
    previous = model.training
    model.eval()
    state = detach_tree(state)
    delta = delta or model.config.swap_step
    try:
        total = state["output_size"] - state["input"].shape[1]
        base = allocate(state["regions"], model.gain, total, model.config.minimum, model.config.capacity,
                        delta, model.config.swap_iterations, method=base_method)
        baseline_state = model.initialize(dict(state), base)
        baseline = model.integrate(baseline_state)
        before = cd_l1(baseline, gt) * canonical_scale.reshape(-1)
        rng = np.random.default_rng(seed)
        records = []
        choices = []
        for batch in range(len(gt)):
            valid = [(a, b) for a in range(base.shape[1]) for b in range(base.shape[1])
                     if a != b and base[batch, a] + delta <= model.config.capacity
                     and base[batch, b] - delta >= model.config.minimum]
            if not valid:
                raise ValueError("No feasible counterfactual budget swap")
            order = rng.choice(len(valid), size=pairs, replace=pairs > len(valid))
            choices.append([valid[int(i)] for i in order])
        batch_ids = torch.arange(len(gt), device=gt.device)
        for index in range(pairs):
            receiver = torch.tensor([c[index][0] for c in choices], device=gt.device)
            donor = torch.tensor([c[index][1] for c in choices], device=gt.device)
            swapped = base.clone()
            swapped[batch_ids, receiver] += delta
            swapped[batch_ids, donor] -= delta
            alternative_state = model.initialize(dict(state), swapped)
            alternative = model.integrate(alternative_state)
            after = cd_l1(alternative, gt) * canonical_scale.reshape(-1)
            records.append({"features": state["regions"]["gain_features"], "budget": base,
                            "receiver": receiver, "donor": donor, "delta": delta,
                            "improvement": before - after, "before": before, "after": after})
        return records
    finally:
        model.train(previous)


def write_cache(model, dataset, path, indices, epochs, *, pairs=4, seed=21, debug=False):
    from pathlib import Path
    if getattr(dataset, "split", None) != "train":
        raise ValueError("Counterfactual training teachers can ONLY use split=train")
    model.eval()
    signature = generator_signature(model)
    fields = {key: [] for key in ["features", "budget", "receiver", "donor", "improvement", "before", "after",
                                  "sample_id", "base_id", "object_id", "index", "epoch", "delta", "input_sha256", "gt_sha256"]}
    device = next(model.parameters()).device
    for epoch in epochs:
        for number in indices:
            sample = dataset[(int(epoch), int(number))]
            points = torch.from_numpy(sample["input_xyz"])[None].to(device)
            gt = torch.from_numpy(sample["gt_xyz"])[None].to(device)
            scale = points.new_tensor([sample["prediction_scale"]])
            with torch.no_grad():
                state = model.encode(points, for_teacher=True)
                methods = ["uniform", "density", "curvature"]
                selected_seed = stable_seed(seed, number, epoch)
                delta = [max(1, model.config.swap_step // 2), model.config.swap_step, model.config.swap_step * 2][selected_seed % 3]
                records = paired_labels(model, state, gt, scale, pairs=pairs, delta=delta,
                                        base_method=methods[(selected_seed // 3) % len(methods)], seed=selected_seed)
            for record in records:
                for key in ["features", "budget", "receiver", "donor", "improvement", "before", "after"]:
                    fields[key].append(record[key][0].cpu().numpy())
                for key in ["sample_id", "base_id", "object_id"]:
                    fields[key].append(sample[key])
                fields["index"].append(number)
                fields["epoch"].append(epoch)
                fields["delta"].append(record["delta"])
                for key in ["input", "gt"]:
                    fields[key + "_sha256"].append(hashlib.sha256(sample[key + "_xyz"].tobytes()).hexdigest())
            print(json.dumps({"teacher_epoch": epoch, "sample": int(number), "swaps": len(fields["delta"])}), flush=True)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **{key: np.asarray(value) for key, value in fields.items()})
    metadata = {"protocol": "V3", "split": "train", "generator_sha256": signature,
                "training_mode": dataset.training_mode,
                "configuration": asdict(model.config), "records": len(fields["delta"]),
                "seed": seed, "epochs": list(epochs), "base_indices": list(map(int, indices)),
                "teacher_error": "CD-L1 in common canonical coordinates; original uniform-sampled GT",
                "candidate_policy": "immutable nested region pools; identical other budgets and generator",
                "partial_debug_run": debug, "untrained_generator": debug}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
