"""Matched Our2 training with the original sampler, loss, optimizer and checkpoints."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from our2_p0 import load_config, read_json, run
from our2_v56 import midpoint
from our2_diagnosis_seeds import direction_seeds
from adaptive_resampling.v3_transport import configure_assignment_workers


TRAIN_GROUPS = ("T0", "T1", "T2", "D0-FT", "D1-L-FT", "D1-U-FT")
PROBE_STRATA = {"online_normal", "online_sparse", "online_hole", "online_noise", "online_mixed",
                "online_noise_outliers", "online_mixed_outliers"}


def options_for(args, group, output):
    m1 = group in ("T1", "T2")
    config = load_config(args, "m1_refine")
    if not m1:
        config.update(stage="baseline", enable_m1=False)
    if group == "T1":
        config["recovery_weight"] = 0.0
    config["epochs"] = args.epochs
    options = SimpleNamespace(**vars(args))
    options.command = "train"
    options.output = output
    options.limit_batches = args.debug
    options.limit_samples = args.debug
    options.checkpoint = options.m1_checkpoint = options.m2_checkpoint = None
    options.init = args.training_run / "m1_hole" / "best.pt" if m1 else None
    options.allow_debug = False
    options.fused_optimizer = read_json(args.training_run / "m1_refine" / "manifest.json")["fused_optimizer"]
    options.resume = output / "last.pt" if (output / "last.pt").is_file() else None
    return options, config


def attach_seed_path(model, group, head):
    if group == "T1":
        original_prepare = model.prepare

        def without_movement(observed):
            state = original_prepare(observed)
            state["initial"] = state["original"]
            state.pop("movement", None)
            state["diagnostics"].pop("movement_mean", None)
            return state

        model.prepare = without_movement
        # Both arms share the pretrained heads. The final T1 window updates Flow only.
        model.m1.requires_grad_(False)
    if group in ("D1-L-FT", "D1-U-FT"):
        direction = group.removesuffix("-FT")
        last_seed_stats = {}

        @torch.no_grad()
        def prepare(observed):
            # Per-item Midpoint also matches the frozen Patch evaluation.
            values = [direction_seeds(midpoint(points), points, head, "", (direction,))[1][direction]
                      for points in observed.split(1)]
            last_seed_stats.update(actual_seed_replacements=np.mean([value[2] for value in values]),
                                   uniform_relaxed_fraction=np.mean([value[3].startswith("uniform_")
                                                                    and value[3] != "uniform_target" for value in values]))
            return {"initial": torch.cat([value[0] for value in values]), "diagnostics": {}}

        training_loss = model.training_loss

        def with_seed_stats(batch):
            loss = training_loss(batch)
            model.loss_diagnostics.update({key: float(value) for key, value in last_seed_stats.items()})
            return loss

        model.prepare = prepare
        model.training_loss = with_seed_stats


def train_group(args, group, output, head):
    options, config = options_for(args, group, output)
    configure_assignment_workers(config["assignment_workers"])
    run.seed_all(config["seed"])
    model, saved = run.make_model(options, config)
    attach_seed_path(model, group, head)
    if options.init and saved is None:
        from models.dual import DualPUFM, load_baseline
        original = DualPUFM(config)
        load_baseline(original, run.read_checkpoint(args.baseline_checkpoint))
        for name, value in original.flow.state_dict().items():
            if not torch.equal(value, model.flow.state_dict()[name].detach().cpu()):
                raise ValueError(f"Shared M1 warmup changed the B0 Flow tensor {name}")
    run.seed_all(config["seed"])
    # An interruption before the epoch-zero snapshot can leave evaluation files.
    if options.resume is None and output.exists():
        import uuid
        output.rename(output.with_name(f"{output.name}.interrupted-{uuid.uuid4().hex[:8]}"))
    run.train(model, saved, options, config)
    last = run.read_checkpoint(output / "last.pt")
    return {"config": config, "updates": sum(row["optimizer_steps"] for row in last["history"]),
            "initial_flow_sha256": run.file_hash(args.baseline_checkpoint),
            "shared_parent_sha256": run.file_hash(options.init) if options.init else None,
            "best_epoch": read_json(output / "best.json")["epoch"],
            "best_sha256": run.file_hash(output / "best.pt"), "last_sha256": run.file_hash(output / "last.pt"),
            "fused_optimizer": options.fused_optimizer, "diagnostic_short_run": bool(args.debug)}


def load_trained(args, group, output, head, checkpoint="last"):
    options, config = options_for(args, group, output)
    options.resume = None
    options.command = "predict"
    options.allow_debug = bool(args.debug)
    options.checkpoint = output / f"{checkpoint}.pt"
    model, _ = run.make_model(options, config)
    attach_seed_path(model, group, head)
    return model.eval().requires_grad_(False)


def probe_stratum(sample):
    condition = sample["condition"]
    labels = sample.get("supervision", sample)
    if condition in ("online_noise", "online_mixed") and labels["outlier_mask_input"].any():
        return f"{condition}_outliers"
    return condition


def probe_indices(dataset, epoch):
    from adaptive_resampling.v3_dataset import EpochSampler

    counts = defaultdict(int)
    sampler = EpochSampler(dataset)
    sampler.set_epoch(epoch)
    for item in sampler:
        raw = dataset.raw_sample(item[1], item[0])
        condition = probe_stratum(raw)
        if counts[condition] < 3:
            counts[condition] += 1
            yield item
        if all(counts[c] == 3 for c in PROBE_STRATA):
            return


def gradient_summary(named, optimizer_ids):
    def summarize(values):
        grads = [p.grad for p in values if p.grad is not None]
        return {
            "parameters": sum(p.numel() for p in values),
            "trainable": sum(p.numel() for p in values if p.requires_grad),
            "optimizer_parameters": sum(p.numel() for p in values if id(p) in optimizer_ids),
            "gradient_l2": sum(float(g.double().square().sum()) for g in grads) ** .5,
            "gradient_nonzero_fraction": sum(int(g.count_nonzero()) for g in grads) / sum(p.numel() for p in values),
        }
    blocks = defaultdict(list)
    for name, param in named:
        blocks[".".join(name.split(".")[:2]) if not name.startswith("flow.") else "flow"].append(param)
    return {key: summarize(values) for key, values in blocks.items()}


def gradient_components(components, loss, named):
    """Measure auxiliary and generative gradients on the same actual forward graph."""
    if not components:
        return []
    remaining = loss - sum(components.values())
    components = {**components, "fm_recovery_move": remaining}
    parameters = [(name, p) for name, p in named if p.requires_grad]
    grads = {key: torch.autograd.grad(term, [p for _, p in parameters], allow_unused=True, retain_graph=True)
             for key, term in components.items() if term.requires_grad}
    rows = []
    for block in ("flow", "m1", "m2"):
        indexes = [i for i, (name, _) in enumerate(parameters) if name.startswith(f"{block}.")]
        if not indexes:
            continue
        for name, grad in grads.items():
            norm = sum(float(grad[i].detach().double().square().sum())
                       for i in indexes if grad[i] is not None) ** .5
            rows.append({"parameter_block": block, "loss_component": name, "gradient_l2": norm})
        if "budget" in grads and "fm_recovery_move" in grads:
            left, right = grads["budget"], grads["fm_recovery_move"]
            a = sum(float(left[i].detach().double().square().sum()) for i in indexes if left[i] is not None)
            b = sum(float(right[i].detach().double().square().sum()) for i in indexes if right[i] is not None)
            dot = sum(float((left[i].detach().double() * right[i].detach()).sum())
                      for i in indexes if left[i] is not None and right[i] is not None)
            rows.append({"parameter_block": block, "loss_component": "budget_vs_fm_recovery_move",
                         "gradient_cosine": dot / (a * b) ** .5 if a and b else None,
                         "cosine_reason": "" if a and b else "ZERO_GRADIENT"})
    return rows


def probe(args, stage, model):
    """Use the real loss on an isolated model and discard the optimizer update."""
    from torch.utils.data._utils.collate import default_collate
    from torch.nn import functional as F

    config = load_config(args, stage)
    options = SimpleNamespace(init=None, resume=None,
                              baseline_checkpoint=args.baseline_checkpoint,
                              fused_optimizer=read_json(args.training_run / stage / "manifest.json")["fused_optimizer"])
    configure_assignment_workers(config["assignment_workers"])
    ds = run.training_dataset(args, config)
    try:
        shared_epoch = load_config(args, "m1_refine")["data_epoch_offset"]
        samples = [ds[i] for i in probe_indices(ds, shared_epoch)]
    finally:
        ds.close()
    if not samples:
        raise ValueError("No training probes were selected")
    if len(samples) != 21 or {probe_stratum(s) for s in samples} != PROBE_STRATA:
        raise ValueError("Could not select the same three training probes for each condition")
    model.set_stage(stage)
    model.train()
    optimizer = run.make_stage_optimizer(model, options, config)
    named = list(model.named_parameters())
    params = [p for _, p in named if p.requires_grad]
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    rows = []
    run.seed_all(21)
    for index in range(0, len(samples), 7):
        chunk = samples[index:index + 7]
        if len(chunk) < 2:
            chunk = samples[-2:]
        batch = run.batch_to(default_collate(chunk), args.device)
        optimizer.zero_grad(set_to_none=True)
        state = {}
        components = {}
        targets = {}
        prepare = model.prepare
        bce, kl = F.binary_cross_entropy_with_logits, F.kl_div

        def record_prepare(points):
            state.update(prepare(points))
            return state

        def record_bce(logits, target, **kwargs):
            raw = bce(logits, target, **kwargs)
            name = "rel" if logits is state["m1"]["rlogit"] else "hole"
            weights = 1 + 4 * ((1 - target) if name == "rel" else target)
            components[name] = .01 * (raw * weights).mean()
            targets[name] = target
            return raw

        def record_kl(logits, target, **kwargs):
            raw = kl(logits, target, **kwargs)
            components["budget"] = .01 * raw
            targets["budget"] = target
            return raw

        with (patch.object(model, "prepare", record_prepare),
              patch.object(F, "binary_cross_entropy_with_logits", record_bce),
              patch.object(F, "kl_div", record_kl)):
            loss = model.training_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite probe loss")
        for name, term in components.items():
            if not np.isclose(float(term.detach()), model.loss_diagnostics[name], rtol=1e-6, atol=1e-10):
                raise ValueError(f"Probe {name} does not match the actual training objective")
        component_rows = gradient_components(components, loss, named)
        named_outputs = {k: v for k, v in (state.get("m1") or {}).items() if k in ("rlogit", "hlogit")}
        named_outputs.update({k: state[k] for k in ("movement", "logits") if k in state})
        for value in named_outputs.values():
            if value.requires_grad:
                value.retain_grad()
        loss.backward()
        gradients = gradient_summary(named, opt_ids)
        before = [p.detach().clone() for p in params]
        norm = torch.nn.utils.clip_grad_norm_(params, config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        update = {id(p): float((p.detach() - old).double().square().sum()) for p, old in zip(params, before)}
        common = {"stage": stage, "batch": index // 7, "examples": len(chunk), "loss": float(loss.detach()),
                  "global_preclip_gradient_l2": float(norm),
                  "sample_ids": "|".join(x["sample_id"] for x in chunk),
                  "conditions": "|".join(probe_stratum(x) for x in chunk),
                  "reliability_target_mean": float(batch["supervision"]["reliability_target_input"].mean()),
                  "unreliable_target_fraction": float((batch["supervision"]["reliability_target_input"] < .5).float().mean()),
                  "hole_gt_fraction": float(batch["supervision"]["hole_mask_gt"].mean()),
                  **{f"loss_{k}": value for k, value in model.loss_diagnostics.items()}}
        if "hole" in targets:
            common.update(h_candidate_count=targets["hole"].numel(),
                          h_target_positive_fraction=float((targets["hole"] >= .5).float().mean()))
        if "budget" in targets:
            probability = (state["logits"].detach() / config["temperature"]).softmax(-1)
            common.update(budget_temperature=config["temperature"],
                          budget_entropy=float(-(probability * probability.clamp_min(1e-10).log()).sum(-1).mean()),
                          proxy_min=float(targets["budget"].min()), proxy_max=float(targets["budget"].max()),
                          hard_budget_differentiable=state["budget"]["hard"].requires_grad if "budget" in state else None)
        rows.extend({**common, **item} for item in component_rows)
        for name, values in gradients.items():
            delta = sum(update.get(id(p), 0.) for key, p in named if key.startswith(f"{name}.") or key == name)
            rows.append({**common, "parameter_block": name, **values, "update_l2": delta ** .5})
        for name, value in named_outputs.items():
            rows.append({**common, "parameter_block": name, "tensor_requires_grad": value.requires_grad,
                         "gradient_l2": float(value.grad.norm()) if value.requires_grad and value.grad is not None else 0.})
    return rows
