"""Validation-only checkpoint selection, with reproducible resume state."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from scipy.spatial import cKDTree
import torch

from adaptive_resampling.v3_core import VALIDATION_CONDITIONS
from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset


METRIC = "canonical_cd_l1_object_condition_family_mean"


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def evaluation_state(model):
    modes = [(module, module.training) for module in model.modules()]
    configuration = model.config
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        model.config = configuration
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def aggregate(rows):
    """Average patches within objects, objects within conditions, then 5 families."""
    by_object = {}
    for condition, object_id, score in rows:
        by_object.setdefault((condition, object_id), []).append(score)
    conditions = {}
    for (condition, _), values in by_object.items():
        conditions.setdefault(condition, []).append(float(np.mean(values)))
    conditions = {key: float(np.mean(values)) for key, values in conditions.items()}
    families = {}
    for condition, score in conditions.items():
        families.setdefault(condition[0], []).append(score)
    families = {key: float(np.mean(values)) for key, values in families.items()}
    if set(conditions) != set(VALIDATION_CONDITIONS) or set(families) != set("ABCDE"):
        raise ValueError("Checkpoint selection requires all seven validation conditions")
    return float(np.mean(list(families.values()))), conditions, families


class ValidationSelection:
    def __init__(self, args, saved=None):
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        manifest_path = Path(args.root) / "metadata/base_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        entries = manifest["groups"]["validation"]
        ranked = sorted(entries, key=lambda e: hashlib.sha256(
            f"{args.seed}/validation/{e['object_id']}".encode()).hexdigest())
        chosen = ranked[:args.val_objects] if args.val_objects else ranked
        self.objects = {entry["object_id"] for entry in chosen}
        self.protocol = dict(split="validation", metric=METRIC,
                             conditions=list(VALIDATION_CONDITIONS),
                             data_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                             object_ids=sorted(self.objects), objects=len(chosen),
                             expected_samples=sum(int(e["rows"]) for e in chosen) * len(VALIDATION_CONDITIONS),
                             full_validation=len(chosen) == len(entries),
                             prediction="rgr_midpoint_frozen_flow" if args.stage == "rgr" else "model_output")
        self.best = saved.get("best_validation") if saved is not None else None
        self.latest = None
        if self.best and self.best["protocol"] != self.protocol:
            raise ValueError("Resume must preserve the validation data and selection protocol")
        destination = self.output / "best.pt"
        if self.best:
            source = Path(args.resume).parent / "best.pt"
            if saved["epoch"] == self.best["epoch"]:
                atomic_save({**saved, "checkpoint_role": "validation_best"}, destination)
            elif source.is_file():
                # Check provenance before carrying an older best to the resumed run.
                previous = torch.load(source, map_location="cpu", weights_only=False)
                if previous.get("best_validation") != self.best or previous.get("stage") != args.stage:
                    raise ValueError("best.pt does not match the resumed checkpoint")
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
            else:
                raise FileNotFoundError("Resume needs best.pt alongside last.pt to preserve the previous best")
        elif destination.exists() and not args.resume:
            raise FileExistsError("Output already contains best.pt; resume that run or choose a new output")

    def due(self, epoch):
        return self.args.val_every > 0 and ((epoch + 1) % self.args.val_every == 0 or epoch + 1 == self.args.epochs)

    def evaluate(self, model, epoch):
        rows = []
        ds = AdaptiveResamplingDataset(self.args.root, "validation", condition=None,
                                       include_supervision=False, augment_rotation=False)
        try:
            with evaluation_state(model):
                if self.args.stage == "rgr":
                    # GainNet is not yet trained during frontend warmup.
                    model.config = replace(model.config, allocation="midpoint")
                for index in range(len(ds)):
                    sample = ds[index]
                    if sample["object_id"] not in self.objects:
                        continue
                    points = torch.as_tensor(sample["input_xyz"], device=self.args.device)[None]
                    prediction = model(points)[0].cpu().numpy()
                    target = sample["gt_xyz"]
                    if prediction.shape != target.shape or not np.isfinite(prediction).all():
                        raise FloatingPointError("Invalid validation prediction")
                    forward = cKDTree(target).query(prediction, workers=1)[0]
                    backward = cKDTree(prediction).query(target, workers=1)[0]
                    cd = float(.5 * (forward.mean() + backward.mean()) * sample["prediction_scale"])
                    rows.append((sample["condition"], sample["object_id"], cd))
                    if len(rows) % 100 == 0:
                        print(f"Validation epoch {epoch + 1}: {len(rows)}/{self.protocol['expected_samples']}", flush=True)
        finally:
            ds.close()
        if len(rows) != self.protocol["expected_samples"]:
            raise ValueError("Incomplete validation population")
        value, conditions, families = aggregate(rows)
        report = dict(epoch=epoch, value=value, protocol=self.protocol,
                      samples=len(rows), by_condition=conditions, by_family=families)
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite validation score")
        self.latest = report
        improved = self.best is None or value < self.best["value"]
        if improved:
            self.best = report
            write_json(self.output / "best.json", dict(**report, completed_epochs=0,
                                                     checkpoint="best.pt", epoch_one_based=epoch + 1))
        with (self.output / "validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**report, "improved": improved}) + "\n")
        print(json.dumps(dict(validation_epoch=epoch + 1, validation_cd_l1=value,
                              best_epoch=self.best["epoch"] + 1, best_cd_l1=self.best["value"])), flush=True)
        return improved

    def complete(self):
        if not self.best:
            return
        path = self.output / "best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["selection_completed_epochs"] = self.args.epochs
        payload["args"]["epochs"] = self.args.epochs
        atomic_save(payload, path)
        write_json(self.output / "best.json", dict(**self.best, completed_epochs=self.args.epochs,
                                                 checkpoint="best.pt", epoch_one_based=self.best["epoch"] + 1))
