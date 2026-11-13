"""Fixed ModelNet40 validation and resumable best-checkpoint selection."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from scipy.spatial import cKDTree
import torch

from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def validation_state(model):
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def modelnet_validation(root, *, condition="A0_normal", include_supervision=False, include_reference_metadata=False):
    selection_path = Path(__file__).with_name("modelnet40_validation_ids.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["object_ids"]
    if len(selected) != 100 or len(set(selected)) != 100:
        raise ValueError("ModelNet40 validation selection must contain 100 unique objects")
    ds = AdaptiveResamplingDataset(root, "modelnet40_patch", condition=condition,
                                   include_supervision=include_supervision,
                                   include_reference_metadata=include_reference_metadata, augment_rotation=False)
    indices, found, shards = [], set(), {}
    selected_set = set(selected)
    start = 0
    for entry, end in zip(ds.entries, ds.ends):
        if entry["object_id"] in selected_set:
            if entry["object_id"] in found:
                ds.close()
                raise ValueError("Duplicate validation object in ModelNet40 manifest")
            found.add(entry["object_id"])
            indices.extend(range(start, end))
            path = Path(root) / "processed" / "modelnet40_patch" / Path(entry["path"]).name
            shards[entry["object_id"]] = digest_file(path)
        start = end
    if found != selected_set:
        ds.close()
        raise ValueError(f"ModelNet40 validation objects missing from dataset: {sorted(selected_set - found)}")
    return ds, indices, shards, selection_path


class ModelNet40HoldoutDataset:
    """The frozen ModelNet40 V3 objects outside the validation selection."""

    def __init__(self, root, condition, *, include_supervision=False, include_reference_metadata=False):
        self.dataset, selected, _, _ = modelnet_validation(
            root, condition=condition, include_supervision=include_supervision,
            include_reference_metadata=include_reference_metadata)
        val = set(selected)
        self.indices = [i for i in range(len(self.dataset)) if i not in val]
        if not self.indices:
            self.dataset.close()
            raise ValueError("ModelNet40 holdout is empty")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def close(self):
        self.dataset.close()


class BaselineValidation:
    def __init__(self, args, saved=None):
        self.output = Path(args.output)
        ds, self.indices, shards, selection_path = modelnet_validation(args.root)
        ds.close()
        manifest = Path(args.root) / "metadata" / "base_manifest.json"
        self.protocol = {
            "split": "modelnet40_patch_custom_validation", "source_split": "official_test",
            "condition": "A0_normal", "metric": "canonical_cd_l1_object_mean",
            "manifest_sha256": digest_file(manifest),
            "generation_config_sha256": digest_file(Path(args.root) / "metadata" / "generation_config.json"),
            "selection_sha256": digest_file(selection_path),
            "selected_shards_sha256": hashlib.sha256(json.dumps(shards, sort_keys=True).encode()).hexdigest(),
            "objects": len(shards), "samples": len(self.indices),
            "max_samples": args.validation_limit, "full_validation": args.validation_limit == 0,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": args.device, "fps_backend": os.environ.get("V3_FPS_BACKEND", "auto"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        self.args = args
        self.best = saved.get("best_validation") if saved else None
        best_path = self.output / "best.pt"
        if args.resume:
            if Path(args.resume).resolve() != (self.output / "last.pt").resolve():
                raise ValueError("Baseline resume requires last.pt in the same output directory")
            if not self.best:
                raise ValueError("This older checkpoint has no validation history; start a new baseline run")
            if self.best["protocol"] != self.protocol:
                raise ValueError("Resume must use the same clean-validation data and protocol")
            if saved["epoch"] == self.best["epoch"]:
                # Repair a crash after last.pt was saved but before best.pt was written.
                atomic_save(saved, best_path)
            else:
                if not best_path.is_file():
                    raise FileNotFoundError("Resume requires the previous best.pt alongside last.pt")
                previous = torch.load(best_path, map_location="cpu", weights_only=False)
                if previous.get("best_validation") != self.best or previous.get("epoch") != self.best["epoch"]:
                    raise ValueError("best.pt does not match the resumed validation history")
        elif best_path.exists() or (self.output / "last.pt").exists():
            raise FileExistsError("Use a new output directory or resume its last.pt")

    def evaluate(self, model, epoch):
        ds = AdaptiveResamplingDataset(self.args.root, "modelnet40_patch", condition="A0_normal",
                                       include_supervision=False, augment_rotation=False)
        groups = {}
        limit = min(self.args.validation_limit or len(self.indices), len(self.indices))
        try:
            with validation_state(model):
                for index in range(limit):
                    sample = ds[self.indices[index]]
                    observed = torch.as_tensor(sample["input_xyz"], device=self.args.device)[None]
                    prediction = model(observed)[0].cpu().numpy()
                    target = sample["gt_xyz"]
                    if prediction.shape != target.shape or not np.isfinite(prediction).all():
                        raise FloatingPointError("Invalid clean validation output")
                    forward = cKDTree(target).query(prediction, workers=1)[0]
                    backward = cKDTree(prediction).query(target, workers=1)[0]
                    score = float(.5 * (forward.mean() + backward.mean()) * sample["prediction_scale"])
                    groups.setdefault(sample["object_id"], []).append(score)
                    if (index + 1) % 100 == 0:
                        print(f"ModelNet40 validation epoch {epoch + 1}: {index + 1}/{limit}", flush=True)
        finally:
            ds.close()
        if not groups or (self.args.validation_limit == 0 and
                          (limit != self.protocol["samples"] or len(groups) != self.protocol["objects"])):
            raise ValueError("Clean validation is empty or incomplete")
        value = float(np.mean([np.mean(scores) for scores in groups.values()]))
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite clean validation score")
        report = {"epoch": epoch, "epoch_one_based": epoch + 1, "value": value,
                  "samples": limit, "objects": len(groups),
                  "protocol": self.protocol}
        improved = self.best is None or value < self.best["value"]
        if improved:
            self.best = report
        return report, improved
