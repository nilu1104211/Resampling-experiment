"""One deterministic per-object training subset shared by every V3 network."""
import hashlib
import json

import numpy as np

from .v3_core import seed

DEFAULT_TRAIN_PATCHES_PER_OBJECT = 32


def training_selection(manifest, patches_per_object=DEFAULT_TRAIN_PATCHES_PER_OBJECT, *, training_mode="mixed"):
    if isinstance(patches_per_object, bool) or not isinstance(patches_per_object, int) or patches_per_object < 0:
        raise ValueError("train_patches_per_object must be a nonnegative integer; zero uses all stored patches")
    maps, offsets, identity = [], [], []
    source_count = 0
    for entry in manifest["groups"]["train"]:
        size = int(entry["rows"])
        object_id = entry.get("object_id", entry["path"])
        rows = np.arange(size, dtype="i4")
        if 0 < patches_per_object < size:
            # Same subset across models, training seeds, epochs and worker counts.
            # Smaller budgets are nested subsets, with no bias toward early FPS centers.
            rng = np.random.default_rng(seed(object_id, "training-patch-subset-v1"))
            rows = np.sort(rng.permutation(size)[:patches_per_object]).astype("i4")
        maps.append(rows)
        offsets.append(source_count)
        source_count += size
        identity.append([object_id, entry["path"], size, rows.tolist()])
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    profile = {
        "policy": "per-object-seeded-subset-v1", "patches_per_object": patches_per_object,
        "objects": len(maps), "samples": sum(map(len, maps)), "source_samples": source_count,
        "selection_sha256": digest, "source_config_digest": manifest.get("config_digest"),
        "training_mode": training_mode,
    }
    return maps, offsets, profile


def require_training_profile(saved, current, *, context="Resume"):
    previous = saved.get("training_profile")
    if previous is None:
        # Checkpoints/caches predating subset support always used every stored row.
        if current["samples"] != current["source_samples"]:
            raise ValueError(f"{context} cannot change the legacy full-data training budget to a patch subset; "
                             "start a new run or use --train-patches-per-object 0 for legacy full data")
        return
    keys = ("selection_sha256", "source_config_digest", "samples", "source_samples", "training_mode")
    if any(previous.get(key) != current.get(key) for key in keys):
        raise ValueError(f"{context} requires the same training patch selection and data mode; "
                         "use a new output/checkpoint stage when changing the training budget")
    if previous.get("input_points", 256) != current.get("input_points", 256):
        raise ValueError(f"{context} cannot change the training observation point budget")
