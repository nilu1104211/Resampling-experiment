"""Read PUFM's supplied 256/1024 training pairs directly from PU1K HDF5."""

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np

from adaptive_resampling.core import input_transform
from adaptive_resampling.v3_core import seed


def source_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PU1KSparseDataset:
    def __init__(self, path, split, *, source_sha256):
        if split != "train":
            raise ValueError("PU1K HDF5 is used only for training")
        self.path = str(Path(path).resolve(strict=True))
        self.split = split
        self.epoch = 0
        self._file = None
        self._pid = None
        with h5py.File(self.path, "r") as handle:
            if "poisson_256" not in handle or "poisson_1024" not in handle:
                raise ValueError("PU1K HDF5 must contain poisson_256 and poisson_1024")
            shape_in, shape_out = handle["poisson_256"].shape, handle["poisson_1024"].shape
            if (len(shape_in) != 3 or len(shape_out) != 3 or shape_in[1:] != (256, 3)
                    or shape_out[1:] != (1024, 3) or shape_in[0] != shape_out[0] or shape_in[0] < 2):
                raise ValueError(f"Unexpected PU1K pair shapes: {shape_in}, {shape_out}")
            self.source_samples = shape_in[0]
        self.indices = np.arange(self.source_samples, dtype=np.int64)
        identity = hashlib.sha256(self.indices.astype("<i8").tobytes()).hexdigest()
        self.training_profile = {
            "policy": "all_original_training_rows",
            "samples": len(self.indices), "source_samples": self.source_samples,
            "selection_sha256": identity, "source_config_digest": source_sha256,
            "training_mode": "clean", "input_points": 256,
            "object_disjoint": "unknown",
        }

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, item):
        epoch, index = item if isinstance(item, tuple) else (self.epoch, item)
        row = int(self.indices[int(index)])
        if self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        observed = self._file["poisson_256"][row]
        target = self._file["poisson_1024"][row]
        if not np.isfinite(observed).all() or not np.isfinite(target).all():
            raise ValueError(f"Non-finite coordinates in PU1K row {row}")
        observed, target, _, _ = input_transform(observed, target)
        if self.split == "train":
            rng = np.random.default_rng(seed(f"pu1k_h5/{row}", "rotation", int(epoch)))
            rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            rotation = rotation.astype("f4")
            observed, target = observed @ rotation.T, target @ rotation.T
        sample_id = f"pu1k_h5/{row}"
        return {"input_xyz": observed.astype("f4"), "gt_xyz": target.astype("f4"),
                "sample_id": sample_id, "object_id": sample_id, "condition": "A0_normal",
                # CD is evaluated in the normalized input frame, not source coordinates.
                "prediction_scale": np.float32(1)}

    def close(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
        self._file = None
        self._pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = state["_pid"] = None
        return state

    def __del__(self):
        self.close()
