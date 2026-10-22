"""Framework-neutral data access. Metadata/region masks are deliberately opt-in."""
from pathlib import Path

import h5py
import numpy as np

from .core import input_transform


def load_sample(path, index, *, normalize_for_model=True, include_supervision=False):
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError("Incomplete generated dataset")
        inp, gt = handle["input_xyz"][index], handle["gt_xyz"][index]
        result = {"sample_id": handle["sample_id"].asstr()[index]}
        if normalize_for_model:
            inp, gt, center, scale = input_transform(inp, gt)
        else:
            center, scale = np.zeros(3, dtype="f4"), 1.0
        result.update(input_xyz=inp, gt_xyz=gt, prediction_center=center, prediction_scale=scale)
        if include_supervision:
            for key in ["hole_mask_gt", "sparse_mask_gt", "noise_mask_input"]:
                result[key] = handle[key][index]
        return result


def inverse_prediction(prediction, sample):
    """Return canonical benchmark coordinates for shared evaluation."""
    return np.asarray(prediction) * sample["prediction_scale"] + sample["prediction_center"]


class H5PatchDataset:
    """Duck-typed PyTorch Dataset; no dependency on torch or CUDA for data preparation."""
    def __init__(self, path, normalize_for_model=True, include_supervision=False):
        self.path = str(Path(path))
        self.normalize = normalize_for_model
        self.supervision = include_supervision
        with h5py.File(self.path, "r") as f:
            self.length = len(f["input_xyz"])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return load_sample(self.path, index, normalize_for_model=self.normalize, include_supervision=self.supervision)
