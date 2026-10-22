"""Direct PyTorch-compatible V3 datasets; torch itself is optional for data QA."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import json
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .core import input_transform
from .v3_core import CONDITIONS, VALIDATION_CONDITIONS, corrupt_base, training_condition, seed
from .v3_pipeline import read_row
from .v3_paths import DEFAULT_ROOT, SourcePaths

SUPERVISION_KEYS = ["clean_input_xyz", "clean_input_normals", "denoise_offset", "gt_normals", "gt_curvature",
                    "gt_normal_variation", "hole_mask_gt", "sparse_mask_gt", "noise_mask_input",
                    "outlier_mask_input", "hole_boundary_mask_input", "hole_boundary_distance_input",
                    "reliability_target_input"]


def input_geometry_features(points, scales=(8, 24)):
    """Eight observable features for A/B modules, computed without any reference.

For each k: PCA surface variation, relative KNN spacing, normalized plane
residual, and unoriented neighbor-normal variation. No GT normals are used.
"""
    points = np.asarray(points, dtype="f8")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite input points with shape N x 3")
    distances, ids = cKDTree(points).query(points, k=min(max(scales), len(points)), workers=1)
    columns = []
    for k in scales:
        neighbors = ids[:, :min(k, len(points))]
        local = points[neighbors]
        local = local - local.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", local, local) / neighbors.shape[1]
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 0)
        normal = vectors[:, :, 0]
        spacing = distances[:, 1:neighbors.shape[1]].mean(axis=1)
        curvature = values[:, 0] / np.maximum(values.sum(axis=1), 1e-15)
        relative_density = spacing / max(float(np.median(spacing)), 1e-12)
        residual = np.sqrt(values[:, 0]) / np.maximum(spacing, 1e-12)
        alignment = np.abs(np.einsum("ni,nki->nk", normal, normal[neighbors]))
        normal_variation = (1 - np.clip(alignment, 0, 1)).mean(axis=1)
        columns.extend([curvature, relative_density, residual, normal_variation])
    return np.column_stack(columns).astype("f4")


def inverse_prediction(prediction, sample, *, raw_coordinates=False):
    """Undo training rotation and input-only normalization before evaluation."""
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    prediction = array(prediction)
    rotation = array(sample.get("prediction_rotation", np.eye(3)))
    def affine(points, scale, center):
        scale, center = array(scale), array(center)
        if points.ndim >= 3 and center.ndim == 2:
            center = center[:, None, :]
            scale = scale.reshape((-1, 1, 1))
        return points * scale + center
    canonical = affine(prediction @ rotation, sample["prediction_scale"], sample["prediction_center"])
    if raw_coordinates:
        canonical = affine(canonical, sample["canonical_scale"], sample["canonical_center"])
    return canonical


def prepare_inference(points):
    """Real-data entry point: no clean input, GT, masks or mesh is required."""
    normalized, _, center, scale = input_transform(points)
    return {"input_xyz": normalized, "prediction_center": center, "prediction_scale": np.float32(scale),
            "prediction_rotation": np.eye(3, dtype="f4")}


class AdaptiveResamplingDataset:
    """Train on online mesh-backed corruptions; evaluate on frozen conditions.

    Use EpochSampler when persistent DataLoader workers are enabled. It carries
    the epoch in every index, so worker copies never use a stale epoch value.
    """
    def __init__(self, root=DEFAULT_ROOT, split="train", *, condition=None, training_mode="mixed",
                 include_supervision=None, normalize_for_model=True, augment_rotation=None,
                 include_geometry_features=False, include_reference_metadata=False, max_open_files=8):
        self.root = Path(root)
        self.split = split
        self.training = split == "train"
        self.training_mode = training_mode
        if training_mode not in {"mixed", "clean"}:
            raise ValueError("training_mode must be mixed or clean")
        self.include_supervision = self.training if include_supervision is None else bool(include_supervision)
        self.normalize = normalize_for_model
        self.rotate = self.training if augment_rotation is None else bool(augment_rotation)
        if self.rotate and not self.training:
            raise ValueError("Evaluation rotations must remain fixed; disable augment_rotation")
        self.features = include_geometry_features
        self.reference_metadata = include_reference_metadata
        self.source_paths = SourcePaths(self.root) if include_reference_metadata else None
        self.max_open_files = max_open_files
        self.epoch = 0
        self._handles = OrderedDict()
        self._pid = os.getpid()
        manifest = json.loads((self.root / "metadata/base_manifest.json").read_text(encoding="utf-8"))
        if split not in manifest["groups"]:
            raise ValueError(f"Unavailable split: {split}")
        self.entries = manifest["groups"][split]
        self.paths, self.row_maps, self.ends = [], [], []
        available = VALIDATION_CONDITIONS if split == "validation" else list(CONDITIONS)
        chosen = available if condition is None else [condition] if isinstance(condition, str) else list(condition)
        if not self.training and (not chosen or any(c not in available for c in chosen)):
            raise ValueError("Requested condition is not part of this evaluation split")
        total = 0
        for entry in self.entries:
            if self.training:
                path = self.root / entry["path"]
                rows = np.arange(entry["rows"], dtype="i4")
            else:
                path = self.root / "processed" / split / Path(entry["path"]).name
                rows = np.asarray([i * len(available) + available.index(c) for i in range(entry["rows"]) for c in chosen], dtype="i4")
            if not path.exists():
                raise FileNotFoundError(f"Dataset generation is not finished: {path}")
            self.paths.append(path)
            self.row_maps.append(rows)
            total += len(rows)
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _handle(self, path):
        if os.getpid() != self._pid:
            self._handles = OrderedDict()
            self._pid = os.getpid()
        key = str(path)
        if key in self._handles:
            return self._touch(key)
        handle = h5py.File(path, "r")
        if not handle.attrs.get("complete", False):
            handle.close()
            raise ValueError("Incomplete V3 shard")
        self._handles[key] = handle
        while len(self._handles) > self.max_open_files:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def _touch(self, key):
        handle = self._handles.pop(key)
        self._handles[key] = handle
        return handle

    def raw_sample(self, index, epoch=None):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = bisect_right(self.ends, index)
        previous = self.ends[file_index - 1] if file_index else 0
        row = int(self.row_maps[file_index][index - previous])
        data = read_row(self._handle(self.paths[file_index]), row)
        if self.training:
            selected_epoch = self.epoch if epoch is None else int(epoch)
            condition, spec = training_condition(data["base_id"], index, selected_epoch, self.training_mode)
            data = corrupt_base(data, condition, spec=spec, epoch=selected_epoch)
        return data

    def __getitem__(self, index):
        epoch, number = index if isinstance(index, tuple) else (self.epoch, index)
        data = self.raw_sample(number, epoch)
        inp, gt = data["input_xyz"], data["gt_xyz"]
        if self.normalize:
            inp, gt, center, scale = input_transform(inp, gt)
        else:
            center, scale = np.zeros(3, dtype="f4"), 1.0
        rotation = np.eye(3, dtype="f4")
        if self.rotate:
            rng = np.random.default_rng(seed(data["base_id"], "rotation", epoch))
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            rotation = q.astype("f4")
            inp, gt = inp @ rotation.T, gt @ rotation.T
        result = {"input_xyz": inp.astype("f4"), "gt_xyz": gt.astype("f4"),
                  "sample_id": data["sample_id"], "base_id": data["base_id"], "object_id": data["object_id"],
                  "condition": data["condition"], "prediction_center": center, "prediction_scale": np.float32(scale),
                  "prediction_rotation": rotation, "canonical_center": data["center"], "canonical_scale": data["scale"]}
        if self.features:
            result["input_features"] = input_geometry_features(result["input_xyz"])
        if self.include_supervision:
            supervision = {}
            for key in SUPERVISION_KEYS:
                value = data[key].copy()
                if key == "clean_input_xyz":
                    value = ((value - center) / scale) @ rotation.T
                elif key == "denoise_offset":
                    value = (value / scale) @ rotation.T
                elif key in {"clean_input_normals", "gt_normals"}:
                    value = value @ rotation.T
                elif key == "hole_boundary_distance_input":
                    value = value / scale
                supervision[key] = value.astype("f4") if value.dtype.kind == "f" else value
            result["supervision"] = supervision
        if self.reference_metadata:
            result["reference"] = {key: data[key] for key in ["mesh_path", "mesh_sha256", "geometry_id", "region_spec", "quality"]}
            result["reference"]["mesh_path"] = self.source_paths.resolve(data["mesh_path"])
        return result

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def __del__(self):
        if hasattr(self, "_handles"):
            self.close()


class EpochSampler:
    """Duck-typed torch Sampler, safe with persistent multiprocessing workers."""
    def __init__(self, dataset, *, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        order = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.default_rng(seed("sampler", self.epoch)).shuffle(order)
        return iter((self.epoch, int(index)) for index in order)


class OfficialPU1KDataset:
    """Unmodified public 2048/8192 coordinate pairs, normalized from input only."""
    def __init__(self, root=DEFAULT_ROOT, *, include_reference_metadata=False):
        root = Path(root)
        self.source_paths = SourcePaths(root)
        self.entries = json.loads((root / "metadata/official_benchmark.json").read_text(encoding="utf-8"))["pairs"]
        self.reference_metadata = include_reference_metadata
        split = json.loads((root / "metadata/dataset_splits.json").read_text(encoding="utf-8"))
        self.meshes = {record["object_id"].split("/", 1)[1]: record for record in split["pu1k_test"]}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        input_path = self.source_paths.resolve(entry["input_path"])
        target_path = self.source_paths.resolve(entry["target_path"])
        raw_input = np.loadtxt(input_path, dtype="f8")[:, :3]
        raw_target = np.loadtxt(target_path, dtype="f8")[:, :3]
        if raw_input.shape != (2048, 3) or raw_target.shape != (8192, 3):
            raise ValueError("Official point counts differ from the declared protocol")
        points, target, center, scale = input_transform(raw_input, raw_target)
        name = Path(input_path).stem
        result = {"input_xyz": points, "gt_xyz": target, "sample_id": entry["sample_id"],
                  "object_id": "pu1k/" + name, "base_id": entry["sample_id"], "condition": "official_A0",
                  "prediction_center": np.zeros(3, dtype="f4"), "prediction_scale": np.float32(1),
                  "prediction_rotation": np.eye(3, dtype="f4"), "canonical_center": center, "canonical_scale": scale}
        if self.reference_metadata:
            record = self.meshes[name]
            result["reference"] = {"mesh_path": self.source_paths.resolve(record["path"]), "mesh_sha256": record["sha256"],
                                   "geometry_id": record["geometry_id"]}
        return result
