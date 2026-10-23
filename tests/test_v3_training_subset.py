"""Training budgets must preserve objects, original indices and frozen evaluation."""
import copy
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset, EpochSampler
from adaptive_resampling.v3_training import training_selection, require_training_profile


def manifest():
    return {"config_digest": "original-v3-config", "groups": {
        "train": [{"path": f"train_bases/object_{i}.h5", "object_id": f"unit/object_{i}", "rows": 64}
                  for i in range(3)],
        "validation": [{"path": "validation_bases/object_val.h5", "rows": 4}],
    }}


class TrainingSubsetTests(unittest.TestCase):
    def test_default_keeps_32_unique_patches_from_every_object(self):
        maps, offsets, profile = training_selection(manifest())
        self.assertEqual(offsets, [0, 64, 128])
        self.assertEqual(profile["samples"], 96)
        self.assertEqual(profile["source_samples"], 192)
        self.assertEqual(profile["objects"], 3)
        for rows in maps:
            self.assertEqual(len(rows), 32)
            self.assertEqual(len(np.unique(rows)), 32)
            self.assertTrue(np.all(rows[:-1] < rows[1:]))
            self.assertGreater(rows[-1], 31)
        again, _, repeated = training_selection(manifest())
        self.assertEqual(profile, repeated)
        for a, b in zip(maps, again):
            np.testing.assert_array_equal(a, b)
        smaller, _, _ = training_selection(manifest(), 16)
        for small, large in zip(smaller, maps):
            self.assertTrue(set(small) <= set(large))

    def test_full_budget_and_short_objects_keep_all_stored_rows(self):
        for budget in (0, 64, 128):
            maps, _, profile = training_selection(manifest(), budget)
            self.assertEqual(profile["samples"], 192)
            for rows in maps:
                np.testing.assert_array_equal(rows, np.arange(64))
        short = manifest()
        short["groups"]["train"][0]["rows"] = 5
        maps, offsets, profile = training_selection(short)
        self.assertEqual(list(map(len, maps)), [5, 32, 32])
        self.assertEqual(offsets, [0, 5, 69])
        for bad in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                training_selection(manifest(), bad)

    def test_reader_preserves_original_corruption_indices_and_epoch_sampler(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            data = manifest()
            (root / "metadata/base_manifest.json").write_text(json.dumps(data), encoding="utf-8")
            for entry in data["groups"]["train"]:
                path = root / entry["path"]
                path.parent.mkdir(exist_ok=True)
                path.touch()
            ds = AdaptiveResamplingDataset(root)
            try:
                sampler = EpochSampler(ds)
                sampler.set_epoch(9)
                order = list(sampler)
                self.assertEqual(len(order), 96)
                self.assertEqual({epoch for epoch, _ in order}, {9})
                self.assertEqual(sorted(i for _, i in order), list(range(96)))
                restored = pickle.loads(pickle.dumps(ds))
                self.assertEqual(restored.training_profile, ds.training_profile)
                for index in (0, 31, 32, 48, 64, 95, -1):
                    positive = index % len(ds)
                    object_index, local = divmod(positive, 32)
                    expected = 64 * object_index + int(ds.row_maps[object_index][local])
                    self.assertEqual(ds.source_index(index), expected)
                    self.assertEqual(restored.source_index(index), expected)
                    with patch.object(ds, "_handle", return_value=None), \
                         patch("adaptive_resampling.v3_dataset.read_row", return_value={"base_id": "the-original-patch"}), \
                         patch("adaptive_resampling.v3_dataset.training_condition", return_value=("A0_normal", {})) as condition, \
                         patch("adaptive_resampling.v3_dataset.corrupt_base", side_effect=lambda data, *a, **kw: data):
                        ds.raw_sample(index, epoch=9)
                        condition.assert_called_once_with("the-original-patch", expected, 9, "mixed")
                restored.close()
            finally:
                ds.close()

    def test_resume_and_teacher_profiles_reject_changed_budget(self):
        data = manifest()
        _, _, profile32 = training_selection(data, 32)
        _, _, profile64 = training_selection(data, 0)
        require_training_profile({"training_profile": profile32}, profile32)
        require_training_profile({}, profile64)  # Legacy checkpoints used all rows.
        for saved in ({}, {"training_profile": profile64}):
            with self.assertRaises(ValueError):
                require_training_profile(saved, profile32)
        changed = copy.deepcopy(profile32)
        changed["selection_sha256"] = "different-patches"
        with self.assertRaises(ValueError):
            require_training_profile({"training_profile": changed}, profile32, context="Teacher cache")
        changed = dict(profile32, training_mode="clean")
        with self.assertRaises(ValueError):
            require_training_profile({"training_profile": changed}, profile32)

    def test_comparison_checkpoint_records_and_enforces_training_profile(self):
        import torch
        root = Path(__file__).resolve().parents[1]
        comparison = root / "compare-change"
        if not comparison.is_dir():
            comparison = root / "network_adaptation/compare-change"
        sys.path.insert(0, str(comparison))
        import v3_common

        class TinyDataset:
            def __init__(self, budget):
                self.training_profile = training_selection(manifest(), budget)[2]

            def __len__(self):
                return self.training_profile["samples"]

            def __getitem__(self, index):
                return {"input_xyz": torch.ones(3), "gt_xyz": torch.zeros(3)}

            def set_epoch(self, epoch):
                pass

            def close(self):
                pass

        class TinyAdapter(torch.nn.Linear):
            compatibility_keys = ()
            training_description = "checkpoint integration test"

            def training_loss(self, points, gt):
                return (self(points) - gt).square().mean()

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            args = SimpleNamespace(batch_size=32, workers=0, lr=1e-4, weight_decay=0., fused_optimizer=False,
                                   resume=None, checkpoint=None, output=directory, epochs=1, device="cpu",
                                   max_batches=0, log_every=1000, validation_samples=0,
                                   training_mode="mixed", model_name="unit", train_patches_per_object=32)
            ds = TinyDataset(32)
            with patch.object(v3_common, "dataset", return_value=ds), redirect_stdout(io.StringIO()):
                v3_common.train(TinyAdapter(3, 3), args)
            path = Path(directory) / "last.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["training_profile"], ds.training_profile)
            self.assertEqual(saved["args"]["train_patches_per_object"], 32)
            self.assertFalse(saved["result"]["partial_debug_run"])
            self.assertEqual(saved["result"]["batches"], 3)
            args.resume, args.epochs = str(path), 2
            with patch.object(v3_common, "dataset", return_value=ds), redirect_stdout(io.StringIO()):
                v3_common.train(TinyAdapter(3, 3), args)
            self.assertEqual(torch.load(path, map_location="cpu", weights_only=False)["epoch"], 1)
            with patch.object(v3_common, "dataset", return_value=TinyDataset(16)), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "same training patch selection"):
                    v3_common.train(TinyAdapter(3, 3), args)


if __name__ == "__main__":
    unittest.main()
