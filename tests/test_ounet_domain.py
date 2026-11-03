"""OUNet preflight must cover all planned observations without altering data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_resampling.core import input_transform
from adaptive_resampling.v3_core import CONDITIONS, corrupt_base, training_condition
from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
from adaptive_resampling.v3_pipeline import DEFAULT, write_rows
from tools.check_ounet_domain import check_domain, domain_radius_bound
from test_v3_data import make_surface_base


class OUNetDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = make_surface_base()

    def test_fast_coordinates_match_full_protocol_for_fixed_and_online_conditions(self):
        cases = [(condition, None, None) for condition in CONDITIONS]
        cases += [(*training_condition(self.base["base_id"], 20672, epoch), epoch) for epoch in range(60)]
        for condition, spec, epoch in cases:
            with self.subTest(condition=condition, epoch=epoch):
                full = corrupt_base(self.base, condition, spec=spec, epoch=epoch)
                fast = corrupt_base(self.base, condition, spec=spec, epoch=epoch, coordinates_only=True)
                for key in ("input_xyz", "gt_xyz"):
                    np.testing.assert_array_equal(full[key], fast[key])

    def test_bound_covers_shared_normalization_and_arbitrary_rotations(self):
        rng = np.random.default_rng(5)
        observations = (rng.normal(size=(256, 3)) * .2 + [.5, -.7, .1]).astype("f4")
        gt = (rng.normal(size=(1024, 3)) * 2).astype("f4")
        original = observations.copy()
        bound = domain_radius_bound(observations, float(np.linalg.norm(gt.astype("f8"), axis=1).max()))
        normalized, target, _, _ = input_transform(observations, gt)
        for _ in range(100):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotation = rotation.astype("f4")
            self.assertLess(float(np.abs(normalized @ rotation.T).max()), bound)
            self.assertLess(float(np.abs(target @ rotation.T).max()), bound)
        np.testing.assert_array_equal(original, observations)
        for points, radius in ((np.zeros((256, 3)), 1.), (observations * np.nan, 1.), (observations, np.inf)):
            with self.assertRaises(ValueError):
                domain_radius_bound(points, radius)

    def test_checker_uses_exact_shared_selection_epochs_and_rejects_unsafe_scale(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            config = dict(DEFAULT, output_root=str(root))
            base = dict(self.base, gt_xyz=self.base["gt_xyz"] * 8)
            write_rows(root / "train_bases/object.h5", [base] * 4, config, "independent_surface_base")
            (root / "metadata").mkdir()
            (root / "metadata/base_manifest.json").write_text(json.dumps({"config_digest": "test", "groups": {
                "train": [{"path": "train_bases/object.h5", "rows": 4, "object_id": "unit/shape"}]}}))
            bad = check_domain(root, epochs=5, octree_scale=.5, train_patches_per_object=2)
            good = check_domain(root, epochs=5, octree_scale=.01, train_patches_per_object=2)
            self.assertEqual(bad["status"], "failed_conservative_bound")
            self.assertGreater(bad["conservative_violations"], 0)
            self.assertEqual(good["status"], "passed")
            self.assertEqual(good["observations_checked"], 10)
            ds = AdaptiveResamplingDataset(root, include_supervision=False, train_patches_per_object=2)
            try:
                self.assertEqual(good["training_profile"], ds.training_profile)
                for epoch in range(5):
                    for number in range(len(ds)):
                        sample = ds[(epoch, number)]
                        self.assertLess(float(np.abs(sample["gt_xyz"]).max()) * .01, 1)
            finally:
                ds.close()


if __name__ == "__main__":
    unittest.main()
