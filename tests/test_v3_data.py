import json
from pathlib import Path
import pickle
import tempfile
import sys
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from adaptive_resampling.v3_core import (CONDITIONS, VALIDATION_CONDITIONS, prepare_base,
                                         corrupt_base, training_condition)
from adaptive_resampling.v3_pipeline import DEFAULT, geometry_fingerprint, object_split, write_rows
from adaptive_resampling.v3_dataset import (AdaptiveResamplingDataset, EpochSampler,
                                            input_geometry_features, inverse_prediction, prepare_inference)
from adaptive_resampling.v3_metrics import TriangleBVH, point_triangle_squared, evaluate_sample


def make_surface_base():
    rng = np.random.default_rng(77)
    xy = rng.uniform(-1, 1, (4096, 2))
    xyz = np.column_stack([xy, 0.2 * xy[:, 0] ** 2 + 0.1 * xy[:, 1] ** 2])
    normals = np.column_stack([-0.4 * xy[:, 0], -0.2 * xy[:, 1], np.ones(len(xy))])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return prepare_base(xyz[1024:], xyz[:1024], normals[1024:], normals[:1024], base_id="unit/shape/patch_000",
                        object_id="unit/shape", category="unit", mesh_path="synthetic.off", mesh_sha256="unit",
                        geometry_id="unit", gt_face_index=np.zeros(1024, dtype="i4"),
                        gt_barycentric=np.tile([1, 0, 0], (1024, 1)), gt_sample_index=np.arange(1024),
                        pool_sample_index=np.arange(1024, 4096))


class V3GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = make_surface_base()

    def test_every_condition_is_independent_and_labels_match(self):
        for name in CONDITIONS:
            with self.subTest(name=name):
                result = corrupt_base(self.base, name)
                self.assertEqual(result["input_xyz"].shape, (256, 3))
                self.assertEqual(result["gt_xyz"].shape, (1024, 3))
                self.assertEqual(result["quality"]["input_gt_exact_overlap"], 0)
                self.assertEqual(result["quality"]["clean_hole_intrusions"], 0)
                np.testing.assert_array_equal(result["gt_xyz"], self.base["gt_xyz"])
                np.testing.assert_array_equal((result["input_xyz"] != result["clean_input_xyz"]).any(-1), result["noise_mask_input"].astype(bool))
                np.testing.assert_allclose(result["input_xyz"] + result["denoise_offset"], result["clean_input_xyz"], atol=1e-7)
                self.assertEqual(len(np.unique(result["selected_pool_index"])), 256)
                self.assertTrue(np.isfinite(result["reliability_target_input"]).all())

    def test_paired_holes_density_and_noise_strengths(self):
        holes = [corrupt_base(self.base, f"C{i}_hole_{p}")["hole_mask_gt"] for i, p in [(1, "05"), (2, "10"), (3, "20")]]
        self.assertTrue(np.all(holes[0] <= holes[1]))
        self.assertTrue(np.all(holes[1] <= holes[2]))
        ratios = [corrupt_base(self.base, c)["quality"]["sparse_density_ratio_actual"] for c in ["B1_sparse_light", "B2_sparse_mid", "B3_sparse_heavy"]]
        self.assertGreater(ratios[0], ratios[1])
        self.assertGreater(ratios[1], ratios[2])
        low = corrupt_base(self.base, "D1_noise_002")["denoise_offset"]
        high = corrupt_base(self.base, "D3_noise_010")["denoise_offset"]
        np.testing.assert_allclose(high, 5 * low, atol=3e-7)

    def test_unseen_noise_is_not_sampled_in_training(self):
        for index in range(5):
            for epoch in range(100):
                _, spec = training_condition(self.base["base_id"], index, epoch)
                self.assertNotEqual(spec.get("distribution"), "laplace")
        self.assertEqual([training_condition("shape", 0, e)[1]["kind"] for e in range(5)], list(range(5)))

    def test_geometric_identity_and_group_split(self):
        rng = np.random.default_rng(9)
        points = rng.normal(size=(100, 3))
        other = (points * 7 + [10, -4, 6])[rng.permutation(100)]
        self.assertEqual(geometry_fingerprint(points), geometry_fingerprint(other))
        records = [{"geometry_id": f"g{i}", "object_id": f"object{i}", "category": "a"} for i in range(20)]
        records += [dict(records[0], object_id="alias")]
        train, val = object_split(records)
        self.assertFalse({r["geometry_id"] for r in train} & {r["geometry_id"] for r in val})

    def test_observable_features_and_real_inference_do_not_require_GT(self):
        x = corrupt_base(self.base, "D2_noise_005")["input_xyz"]
        features = input_geometry_features(x)
        self.assertEqual(features.shape, (256, 8))
        self.assertTrue(np.isfinite(features).all())
        sample = prepare_inference(x)
        self.assertNotIn("gt_xyz", sample)
        np.testing.assert_allclose(inverse_prediction(sample["input_xyz"], sample), x, atol=2e-7)

    def test_exact_triangle_distance_interior_edge_and_vertex(self):
        triangles = np.asarray([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
        for point, expected in [([0.2, 0.2, 3], 9), ([-2, 0, 0], 4), ([0.5, -2, 0], 4)]:
            self.assertAlmostEqual(point_triangle_squared(np.asarray(point), triangles)[0], expected)

    def test_mesh_BVH_matches_exhaustive_triangle_distances(self):
        rng = np.random.default_rng(902)
        triangles, points = rng.normal(size=(100, 3, 3)), rng.normal(size=(20, 3))
        distances, _ = TriangleBVH(triangles, leaf_size=4).query(points)
        exact = [np.sqrt(point_triangle_squared(point, triangles).min()) for point in points]
        np.testing.assert_allclose(distances, exact, atol=1e-12)


class V3DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        cls.root = Path(cls.temporary.name)
        cls.base = make_surface_base()
        config = dict(DEFAULT, output_root=str(cls.root))
        train_path = cls.root / "train_bases/shape.h5"
        write_rows(train_path, [cls.base], config, "independent_surface_base")
        fixed_path = cls.root / "processed/validation/shape.h5"
        write_rows(fixed_path, [corrupt_base(cls.base, c) for c in VALIDATION_CONDITIONS], config, "fixed_evaluation")
        manifest = {"groups": {"train": [{"path": "train_bases/shape.h5", "rows": 1}],
                               "validation": [{"path": "validation_bases/shape.h5", "rows": 1}]}}
        (cls.root / "metadata").mkdir()
        (cls.root / "metadata/base_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_online_epochs_replay_and_rotation_targets_are_aligned(self):
        dataset = AdaptiveResamplingDataset(self.root, include_geometry_features=True)
        sampler = EpochSampler(dataset)
        sampler.set_epoch(3)
        sample = dataset[next(iter(sampler))]
        repeated = dataset[(3, 0)]
        np.testing.assert_array_equal(sample["input_xyz"], repeated["input_xyz"])
        targets = sample["supervision"]
        np.testing.assert_allclose(sample["input_xyz"] + targets["denoise_offset"], targets["clean_input_xyz"], atol=3e-7)
        np.testing.assert_allclose(inverse_prediction(sample["gt_xyz"], sample), self.base["gt_xyz"], atol=3e-7)
        np.testing.assert_allclose(np.linalg.norm(targets["gt_normals"], axis=1), 1, atol=3e-7)
        self.assertNotIn("gt_normals", sample)
        dataset.close()

    def test_fixed_validation_is_epoch_invariant_and_filters_conditions(self):
        dataset = AdaptiveResamplingDataset(self.root, "validation", condition="C2_hole_10")
        self.assertEqual(len(dataset), 1)
        first, second = dataset[(0, 0)], dataset[(999, 0)]
        self.assertNotIn("supervision", first)
        np.testing.assert_array_equal(first["input_xyz"], second["input_xyz"])
        self.assertEqual(first["condition"], "C2_hole_10")
        full = AdaptiveResamplingDataset(self.root, "validation", condition="C2_hole_10", train_patches_per_object=0)
        self.assertEqual(len(full), len(dataset))
        np.testing.assert_array_equal(full[0]["input_xyz"], first["input_xyz"])
        np.testing.assert_array_equal(full[0]["gt_xyz"], first["gt_xyz"])
        full.close()
        dataset.close()

    def test_shared_spu_reads_identical_inputs_gt_and_training_profile(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compare-change"))
        import v3_common
        values = dict(root=self.root, condition=None, training_mode="mixed", train_patches_per_object=32)
        spu = v3_common.dataset(SimpleNamespace(**values, model_name="spu-imr", spu_profile="shared-v3"), "train")
        pufmpp = v3_common.dataset(SimpleNamespace(**values, model_name="pufm++"), "train")
        try:
            self.assertEqual(spu.training_profile, pufmpp.training_profile)
            for epoch in range(5):
                a, b = spu[(epoch, 0)], pufmpp[(epoch, 0)]
                self.assertEqual(a["input_xyz"].shape, (256, 3))
                self.assertEqual(a["gt_xyz"].shape, (1024, 3))
                np.testing.assert_array_equal(a["input_xyz"], b["input_xyz"])
                np.testing.assert_array_equal(a["gt_xyz"], b["gt_xyz"])
                self.assertEqual(a["sample_id"], b["sample_id"])
            with self.assertRaisesRegex(ValueError, "excluded from shared training"):
                v3_common.dataset(SimpleNamespace(**values, model_name="spu-imr", spu_profile="native"), "train")
        finally:
            spu.close()
            pufmpp.close()

    def test_cached_files_are_not_pickled_into_workers(self):
        dataset = AdaptiveResamplingDataset(self.root)
        dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(len(restored._handles), 0)
        np.testing.assert_array_equal(dataset[0]["input_xyz"], restored[0]["input_xyz"])
        dataset.close()
        restored.close()

    def test_batched_prediction_inverse(self):
        dataset = AdaptiveResamplingDataset(self.root)
        samples = [dataset[(0, 0)], dataset[(3, 0)]]
        batch = {key: np.stack([s[key] for s in samples]) for key in ["prediction_center", "prediction_scale", "prediction_rotation"]}
        canonical = inverse_prediction(np.stack([s["gt_xyz"] for s in samples]), batch)
        np.testing.assert_allclose(canonical, np.stack([self.base["gt_xyz"]] * 2), atol=3e-7)
        dataset.close()

    def test_shared_evaluation_and_fixed_budget(self):
        dataset = AdaptiveResamplingDataset(self.root, "validation", condition="C2_hole_10", include_supervision=True)
        sample = dataset[0]
        result = evaluate_sample(sample["gt_xyz"], sample)
        self.assertLess(result["cd_l1"], 1e-6)
        self.assertEqual(result["hole_coverage_fraction"], 1)
        self.assertEqual(result["f_score"], 1)
        with self.assertRaises(ValueError):
            evaluate_sample(sample["gt_xyz"][:100], sample)
        dataset.close()


if __name__ == "__main__":
    unittest.main()
