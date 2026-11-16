"""Regression checks for budgets, geometry, supervision and stage isolation."""
import os
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
from dataclasses import asdict
from unittest.mock import patch
os.environ["V3_FPS_BACKEND"] = "torch"
os.environ["PUFM_POINTOPS"] = "torch"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"D2"/"PUFM-main"))
sys.path.insert(1, str(ROOT.parent))
import torch
from torch import nn
from geometry_m2.geometry_features import geometry_feature, local_regions
from geometry_m2.allocation import allocate, candidate_pools, initialize, quota
from geometry_m2.single_m2 import Config, SingleM2PUFM
from geometry_m2.training import gain_objective, create_model
from geometry_m2.teacher import generator_signature, build_teacher
from baseline_validation import BaselineValidation, atomic_save, digest_file
from models.adaptive_ops import assign_regions, aligned_target


class TinyFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = nn.Conv1d(3, 3, 1)

    def forward(self, points, initial, time):
        return self.layer(points)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        cls.points = torch.rand(2, 256, 3)*2-1
        cls.points[..., 2] = .15*cls.points[..., 0].square()+.08*cls.points[..., 1].square()

    def model(self, mode="D2"):
        with patch("geometry_m2.single_m2.PUFM_w_attn", TinyFlow):
            return SingleM2PUFM(Config(mode=mode))

    def test_geometry_definitions_and_degeneracy(self):
        region = local_regions(self.points)
        for mode in ("D1", "D2", "D3"):
            value, valid = geometry_feature(region, mode)
            self.assertTrue(valid.any())
            self.assertTrue(torch.isfinite(value).all())
            if mode != "D1":
                self.assertLess(float(value[valid].mean()), 1e-4)
        line = self.points.clone()
        line[..., 1:] = 0
        for mode in ("D1", "D2", "D3"):
            _, valid = geometry_feature(local_regions(line), mode)
            self.assertFalse(valid.any())

    def test_capacity_unique_ownership_and_nested_prefixes(self):
        model = self.model()
        state = model.encode(self.points)
        budget = allocate(model, state, "uniform")
        self.assertTrue((budget.sum(-1)==768).all())
        self.assertTrue((budget<=state["capacities"]).all())
        original = initialize(state, budget, 4)
        self.assertEqual(tuple(original["initial"].shape), (2, 1024, 3))
        self.assertTrue(torch.equal(original["initial"][:, :256], self.points))
        owners = assign_regions(state["pools"].flatten(1, 2), state["regions"]["centers"]).reshape(2, 16, 64)
        for batch in range(2):
            for index in range(16):
                count = int(state["capacities"][batch, index])
                self.assertTrue((owners[batch, index, :count]==index).all())
        changed = budget.clone()
        for batch in range(2):
            receiver = int((budget[batch]+8 <= state["capacities"][batch]).nonzero()[0])
            donor = next(i for i in range(16) if i != receiver and budget[batch, i] >= 12)
            changed[batch, receiver] += 8
            changed[batch, donor] -= 8
        alternative = initialize(state, changed, 4)
        self.assertFalse(torch.equal(original["initial"], alternative["initial"]))
        self.assertTrue(torch.equal(original["initial"][:, :256], alternative["initial"][:, :256]))
        with self.assertRaises(ValueError):
            quota(torch.ones(1, 8), 768, 4, torch.full((1, 8), 64))
        with self.assertRaises(ValueError):
            initialize(state, budget+1, 4)

    def test_gain_gradient_and_frozen_generator(self):
        model = self.model()
        model.requires_grad_(False)
        model.m2.requires_grad_(True)
        region = local_regions(self.points)
        record = dict(input=self.points, budget=torch.full((2, 16), 48), receiver=torch.tensor([[0], [0]]),
                      donor=torch.tensor([[1], [1]]), delta=8, improvement=torch.tensor([[.001], [-.001]]))
        loss, _, _ = gain_objective(model, region, record, 0, "cpu", .1)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.m2.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in model.flow.parameters()))

    def test_sparse_region_replenishes_unique_candidates(self):
        points = self.points[:1].clone()
        points[0, -3:] = torch.tensor([[3., 0., 0.], [10., 0., 0.], [10., .1, 0.]])
        model = self.model()
        region = local_regions(points)
        midpoint_pools, midpoint_caps = candidate_pools(points, region, 64, minimum=0, total=768)
        sparse = midpoint_caps[0] < 4
        self.assertTrue(sparse.any())
        state = model.encode(points)
        expected = midpoint_caps.clamp_min(4)
        self.assertTrue(torch.equal(state["capacities"], expected))
        for index in range(16):
            count = int(midpoint_caps[0, index])
            torch.testing.assert_close(state["pools"][0, index, :count], midpoint_pools[0, index, :count])
        budget = allocate(model, state, "uniform")
        cloud = initialize(state, budget, 4)["initial"]
        self.assertEqual(len(torch.unique(cloud[0], dim=0)), 1024)
        owners = torch.arange(16).repeat_interleave(budget[0])
        distances = torch.cdist(cloud[:, 256:], region["centers"], compute_mode="donot_use_mm_for_euclid_dist")[0]
        torch.testing.assert_close(distances.gather(1, owners[:, None]).squeeze(1), distances.amin(1), atol=1e-6, rtol=0)

        # The same isolated region also forces the global-shortfall branch
        # when the requested total reaches the sum of all upper bounds.
        _, full_caps = candidate_pools(points, region, 64, minimum=4, total=1024)
        self.assertTrue(torch.equal(full_caps, torch.full_like(full_caps, 64)))

    def test_current_initial_matching_condition_gradient_and_repeat(self):
        model = self.model()
        state = model.prepare(self.points)
        target = torch.randn(2, 1024, 3)
        matched = aligned_target(state["initial"], target)
        self.assertEqual(tuple(matched.shape), (2, 1024, 3))
        t = torch.full((2,), .5)
        velocity = model.velocity((state["initial"]+matched)/2, t, state)
        (velocity-(matched-state["initial"])).square().mean().backward()
        self.assertGreater(float(model.conditioner[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.flow.layer.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.m2.parameters()))
        model.eval()
        with torch.no_grad():
            first = model.integrate(state)
            second = model.integrate(state)
        self.assertTrue(torch.equal(first, second))

    def test_m2_scores_control_allocation_and_output(self):
        model = self.model()
        state = model.encode(self.points)
        state["valid"].fill_(True)
        uniform = allocate(model, state, "uniform")
        def gains(features, counts, delta):
            result = torch.ones_like(counts, dtype=torch.float)
            result[:, 0] = 100
            return result
        with patch.object(model.m2.gain, "batch_gain", side_effect=gains):
            learned = allocate(model, state)
        self.assertFalse(torch.equal(uniform, learned))
        self.assertTrue((learned.sum(-1)==768).all())
        first = model.integrate(initialize(state, uniform, 4))
        second = model.integrate(initialize(state, learned, 4))
        self.assertFalse(torch.equal(first, second))

    def test_teacher_signature_and_training_split(self):
        model = self.model()
        signature = generator_signature(model)
        with torch.no_grad():
            next(model.m2.parameters()).add_(1)
        self.assertEqual(signature, generator_signature(model))
        with torch.no_grad():
            next(model.conditioner.parameters()).add_(1)
        self.assertNotEqual(signature, generator_signature(model))
        with self.assertRaises(ValueError):
            build_teacher(model, SimpleNamespace(split="validation"), SimpleNamespace())

    def test_variant_checkpoint_loading(self):
        model = self.model("D2")
        saved = dict(protocol="PUFM-single-M2-v1", stage="gain", configuration=asdict(model.config), model=model.state_dict())
        args = SimpleNamespace(resume=None, checkpoint="unused", flow_steps=None, device="cpu", allocator=None)
        with patch("geometry_m2.training.read_checkpoint", return_value=saved), patch("geometry_m2.single_m2.PUFM_w_attn", TinyFlow):
            loaded, _ = create_model(args, "D2")
            self.assertEqual(loaded.config.mode, "D2")
            with self.assertRaises(ValueError):
                create_model(args, "D1")

    def test_best_resume_preserves_hash_and_repairs_interrupted_save(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(output=output, root=output, validation_limit=1, device="cpu", resume=None)
            dataset = SimpleNamespace(close=lambda: None)
            with patch("baseline_validation.modelnet_validation", return_value=(dataset, [0], {"object": "shard"}, output/"ids.json")), patch("baseline_validation.digest_file", return_value="identity"):
                selection = BaselineValidation(args)
                best = dict(epoch=0, epoch_one_based=1, value=.1, protocol=selection.protocol)
                saved = dict(epoch=0, best_validation=best, model={"weight": torch.ones(2)})
                atomic_save(saved, output/"last.pt")
                atomic_save(saved, output/"best.pt")
                before = digest_file(output/"best.pt")
                args.resume = output/"last.pt"
                BaselineValidation(args, saved)
                self.assertEqual(before, digest_file(output/"best.pt"))
                (output/"best.pt").unlink()
                BaselineValidation(args, saved)
                self.assertEqual(torch.load(output/"best.pt", weights_only=False)["best_validation"], best)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_geometry_cpu_cuda_agreement(self):
        cpu = local_regions(self.points)
        gpu = local_regions(self.points.cuda())
        torch.testing.assert_close(cpu["centers"], gpu["centers"].cpu())
        for mode in ("D1", "D2", "D3"):
            a, valid_a = geometry_feature(cpu, mode)
            b, valid_b = geometry_feature(gpu, mode)
            self.assertTrue(torch.equal(valid_a, valid_b.cpu()))
            torch.testing.assert_close(a, b.cpu(), rtol=.02, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
