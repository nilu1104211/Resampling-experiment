"""Test the actual source normalization/filter/RNG contract without training."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
os.environ["V3_CODE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "compare-change"))
sys.path.insert(0, str(ROOT / "compare-change/spu-imr-main"))
spec = importlib.util.spec_from_file_location("spu_adapter_test", ROOT / "compare-change/spu-imr-main/v3_adapter.py")
spu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spu)


class FakeRecovery(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))

    def forward(self, points, device, idxy=0, vis=False):
        if vis:
            torch.randint(0, points.shape[1], (1,), dtype=torch.long)
            return points + idxy * .0003
        return points.square().mean() + self.weight.square()


def make_adapter(profile="native"):
    args = SimpleNamespace(spu_profile=profile, spu_groups=None, spu_transport="hungarian", spu_passes=8,
                           spu_schedule_epochs=6000, spu_warmup_epochs=50)
    with patch.object(spu, "Network_Model", return_value=FakeRecovery()):
        return spu.Adapter(args).eval()


class SPUNativeTests(unittest.TestCase):
    def tearDown(self):
        spu.utils.configure_transport("hungarian")

    def test_native_and_legacy_group_profiles_are_explicit(self):
        self.assertEqual(make_adapter().args.spu_groups, 256)
        self.assertEqual(make_adapter("legacy-patch256").args.spu_groups, 64)

    def test_old_checkpoint_requires_legacy_profile(self):
        import v3_common
        model = make_adapter()
        model.args.model_name = "spu-imr"
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            checkpoint = Path(directory) / "old.pt"
            torch.save({"protocol": "V3", "args": {"model_name": "spu-imr", "spu_groups": 64},
                        "model": model.state_dict()}, checkpoint)
            with self.assertRaisesRegex(ValueError, "spu_profile"):
                v3_common.load_weights(model, checkpoint, model.args)
            legacy = make_adapter("legacy-patch256")
            legacy.args.model_name = "spu-imr"
            v3_common.load_weights(legacy, checkpoint, legacy.args)

    def test_native_training_ignores_clean_gt_and_requires_real_observations(self):
        model = make_adapter()
        x = torch.randn(1, 1024, 3)
        a = model.training_loss(x, torch.zeros(1, 1024, 3))
        b = model.training_loss(x, torch.full((1, 1024, 3), 12345.))
        self.assertTrue(torch.equal(a, b))
        a.backward()
        self.assertGreater(model.model.weight.grad.abs().item(), 0)
        with self.assertRaises(ValueError):
            model.training_loss(x[:, :256], x)

    def test_inference_matches_source_bbox_filter_and_seeded_fps(self):
        torch.manual_seed(12)
        points = torch.randn(1, 256, 3)
        model = make_adapter()
        state = torch.get_rng_state().clone()
        with patch.dict(os.environ, V3_FPS_BACKEND="torch"), torch.inference_mode():
            actual = model.predict(points, 1024)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            low, high = points.amin(1, keepdim=True), points.amax(1, keepdim=True)
            center, scale = (low + high) * .5, (high - low).amax()
            with torch.random.fork_rng(devices=[]):
                candidates = []
                for i in range(8):
                    torch.manual_seed(i)
                    candidates.append(model.model((points - center) / scale, "cpu", idxy=i, vis=True))
                cloud = torch.cat(candidates, 1)
                average = torch.cdist(cloud, cloud).topk(30, largest=False).values.mean(-1)
                filtered = cloud[:, average[0] < average.mean() * 2]
                expected = spu.utils.fps(filtered * scale + center, 1024)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(model.last_diagnostics["deduplicated"])

    def test_cosine_recipe_starts_at_source_warmup_rate(self):
        model = make_adapter()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
        scheduler = model.make_scheduler(optimizer, model.args)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        scheduler.step(50)
        self.assertGreater(optimizer.param_groups[0]["lr"], .00099)
        scheduler.step(5999)
        self.assertLess(optimizer.param_groups[0]["lr"], 1.01e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_source_emd_kernel_uses_original_euclidean_cost_and_gradient(self):
        from source_emd import EMDLoss
        # >32 batches also exercises the source kernel's grid-stride loop.
        a = torch.zeros(40, 1, 3, device="cuda", requires_grad=True)
        b = torch.tensor([3., 4., 0.], device="cuda").expand(40, 1, 3).contiguous().requires_grad_()
        cost = EMDLoss()(a, b)
        torch.testing.assert_close(cost, torch.full_like(cost, 5.))
        cost.sum().backward()
        expected = torch.tensor([-.6, -.8, 0.], device="cuda").expand_as(a)
        torch.testing.assert_close(a.grad, expected)
        torch.testing.assert_close(b.grad, -expected)


if __name__ == "__main__":
    unittest.main()
