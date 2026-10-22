import inspect
import unittest
import torch
from types import SimpleNamespace
from gain_teacher import write_cache
from models.adaptive_wrapper import AdaptiveConfig, AdaptivePUFM
from models.adaptive_ops import fps_ids
from models.budget_allocator import quota, allocate, initialize
from models.marginal_gain import GainNet
from models.adaptive_losses import gain_loss


class Contracts(unittest.TestCase):
    def test_rounding_bounds_and_total(self):
        for k in [16, 32, 64]:
            for weights in [torch.zeros(2, k), torch.ones(2, k), torch.exp(torch.randn(2, k) * 10)]:
                counts = quota(weights, 768, 4, 64)
                self.assertTrue(torch.all(counts.sum(-1) == 768))
                self.assertTrue(torch.all((counts >= 4) & (counts <= 64)))

    def test_nested_pool_swap_preserves_other_coordinates(self):
        anchors = torch.randn(1, 8, 3)
        pool = torch.randn(1, 4, 16, 3)
        before = torch.tensor([[8, 8, 8, 8]])
        after = torch.tensor([[12, 4, 8, 8]])
        x, owners = initialize(anchors, pool, before)
        y, new_owners = initialize(anchors, pool, after)
        self.assertEqual(x.shape, y.shape)
        for region in [2, 3]:
            self.assertTrue(torch.equal(x[owners == region], y[new_owners == region]))

    def test_gain_is_batch_sum_and_has_nonzero_gradients(self):
        network = GainNet()
        features = torch.randn(2, 4, 76)
        counts = torch.full((2, 4), 16)
        summed = sum(network.batch_gain(features, counts + step, 1) for step in range(8))
        self.assertTrue(torch.allclose(summed, network.batch_gain(features, counts, 8), atol=1e-6))
        loss = (network.transfer(features, counts, torch.tensor([0, 1]), torch.tensor([2, 3]), 8) - torch.tensor([.2, -.1])).square().mean()
        loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in network.parameters()), 0)

    def test_inference_interface_cannot_receive_gt(self):
        parameters = inspect.signature(AdaptivePUFM.forward).parameters
        self.assertEqual(set(parameters), {"self", "points", "output_size", "return_diagnostics"})

    def test_degenerate_fps_is_finite_unique_indices(self):
        ids = fps_ids(torch.zeros(2, 32, 3), 32)
        for row in ids:
            self.assertEqual(len(row.unique()), 32)

    def test_infeasible_budget_rejected(self):
        with self.assertRaises(ValueError):
            quota(torch.ones(1, 32), 100, 4, 64)

    def test_hard_budget_has_no_fictitious_gain_gradient(self):
        network = GainNet()
        feature = torch.randn(1, 4, 76)
        observable = torch.rand(1, 4, 8)
        budget = allocate({"gain_features": feature, "observable": observable}, network, 32, 4, 16, 4, 8)
        anchors = torch.randn(1, 8, 3, requires_grad=True)
        pool = torch.randn(1, 4, 16, 3, requires_grad=True)
        points, _ = initialize(anchors, pool, budget)
        points.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in network.parameters()))

    def test_validation_cannot_become_a_training_teacher(self):
        with self.assertRaisesRegex(ValueError, "ONLY use split=train"):
            write_cache(None, SimpleNamespace(split="validation"), "unused.npz", [], [])


if __name__ == "__main__":
    unittest.main()
