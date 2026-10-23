"""Parallel matching must preserve the original exact transport and gradients."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_resampling.v3_transport import (
    assignment_indices, batched_assignment_indices, configure_assignment_workers,
)


class TransportTests(unittest.TestCase):
    def tearDown(self):
        configure_assignment_workers(1)

    def test_parallel_matches_scipy_including_ties_and_batch_order(self):
        rng = np.random.default_rng(21)
        costs = rng.random((4, 96, 96), dtype=np.float32)
        costs[0] = 0  # Every permutation is optimal; tie handling must stay identical.
        costs[1] = np.round(costs[1], 1)
        costs[2] = np.eye(96, dtype=np.float32) * -10
        expected = np.stack([linear_sum_assignment(c)[1] for c in costs])
        for workers in (1, 2, 4, 2, 1):
            configure_assignment_workers(workers)
            for batch in (4, 1, 3):
                actual = batched_assignment_indices(costs[:batch])
                np.testing.assert_array_equal(actual, expected[:batch])
                for row in actual:
                    np.testing.assert_array_equal(np.sort(row), np.arange(96))

    def check_alignment_and_gradients(self, device):
        # Check both adapters against the old sequential implementation.
        comparison = ROOT / "compare-change"
        if not comparison.is_dir():
            comparison = ROOT / "network_adaptation/compare-change"
        sys.path.insert(0, str(comparison))
        from v3_common import assignment_target
        adaptive = ROOT / "our/PUFM-main"
        if not adaptive.is_dir():
            adaptive = ROOT / "adaptive_network/PUFM-main"
        spec = importlib.util.spec_from_file_location(
            "transport_test_adaptive_ops", adaptive / "models/adaptive_ops.py")
        ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ops)
        generator = torch.Generator().manual_seed(3)
        source = torch.randn(4, 48, 3, generator=generator).to(device)
        target = torch.randn(4, 48, 3, generator=generator).to(device)
        source[0] = 0
        target[0] = 0
        cost = torch.cdist(source, target).square().cpu().numpy()
        ids = torch.as_tensor(np.stack([linear_sum_assignment(c)[1] for c in cost]), device=device)
        batch = torch.arange(4, device=device)[:, None]

        def run(align):
            x, y = source.clone().requires_grad_(), target.clone().requires_grad_()
            aligned = align(x, y)
            loss = (x - aligned).square().sum(-1).mean()
            gradients = torch.autograd.grad(loss, (x, y))
            return aligned.detach(), loss.detach(), *gradients

        expected = run(lambda x, y: y[batch, ids])
        for workers in (1, 2, 4):
            configure_assignment_workers(workers)
            self.assertTrue(torch.equal(assignment_indices(source, target), ids))
            for align in (assignment_target, ops.aligned_target):
                for actual, wanted in zip(run(align), expected):
                    self.assertTrue(torch.equal(actual, wanted))

    def test_cpu_alignment_loss_and_both_coordinate_gradients(self):
        self.check_alignment_and_gradients("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_alignment_loss_and_both_coordinate_gradients(self):
        self.check_alignment_and_gradients("cuda")

    def test_bad_inputs_still_fail(self):
        for workers in (1, 4):
            configure_assignment_workers(workers)
            for cost in (np.zeros((0, 3, 3)), np.zeros((2, 3, 4)), np.zeros((3, 3)),
                         np.full((2, 3, 3), np.nan), np.full((2, 3, 3), np.inf)):
                with self.assertRaises(ValueError):
                    batched_assignment_indices(cost)
            with self.assertRaises(ValueError):
                assignment_indices(torch.zeros(2, 3, 3), torch.zeros(2, 4, 3))
        for workers in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                configure_assignment_workers(workers)


if __name__ == "__main__":
    unittest.main()
