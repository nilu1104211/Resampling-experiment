import unittest

import torch

from adaptive_endpoint import adaptive_endpoint
from models.adaptive_ops import assign_regions
from models.adaptive_wrapper import AdaptivePUFM
from adaptive_v3 import preset


class Ours1Tests(unittest.TestCase):
    def test_adaptive_endpoint_obeys_quotas_or_falls_back_per_sample(self):
        centers = torch.tensor([[[-1., 0, 0], [1., 0, 0]]]).repeat(2, 1, 1)
        pool = torch.tensor([[[-1., 0, 0], [-.9, 0, 0], [1., 0, 0], [.9, 0, 0]],
                             [[-1., 0, 0], [-.9, 0, 0], [-.8, 0, 0], [-.7, 0, 0]]])
        state = dict(budget=torch.tensor([[1, 0], [1, 0]]),
                     regions=dict(centers=centers, anchor_counts=torch.ones(2, 2)))
        gt = torch.rand(2, 3, 3)
        target, used = adaptive_endpoint(state, pool, gt)
        self.assertEqual(used, 1)
        self.assertTrue(torch.equal(target[1], gt[1]))
        self.assertEqual(assign_regions(target, centers)[0].bincount().tolist(), [2, 1])
        self.assertEqual(len(target[0].unique(dim=0)), 3)

    @torch.no_grad()
    def test_budget_source_alignment_and_condition_affects_velocity(self):
        torch.manual_seed(21)
        model = AdaptivePUFM(preset("gdb_flow")).eval()
        points = torch.randn(1, 256, 3)
        state = model.prepare(points)
        diag = model.diagnostics(state, state["initial"])
        self.assertEqual(state["budget"].sum().item(), 768)
        self.assertTrue((state["budget"] >= diag["coverage_floor"]).all())
        self.assertEqual(diag["target_total_counts"].sum().item(), 1024)
        self.assertTrue((diag["source_kind"][:, :256] == 0).all())
        for j, count in enumerate(state["budget"][0].tolist()):
            mask = diag["source_region"][0, 256:] == j
            self.assertEqual(mask.sum().item(), count)
            self.assertEqual(diag["source_kind"][0, 256:][mask].tolist(), [1 + i % 2 for i in range(count)])
        t = torch.tensor([.4])
        with_condition = model.velocity(state["initial"], t, state)
        neutral = dict(state, film=None)
        self.assertGreater((with_condition - model.velocity(state["initial"], t, neutral)).abs().max().item(), 1e-7)
