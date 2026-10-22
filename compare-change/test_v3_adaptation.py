"""Protocol tests for changes that can silently invalidate a point-cloud comparison."""
import unittest
import numpy as np
import torch
from v3_common import assignment_target, fps_indices, midpoint, transport_loss, validate_prediction


class ProtocolTests(unittest.TestCase):
    def test_fps_never_repeats_indices_in_degenerate_cloud(self):
        ids = fps_indices(torch.zeros(2, 20, 3), 20)
        for row in ids:
            self.assertEqual(len(torch.unique(row)), 20)

    def test_midpoint_budget_and_input_origin(self):
        torch.manual_seed(1)
        points = torch.randn(2, 256, 3)
        initial = midpoint(points)
        self.assertEqual(initial.shape, (2, 1024, 3))
        self.assertTrue(torch.isfinite(initial).all())
        self.assertTrue((initial >= points.amin(1, keepdim=True) - 1e-6).all())
        self.assertTrue((initial <= points.amax(1, keepdim=True) + 1e-6).all())

    def test_transport_is_bijective_and_has_real_coordinate_gradients(self):
        points = torch.tensor([[[0., 0, 0], [2., 0, 0], [4., 0, 0]]], requires_grad=True)
        target = torch.tensor([[[4.1, 0, 0], [.1, 0, 0], [2.1, 0, 0]]])
        aligned = assignment_target(points, target)
        self.assertTrue(torch.allclose(aligned[0, :, 0], torch.tensor([.1, 2.1, 4.1])))
        loss = transport_loss(points, target)
        loss.backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(points.grad.abs().sum().item(), 0)

    def test_invalid_output_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_prediction(torch.zeros(1, 4096, 3), 1, 1024)
        with self.assertRaises(ValueError):
            validate_prediction(torch.full((1, 1024, 3), float("nan")), 1, 1024)


if __name__ == "__main__":
    unittest.main()
