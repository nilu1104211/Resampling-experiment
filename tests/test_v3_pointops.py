"""FPS equivalence must include interpolation ties and the source SPU RNG rule."""
import os
import unittest
from unittest.mock import patch

import torch
from adaptive_resampling.v3_pointops import fps_indices, fps_indices_reference


class PointOpsTests(unittest.TestCase):
    def test_reference_handles_duplicate_xyz_without_repeated_indices(self):
        with patch.dict(os.environ, V3_FPS_BACKEND="torch"):
            ids = fps_indices(torch.zeros(2, 20, 3), 20)
        self.assertTrue(torch.equal(ids, torch.arange(20).expand(2, -1)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_matches_reference_for_midpoint_ties_and_strided_input(self):
        torch.manual_seed(21)
        points = torch.randn(4, 128, 3, device="cuda")
        pool = ((points[:, :, None] + points[:, None, :8]) * .5).flatten(1, 2)
        grid = torch.cartesian_prod(*[torch.arange(4, device="cuda")] * 3).float()[None]
        channel_first = pool.transpose(1, 2).contiguous().transpose(1, 2)
        for value, count in [(pool, 512), (pool[:, ::2], 256), (channel_first, 512), (grid, 64), (torch.zeros(2, 64, 3, device="cuda"), 64)]:
            with self.subTest(shape=value.shape), patch.dict(os.environ, V3_FPS_BACKEND="cuda"):
                expected = fps_indices_reference(value, count)
                actual = fps_indices(value, count)
                self.assertTrue(torch.equal(expected, actual))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_source_spu_fps_random_start_and_duplicate_ties(self):
        torch.manual_seed(9)
        for x in [torch.randn(3, 256, 3, device="cuda"), torch.zeros(2, 32, 3, device="cuda")]:
            starts = torch.randint(0, len(x[0]), (len(x),)).to(x.device)
            options = dict(start_indices=starts, exclude_selected=False, initial_distance=1e10)
            expected = fps_indices_reference(x, x.shape[1], **options)
            with patch.dict(os.environ, V3_FPS_BACKEND="cuda"):
                self.assertTrue(torch.equal(expected, fps_indices(x, x.shape[1], **options)))

    def test_invalid_backend_and_shape_fail(self):
        with patch.dict(os.environ, V3_FPS_BACKEND="bad"), self.assertRaises(ValueError):
            fps_indices(torch.zeros(1, 10, 3), 3)
        with self.assertRaises(ValueError):
            fps_indices(torch.zeros(1, 10, 3), 11)


if __name__ == "__main__":
    unittest.main()
