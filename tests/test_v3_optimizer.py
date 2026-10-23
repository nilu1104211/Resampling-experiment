"""Check fused update accuracy and resumes from CPU step counters."""
from pathlib import Path
import copy
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adaptive_resampling.v3_optimizer import make_optimizer, load_optimizer_state


class OptimizerTests(unittest.TestCase):
    def test_default_cpu_behavior_and_explicit_cuda_requirement(self):
        p = torch.nn.Parameter(torch.ones(3))
        optimizer = make_optimizer(torch.optim.AdamW, [p], lr=.01, weight_decay=.05)
        self.assertIsNone(optimizer.defaults["fused"])
        p.grad = torch.ones_like(p)
        optimizer.step()
        self.assertTrue(torch.isfinite(p).all())
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            make_optimizer(torch.optim.AdamW, [p], fused=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_fused_adam_and_adamw_match_standard_updates(self):
        for kind in (torch.optim.Adam, torch.optim.AdamW):
            torch.manual_seed(7)
            plain = [torch.nn.Parameter(torch.randn(11, 7, device="cuda")),
                     torch.nn.Parameter(torch.randn(5, device="cuda"))]
            fused = [torch.nn.Parameter(p.detach().clone()) for p in plain]
            ordinary = make_optimizer(kind, plain, lr=1e-4, weight_decay=.05)
            accelerated = make_optimizer(kind, fused, lr=1e-4, weight_decay=.05, fused=True)
            for _ in range(8):
                for p, q in zip(plain, fused):
                    p.grad = torch.randn_like(p)
                    q.grad = p.grad.clone()
                ordinary.step()
                accelerated.step()
            for p, q in zip(plain, fused):
                torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_resume_enables_fused_and_moves_old_step_counters(self):
        for kind in (torch.optim.Adam, torch.optim.AdamW):
            p = torch.nn.Parameter(torch.tensor([.3, -.5, 1.2], device="cuda"))
            original = make_optimizer(kind, [p], lr=1e-4, weight_decay=.05)
            for _ in range(3):
                p.grad = torch.tensor([.2, -.1, .3], device="cuda")
                original.step()
            saved = copy.deepcopy(original.state_dict())
            for state in saved["state"].values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cpu()
            q = torch.nn.Parameter(p.detach().clone())
            resumed = make_optimizer(kind, [q], lr=1e-4, weight_decay=.05, fused=True)
            load_optimizer_state(resumed, saved, fused=True)
            self.assertTrue(resumed.param_groups[0]["fused"])
            self.assertFalse(resumed.param_groups[0]["foreach"])
            self.assertEqual(resumed.state[q]["step"].device.type, "cuda")
            for _ in range(5):
                p.grad = torch.tensor([-.2, .6, -.1], device="cuda")
                q.grad = p.grad.clone()
                original.step()
                resumed.step()
            self.assertEqual(int(resumed.state[q]["step"]), 8)
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(original.state[p][key], resumed.state[q][key], atol=1e-8, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
