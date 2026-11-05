"""Meaningful module isolation, solver, and candidate safety checks (unittest)."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
from types import ModuleType
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / 'module_screening'
sys.path.insert(0, str(ROOT))


def local_module(folder, name):
    directory = SCREEN / folder / 'PUFM-main/models'
    package = '_screening_test_' + folder
    if package not in sys.modules:
        mod = ModuleType(package)
        mod.__path__ = [str(directory)]
        sys.modules[package] = mod
    full_name = package + '.' + name
    spec = importlib.util.spec_from_file_location(full_name, directory / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


class IsolationTests(unittest.TestCase):
    def test_seven_physical_copies_and_unchanged_sources(self):
        manifest = json.loads((SCREEN / 'source_manifest.json').read_text(encoding='utf-8'))
        copies = sorted(p for p in SCREEN.iterdir() if p.is_dir() and (p / 'PUFM-main').is_dir())
        self.assertEqual(len(copies), 7)
        paths = [p / 'PUFM-main/models/diffusion.py' for p in copies]
        self.assertEqual(len({p.stat().st_ino for p in paths}), 7)
        for p in paths:
            self.assertFalse(p.is_symlink())
        for name, digest in manifest['baseline_files_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / 'baseline/PUFM-main' / name).read_bytes()).hexdigest(), digest)
        for folder, reference in manifest['references'].items():
            for name, digest in reference['files_sha256'].items():
                self.assertEqual(hashlib.sha256((ROOT / 'reference' / folder / name).read_bytes()).hexdigest(), digest)

    def test_disabled_backbone_ast_is_original(self):
        original = ast.parse((ROOT / 'baseline/PUFM-main/models/diffusion.py').read_text(encoding='utf-8'))
        for folder in SCREEN.iterdir():
            path = folder / 'PUFM-main/models/diffusion.py'
            if not path.is_file():
                continue
            copied = ast.parse(path.read_text(encoding='utf-8'))
            for cls in copied.body:
                if isinstance(cls, ast.ClassDef) and cls.name in {'PUFM', 'PUFM_w_attn'}:
                    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
                    self.assertEqual(forward.args.args[-1].arg, 'feature_residual')
                    forward.args.args.pop()
                    forward.args.defaults.pop()
                    forward.body = [n for n in forward.body if not (isinstance(n, ast.If) and 'feature_residual' in ast.unparse(n.test))]
            self.assertEqual(ast.dump(copied), ast.dump(original))

    def test_all_copies_pin_same_baseline_and_defaults(self):
        configs = [json.loads(p.read_text()) for p in SCREEN.glob('*/config.json')]
        self.assertEqual(len({c['baseline_sha256'] for c in configs}), 1)
        self.assertEqual(len({c['experiment'] for c in configs}), 7)
        for config in configs:
            self.assertEqual(config['epochs'], 20)
            self.assertEqual(config['flow_steps'], 5)
            self.assertEqual(config['train_patches_per_object'], 32)
            self.assertEqual(config['batch_size'], 8)


class ATSTests(unittest.TestCase):
    def setUp(self):
        self.module = local_module('m7_ats', 'candidate')

    def test_uniform_and_zero_errors_are_uniform(self):
        for errors in [np.ones(51), np.zeros(51)]:
            times, cdf = self.module.build_schedule(np.linspace(0, 1, 51), errors)
            np.testing.assert_allclose(times, np.linspace(0, 1, 6), atol=1e-12)
            self.assertEqual(cdf[-1], 1)

    def test_nonuniform_grid_has_five_positive_steps_total_one(self):
        grid = np.linspace(0, 1, 51)
        times, _ = self.module.build_schedule(grid, 1 + 100 * grid ** 3)
        self.assertEqual(len(times) - 1, 5)
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertAlmostEqual(sum(np.diff(times)), 1.)
        self.assertFalse(np.allclose(np.diff(times), .2))

    def test_invalid_error_curve_rejected(self):
        with self.assertRaises(ValueError):
            self.module.build_schedule([0, 1], [1, float('nan')])
        with self.assertRaises(ValueError):
            self.module.build_schedule([0, 0], [1, 1])

    def test_actual_solver_uses_dt_and_five_evaluations(self):
        # Load the actual integrate method without importing the large backbone.
        source = ast.parse((SCREEN / 'm7_ats/PUFM-main/models/screening_model.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ScreeningPUFM')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'integrate')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual integrate>', 'exec'), namespace)
        calls = []
        class ConstantVelocity:
            times = [0., .01, .07, .25, .6, 1.]
            def velocity(self, query, t, state):
                calls.append(float(t[0]))
                return torch.ones_like(query) * 2
        initial = torch.zeros(1, 8, 3)
        result = namespace['integrate'](ConstantVelocity(), {'initial': initial})
        torch.testing.assert_close(result, torch.full_like(initial, 2.))
        np.testing.assert_allclose(calls, ConstantVelocity.times[:-1], atol=1e-7)


class CandidateTests(unittest.TestCase):
    def test_aggregation_keeps_condition_specific_metrics(self):
        spec = importlib.util.spec_from_file_location('_screening_runner_test', SCREEN / 'm1_endpoint_cd/PUFM-main/screening.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        rows = [{'condition': 'A0', 'object_id': 'a', 'cd_l1': 1.},
                {'condition': 'C2', 'object_id': 'a', 'cd_l1': 3., 'hole_coverage': .4}]
        summary, objects = runner.aggregate(rows)
        self.assertEqual(summary['macro']['cd_l1'], 2.)
        self.assertEqual(summary['macro']['hole_coverage'], .4)
        self.assertEqual(summary['metric_condition_counts']['hole_coverage'], 1)

    def test_apml_is_exact_upstream_and_gradient_is_finite(self):
        copied = SCREEN / 'm2_apml/PUFM-main/models/apml_loss.py'
        self.assertEqual(copied.read_bytes(), (ROOT / 'reference/apml-main/src/loss/apml_loss.py').read_bytes())
        loss_cls = local_module('m2_apml', 'apml_loss').AdaptiveProbabilisticMatchingLoss
        source = torch.randn(1, 16, 3, requires_grad=True)
        loss = loss_cls()(source, source.detach().flip(1) + .01)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(float(source.grad.abs().sum()), 0)

    def test_m5_stop_really_preserves_input_and_never_restarts(self):
        module = local_module('m5_adaptive_denoise', 'candidate')
        cfg = json.loads((SCREEN / 'm5_adaptive_denoise/config.json').read_text())
        model = module.AdaptiveDenoiser(cfg).eval()
        with torch.no_grad():
            model.continue_head[-1].weight.zero_()
            model.continue_head[-1].bias.fill_(-100)
            points = torch.randn(2, 32, 3)
            output, diag = model.denoise(points)
        self.assertTrue(torch.equal(points, output))
        self.assertTrue(torch.equal(diag['denoise_steps'], torch.zeros(2, dtype=torch.long)))

    def test_m3_covariance_handles_duplicate_points(self):
        module = local_module('m3_gaussian_features', 'candidate')
        cfg = json.loads((SCREEN / 'm3_gaussian_features/config.json').read_text())
        features = module.FeatureBranch(cfg)(torch.zeros(2, 32, 3))
        self.assertEqual(features.shape, (2, 32, 128))
        self.assertTrue(torch.isfinite(features).all())
        features.square().sum().backward()


if __name__ == '__main__':
    unittest.main()
