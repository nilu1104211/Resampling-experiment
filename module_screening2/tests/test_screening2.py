"""Fairness contracts, meaningful adapter behavior, queue failure and resume."""
from pathlib import Path
import argparse
import ast
import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
from module_screening2 import protocol, train_all as queue
from module_screening2.report import collect


def options(output, **overrides):
    values = dict(output=Path(output), root=Path('dataset'), baseline_checkpoint=Path('baseline.pt'),
                  epochs=15, batch_size=8, workers=0, assignment_workers=8, device='cuda',
                  fps_backend=None, cpu_threads=8, fused_optimizer=True, tasks=None,
                  resume=False, skip_checks=False, limit_batches=0, limit_samples=0)
    values.update(overrides)
    return argparse.Namespace(**values)


def temporary_directory():
    parent = REPO / 'outputs/module_screening2_checks/unit_tmp'
    parent.mkdir(parents=True, exist_ok=True)
    directory = tempfile.TemporaryDirectory(dir=parent)
    if not Path(directory.name).resolve().is_relative_to(parent.resolve()):
        raise AssertionError('Test temporary directory escaped its output root')
    return directory


@contextlib.contextmanager
def candidate_module(folder):
    # Each copied package has its own module objects even inside this test process.
    package_name = '_module2_test_' + folder
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / folder / 'PUFM-main/models')]
    sys.modules[package_name] = package
    saved = {k: v for k, v in sys.modules.items() if k == 'models' or k.startswith('models.')}
    for k in saved:
        del sys.modules[k]
    sys.modules['models'] = package
    os.environ['V3_FPS_BACKEND'] = 'torch'
    try:
        yield importlib.import_module(package_name + '.candidate')
    finally:
        for k in list(sys.modules):
            if k == 'models' or k.startswith('models.') or k == package_name or k.startswith(package_name + '.'):
                del sys.modules[k]
        sys.modules.update(saved)


class IsolationTests(unittest.TestCase):
    def test_physical_copies_no_generated_files_or_cross_experiment_imports(self):
        paths = [ROOT / name / 'PUFM-main/models/diffusion.py' for name in protocol.FOLDERS]
        self.assertEqual(len(paths), 11)
        self.assertEqual(len({p.stat().st_ino for p in paths}), 11)
        for folder in protocol.FOLDERS:
            project = ROOT / folder / 'PUFM-main'
            self.assertFalse(any(p.is_symlink() for p in project.rglob('*')))
            for source in project.rglob('*.py'):
                text = source.read_text(encoding='utf-8')
                self.assertNotIn('from module_screening.', text)
                self.assertNotIn('from reference2.', text)
                ast.parse(text)
            self.assertFalse(any(project.rglob('*.pt')))

    def test_disabled_backbone_ast_matches_original(self):
        original = ast.parse((REPO / 'baseline/PUFM-main/models/diffusion.py').read_text(encoding='utf-8'))
        for name in protocol.FOLDERS:
            copied = ast.parse((ROOT / name / 'PUFM-main/models/diffusion.py').read_text(encoding='utf-8'))
            for cls in copied.body:
                if isinstance(cls, ast.ClassDef) and cls.name in {'PUFM', 'PUFM_w_attn'}:
                    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
                    self.assertEqual(fn.args.args[-1].arg, 'feature_residual')
                    fn.args.args.pop()
                    fn.args.defaults.pop()
                    fn.body = [n for n in fn.body if not (isinstance(n, ast.If) and 'feature_residual' in ast.unparse(n.test))]
            self.assertEqual(ast.dump(copied), ast.dump(original))

    def test_all_configs_pin_same_15_epoch_protocol_and_reject_budget_changes(self):
        for name in protocol.FOLDERS:
            cfg = json.loads((ROOT / name / 'config.json').read_text())
            protocol.validate_config(cfg)
            self.assertEqual(cfg['epochs'], 15)
            self.assertEqual(cfg['batch_size'], 8)
            for key, value in [('input_points', 512), ('flow_steps', 10), ('seed', 22), ('lr', .001)]:
                with self.assertRaises(ValueError):
                    protocol.validate_config({**cfg, key: value})


class SamplingTests(unittest.TestCase):
    def test_batch_one_fixed_budget_repeatability_and_rng_isolation(self):
        torch.set_num_threads(2)
        points = torch.randn(1, 64, 3) * .2
        variants = {'pointsp_wrs': ['pointsp_wrs', 'uniform_random'], 'pointsp_ffps': ['pointsp_ffps'],
                    'dy_density': ['dy_density_low', 'dy_density_high'], 'dy_curvature': ['dy_curvature'],
                    'pointsp_lgp': ['pointsp_lgp']}
        for folder, names in variants.items():
            with candidate_module(folder) as module:
                ops = importlib.import_module(module.__package__ + '.adapter_ops')
                initial = ops.select_fps(ops.midpoint_pool(points), 256)
                for name in names:
                    cfg = {**protocol.default_config(folder), 'variant': name}
                    sampler = module.build_candidate(cfg)
                    state = torch.get_rng_state().clone()
                    first, _ = sampler.initialize(points, initial)
                    second, _ = sampler.initialize(points, initial)
                    self.assertEqual(first.shape, (1, 256, 3))
                    torch.testing.assert_close(first, second, atol=0, rtol=0)
                    self.assertTrue(torch.isfinite(first).all())
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_lgp_keeps_planar_proposals_in_tangent_plane(self):
        points = torch.randn(1, 64, 3)
        points[..., 2] = 0
        with candidate_module('pointsp_lgp') as module:
            output, _ = module.build_candidate(protocol.default_config('pointsp_lgp')).initialize(points, points.repeat(1, 4, 1))
            torch.testing.assert_close(output[..., 2], torch.zeros(1, 256), atol=1e-6, rtol=0)


class LearnedTests(unittest.TestCase):
    def test_rectifier_starts_as_identity_and_has_supervised_gradient(self):
        with candidate_module('upp_rectify') as module:
            candidate = module.build_candidate(protocol.default_config('upp_rectify'))
            points = torch.randn(2, 64, 3)
            prediction, _ = candidate.refine(points)
            torch.testing.assert_close(prediction, points, atol=0, rtol=0)
            loss = candidate.auxiliary({'supervision': {'clean_input_xyz': points + .01}}, {'observed': prediction})
            loss.backward()
            self.assertGreater(float(candidate.head[-1].bias.grad.abs().sum()), 0)

    def test_nmm_is_permutation_equivariant_and_clean_teacher_is_detached(self):
        with candidate_module('dwc_nmm') as module:
            candidate = module.build_candidate(protocol.default_config('dwc_nmm')).eval()
            points = torch.randn(1, 64, 3)
            perm = torch.randperm(64)
            features, cache = candidate.encode(points)
            permuted, _ = candidate.encode(points[:, perm])
            torch.testing.assert_close(features[:, perm], permuted, atol=2e-6, rtol=2e-5)
            clean = (points + .01).requires_grad_()
            loss = candidate.auxiliary({'input_xyz': points, 'supervision': {'clean_input_xyz': clean}},
                                       {'feature_cache': cache, 'features': features})
            loss.backward()
            self.assertIsNone(clean.grad)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in candidate.noisy_convs.parameters()))

    def test_anchor_slots_retained_and_ranker_receives_gradient(self):
        points = torch.randn(1, 64, 3)
        target = torch.randn(1, 256, 3)
        for folder in ('upp_completion', 'spo_keypoint'):
            with candidate_module(folder) as module:
                candidate = module.build_candidate(protocol.default_config(folder))
                output, cache = candidate.initialize(points, points.repeat(1, 4, 1))
                self.assertEqual(output.shape, (1, 256, 3))
                self.assertEqual(cache['anchors_retained'], 32)
                torch.testing.assert_close(output[:, -32:], cache['anchors'])
                loss = candidate.auxiliary({'input_xyz': points, 'gt_xyz': target}, {'anchor_cache': cache})
                loss.backward()
                if folder == 'spo_keypoint':
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in candidate.query_ranking.parameters()))

    def test_projection_batch_normalization_and_gradient(self):
        with candidate_module('spo_projection') as module:
            points = torch.randn(1, 32, 3, requires_grad=True)
            single = module.projection_loss(points)
            doubled = module.projection_loss(points.repeat(2, 1, 1))
            torch.testing.assert_close(single, doubled)
            single.backward()
            self.assertTrue(torch.isfinite(points.grad).all())
            self.assertGreater(float(points.grad.abs().sum()), 0)


class QueueTests(unittest.TestCase):
    def test_default_queue_and_automatic_fair_controls(self):
        plan = queue.build_plan(options('output'))
        self.assertEqual(len(plan), 23)
        self.assertEqual(sum(t['command'] == 'train' for t in plan), 14)
        self.assertEqual(len({t['output'] for t in plan}), len(plan))
        self.assertTrue(all('predict' not in t['argv'] for t in plan))
        trained = [t for t in plan if t['command'] == 'train']
        self.assertTrue(all(t['config']['epochs'] == 15 and t['config']['batch_size'] == 8 for t in trained))
        subset = queue.build_plan(options('output', tasks=['upp_rectify'], skip_checks=True))
        self.assertEqual([t['name'] for t in subset], ['b0', 'b_extra', 'upp_rectify'])
        wrs = queue.build_plan(options('output', tasks=['pointsp_wrs'], skip_checks=True))
        self.assertEqual([t['name'] for t in wrs], ['b0', 'b_extra', 'uniform_random', 'pointsp_wrs'])

    def test_dry_run_needs_no_data_and_creates_nothing(self):
        with temporary_directory() as directory:
            output = Path(directory) / 'never_created'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'train_all.py'), '--dry-run', '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())
            self.assertIn('spo_projection:', result.stdout)

    def test_fail_fast_resume_and_identity_rejection(self):
        with temporary_directory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.object(queue, 'collect'):
            args = options(Path(directory) / 'queue', tasks=['b_extra', 'upp_rectify'], skip_checks=True)
            plan = queue.build_plan(args)
            calls, fail = [], [True]

            def run_job(command, log):
                output = Path(command[command.index('--output') + 1])
                calls.append((output.name, '--resume' in command))
                output.mkdir(parents=True, exist_ok=True)
                log.write_text('test log')
                if output.name == 'b_extra' and fail[0]:
                    fail[0] = False
                    (output / 'last.pt').write_text('test checkpoint')
                    raise RuntimeError('simulated failure')
                (output / 'complete').touch()

            def verify(task, identity):
                if not (Path(task['output']) / 'complete').exists():
                    raise ValueError('completion missing')

            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                queue.execute(args, plan, {'test': 1}, run_job=run_job, verify=verify)
            self.assertEqual([c[0] for c in calls], ['b0', 'b_extra'])
            self.assertFalse((args.output / 'queue.lock').exists())
            args.resume = True
            queue.execute(args, plan, {'test': 1}, run_job=run_job, verify=verify)
            self.assertEqual(calls[-2:], [('b_extra', True), ('upp_rectify', False)])
            self.assertEqual(queue.read_json(args.output / 'queue.json')['status'], 'completed')
            with self.assertRaisesRegex(ValueError, 'original queue options'):
                queue.execute(args, plan, {'test': 2}, run_job=run_job, verify=verify)

    def test_duplicate_queue_lock_is_rejected(self):
        with temporary_directory() as directory:
            with queue.queue_lock(Path(directory), False):
                with self.assertRaisesRegex(RuntimeError, 'already locked'):
                    with queue.queue_lock(Path(directory), False):
                        self.fail('A second queue obtained the lock')

    def test_active_source_mutation_is_rejected(self):
        relative = 'module_screening2/protocol.py'
        queue.verify_sources({'sources_sha256': {relative: queue.file_hash(REPO / relative)}})
        with self.assertRaisesRegex(ValueError, 'Source modified'):
            queue.verify_sources({'sources_sha256': {relative: 'wrong_digest'}})


class ReportTests(unittest.TestCase):
    def test_paired_input_guard_and_debug_marking(self):
        with temporary_directory() as directory:
            root = Path(directory)
            summary = {'input_selection_sha256': 'same_input', 'metadata_sha256': {}, 'runtime': {},
                       'baseline_sha256': 'same_weight', 'split': 'validation', 'partial_debug_run': False,
                       'parameter_count': 10, 'extra_parameter_count': 0,
                       'macro': {'cd_l1': .1, 'f_score': .2, 'hausdorff_l2': .3, 'prediction_seconds': .01}}
            plan = []
            for name in ('b0', 'b_extra'):
                output = root / name
                output.mkdir()
                task = {'name': name, 'command': 'validate' if name == 'b0' else 'train',
                        'output': str(output), 'expected_debug': name == 'b_extra'}
                plan.append(task)
                target = output
                if name == 'b_extra':
                    (output / 'best.json').write_text(json.dumps({'epoch': 1}))
                    target = output / 'validation/epoch_0001'
                    target.mkdir(parents=True)
                (target / 'summary.json').write_text(json.dumps(summary))
            (root / 'queue.json').write_text(json.dumps({'plan': plan}))
            collect(root)
            rows = json.loads((root / 'comparison.json').read_text())
            self.assertTrue(rows[1]['partial_debug_run'])
            self.assertEqual(rows[1]['control'], 'b0')
            summary['input_selection_sha256'] = 'different_input'
            (root / 'b_extra/validation/epoch_0001/summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'unfair input'):
                collect(root)


if __name__ == '__main__':
    unittest.main()
