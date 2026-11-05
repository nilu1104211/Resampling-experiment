"""Sequential queue ordering, failure isolation, resume, and logging."""
from pathlib import Path
import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'module_screening'))
spec = importlib.util.spec_from_file_location('_screening_queue_test', ROOT / 'module_screening/train_all.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def options(output, **overrides):
    values = dict(output=Path(output), root=Path('dataset'), baseline_checkpoint=Path('baseline.pt'),
        epochs=20, batch_size=8, workers=0, assignment_workers=8, device='cuda',
        fused_optimizer=True, with_capacity_controls=False, tasks=None, resume=False,
        limit_batches=0, limit_samples=0)
    values.update(overrides)
    return argparse.Namespace(**values)


class PlanTests(unittest.TestCase):
    def test_defaults_keep_every_run_independent_and_never_test(self):
        plan = queue.build_plan(options(Path('output')))
        self.assertEqual([t['name'] for t in plan], [t[0] for t in queue.DEFAULT_TASKS])
        self.assertEqual(len(plan), 11)
        self.assertEqual(len({t['output'] for t in plan}), len(plan))
        self.assertTrue(all('predict' not in t['argv'] for t in plan))
        for task in plan:
            self.assertEqual(task['config']['batch_size'], 8)
            self.assertEqual(task['config']['epochs'], 20)
            self.assertEqual(task['expected_debug'], False)
        self.assertEqual(plan[-3]['command'], 'calibrate')
        self.assertEqual(plan[-1]['command'], 'validate')
        self.assertIn('--schedule', plan[-1]['argv'])

    def test_ats_validation_adds_its_calibration_dependency(self):
        plan = queue.build_plan(options('output', tasks=['m7_ats_validation']))
        self.assertEqual([t['name'] for t in plan], ['m7_calibrate', 'm7_ats_validation'])

    def test_capacity_controls_and_debug_flags_are_explicit(self):
        plan = queue.build_plan(options('output', with_capacity_controls=True,
            limit_batches=1, limit_samples=1))
        self.assertEqual(len(plan), 13)
        self.assertTrue(all(t['expected_debug'] for t in plan))
        self.assertIn('--allow-debug', plan[-1]['argv'])

    def test_dry_run_does_not_create_outputs_or_load_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not_created'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'module_screening/train_all.py'),
                '--dry-run', '--output', str(output), '--root', str(Path(directory) / 'missing')],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('m7_ats_validation:', result.stdout)
            self.assertFalse(output.exists())


class ExecutionTests(unittest.TestCase):
    def test_fail_fast_resume_and_verify_before_skip(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = options(Path(directory) / 'queue', tasks=['b_extra', 'm1a', 'm1b'])
            plan = queue.build_plan(args)
            calls = []
            fail_once = [True]

            def job(command, log):
                output = Path(command[command.index('--output') + 1])
                calls.append((output.name, '--resume' in command))
                output.mkdir(parents=True, exist_ok=True)
                log.write_text('fake child log', encoding='utf-8')
                if output.name == 'm1a' and fail_once[0]:
                    fail_once[0] = False
                    (output / 'last.pt').write_text('mock resume checkpoint')
                    raise RuntimeError('simulated child failure')
                (output / 'completed.json').write_text('{}')

            def verify(task, identity):
                if not (Path(task['output']) / 'completed.json').is_file():
                    raise ValueError('completion evidence missing')

            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                queue.execute(args, plan, {'test': 'identity'}, run_job=job, verify=verify)
            self.assertEqual(calls, [('b_extra', False), ('m1a', False)])
            state = queue.read_json(args.output / 'queue.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['tasks']['m1b']['status'], 'pending')
            self.assertFalse((args.output / 'queue.lock').exists())
            args.resume = True
            queue.execute(args, plan, {'test': 'identity'}, run_job=job, verify=verify)
            self.assertEqual(calls[-2:], [('m1a', True), ('m1b', False)])
            self.assertEqual(queue.read_json(args.output / 'queue.json')['status'], 'completed')
            self.assertEqual((args.output / 'logs/m1a.attempt_001.log').read_text(), 'fake child log')
            self.assertTrue((args.output / 'logs/m1a.attempt_002.log').exists())
            with self.assertRaisesRegex(ValueError, 'original queue options'):
                queue.execute(args, plan, {'test': 'changed source'}, run_job=job, verify=verify)
            (args.output / 'runs/b_extra/completed.json').unlink()
            before = list(calls)
            with self.assertRaisesRegex(ValueError, 'completion evidence missing'):
                queue.execute(args, plan, {'test': 'identity'}, run_job=job, verify=verify)
            self.assertEqual(calls, before)

    def test_second_supervisor_cannot_take_same_queue_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with queue.queue_lock(output, False):
                with self.assertRaisesRegex(RuntimeError, 'already locked'):
                    with queue.queue_lock(output, True):
                        self.fail('A second supervisor acquired the lock')
                self.assertTrue((output / 'queue.lock').is_file())
            self.assertFalse((output / 'queue.lock').exists())

    def test_actual_child_log_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            success = Path(directory) / 'success.log'
            queue.run_child([sys.executable, '-u', '-c', "print('child output')"], success)
            self.assertIn('child output', success.read_text(encoding='utf-8'))
            failure = Path(directory) / 'failure.log'
            with self.assertRaisesRegex(RuntimeError, 'code 9'):
                queue.run_child([sys.executable, '-u', '-c', "print('before error'); raise SystemExit(9)"], failure)
            self.assertIn('before error', failure.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
