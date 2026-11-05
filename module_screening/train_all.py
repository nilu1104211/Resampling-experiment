"""Sequential independent PUFM screening jobs, with logs and explicit resume.

No frozen test evaluation is dispatched. M7 calibrates and validates a schedule.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

from run import FOLDERS

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = [
    ('b_extra', 'm1', 'b_extra', 'train'),
    ('m1a', 'm1', 'm1a', 'train'),
    ('m1b', 'm1', 'm1b', 'train'),
    ('m2a', 'm2', 'm2a', 'train'),
    ('m3a', 'm3', 'm3a', 'train'),
    ('m4a', 'm4', 'm4a', 'train'),
    ('m5a', 'm5', 'm5a', 'train'),
    ('m6a', 'm6', 'm6a', 'train'),
    ('m7_calibrate', 'm7', 'm7b', 'calibrate'),
    ('m7_uniform_validation', 'm7', 'm7a', 'validate'),
    ('m7_ats_validation', 'm7', 'm7b', 'validate'),
]
CONTROLS = [('m3_mlp_control', 'm3', 'mlp_control', 'train'),
            ('m4_mlp_control', 'm4', 'mlp_control', 'train')]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def build_plan(args):
    tasks = list(DEFAULT_TASKS)
    if args.with_capacity_controls:
        tasks[6:6] = CONTROLS
    if args.tasks:
        selected = set(args.tasks)
        if 'm7_ats_validation' in selected:
            selected.add('m7_calibrate')
        tasks = [task for task in tasks if task[0] in selected]
        if selected != {task[0] for task in tasks}:
            raise ValueError('Capacity control tasks need --with-capacity-controls')
    plan = []
    for name, experiment, variant, command in tasks:
        config = read_json(REPO / 'module_screening' / FOLDERS[experiment] / 'config.json')
        config.update(variant=variant, epochs=args.epochs, batch_size=args.batch_size,
                      workers=args.workers, assignment_workers=args.assignment_workers)
        config_path = args.output / 'configs' / (name + '.json')
        output = args.output / 'runs' / name
        cmd = [sys.executable, '-u', '-B', str(REPO / 'module_screening/run.py'), experiment, command,
               '--config', str(config_path), '--root', str(args.root),
               '--baseline-checkpoint', str(args.baseline_checkpoint), '--device', args.device,
               '--output', str(output)]
        if args.fused_optimizer and command == 'train':
            cmd.append('--fused-optimizer')
        if args.limit_batches and command in {'train', 'calibrate'}:
            cmd += ['--limit-batches', str(args.limit_batches)]
        if args.limit_samples and command in {'train', 'validate'}:
            cmd += ['--limit-samples', str(args.limit_samples)]
        if name == 'm7_ats_validation':
            cmd += ['--schedule', str(args.output / 'runs/m7_calibrate/schedule.json')]
            if args.limit_batches:
                cmd.append('--allow-debug')
        plan.append({'name': name, 'experiment': experiment, 'command': command,
                     'config': config, 'config_path': str(config_path), 'output': str(output), 'argv': cmd,
                     'expected_debug': bool(args.limit_batches or args.limit_samples) if command == 'train'
                        else bool(args.limit_batches) if command == 'calibrate' else bool(args.limit_samples)})
    return plan


def preflight(args, plan):
    for name in ('base_manifest.json', 'generation_config.json', 'dataset_splits.json'):
        if not (args.root / 'metadata' / name).is_file():
            raise FileNotFoundError(f'V3 metadata missing: {args.root / "metadata" / name}')
    digest = file_hash(args.baseline_checkpoint)
    if any(task['config']['baseline_sha256'] != digest for task in plan):
        raise ValueError('The common original PUFM checkpoint SHA256 does not match')
    # One shared B-extra is fair only when all M1-M4 continuation budgets match.
    budgets = {(t['config']['epochs'], t['config']['batch_size'], t['config']['lr']) for t in plan
               if t['experiment'] in {'m1', 'm2', 'm3', 'm4'}}
    if len(budgets) > 1:
        raise ValueError('M1-M4 and shared B-extra must retain equal epochs/batch/LR')
    sources = list((REPO / 'module_screening').rglob('*.py'))
    sources += list((REPO / 'adaptive_resampling').glob('*.py')) + [REPO / 'run_subset_eval.py']
    return {'baseline_sha256': digest,
            'metadata_sha256': {name: file_hash(args.root / 'metadata' / name) for name in
                ('base_manifest.json', 'generation_config.json', 'dataset_splits.json')},
            'sources_sha256': {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(sources)}}


def verify_artifacts(task, identity):
    output = Path(task['output'])
    if task['command'] == 'train':
        import torch
        last = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
        for saved in (last, best):
            if (saved.get('schema') != 'pufm-module-checkpoint-v1' or saved.get('config') != task['config']
                    or not saved.get('training_complete')
                    or saved.get('partial_debug_run', False) != task['expected_debug']
                    or saved.get('baseline_sha256') != identity['baseline_sha256']
                    or saved.get('selection_completed_epochs', 0) != task['config']['epochs']):
                raise ValueError(f'{task["name"]}: checkpoint completion/configuration check failed')
            if saved['identity']['metadata_sha256'] != identity['metadata_sha256']:
                raise ValueError('Checkpoint V3 metadata differs from queue')
        if last['completed_epochs'] != task['config']['epochs']:
            raise ValueError('The final training epoch was not completed')
        return
    saved = read_json(output / ('schedule.json' if task['command'] == 'calibrate' else 'summary.json'))
    if (saved.get('config') != task['config'] or saved.get('partial_debug_run', False) != task['expected_debug']
            or saved.get('baseline_sha256') != identity['baseline_sha256']
            or saved.get('metadata_sha256') != identity['metadata_sha256']):
        raise ValueError(f'{task["name"]}: output identity/debug check failed')
    if task['command'] == 'calibrate':
        times = saved['times']
        if (saved.get('split') != 'train' or len(times) != 6 or times[0] != 0 or times[-1] != 1
                or any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError('Invalid M7 training-only five-interval calibration')
    elif saved.get('split') != 'validation' or len(saved['conditions']) != 7:
        raise ValueError('M7 validation must contain all seven validation conditions')


@contextmanager
def queue_lock(output, resume):
    lock = output / 'queue.lock'
    if lock.exists() and resume and os.name != 'nt':
        previous = read_json(lock)
        if previous.get('host') == socket.gethostname():
            try:
                os.kill(previous['pid'], 0)  # POSIX liveness query; never used on Windows.
            except ProcessLookupError:
                lock.unlink()
            except PermissionError:
                pass
    token = uuid.uuid4().hex
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(f'Queue is already locked; check its PID before starting again: {lock}') from None
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump({'host': socket.gethostname(), 'pid': os.getpid(), 'token': token}, handle)
    try:
        yield
    finally:
        if lock.exists() and read_json(lock).get('token') == token:
            lock.unlink()


def run_child(command, log_path):
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
    with log_path.open('w', encoding='utf-8') as log:
        log.write(shlex.join(command) + '\n')
        log.flush()
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            returncode = process.wait()
            if returncode:
                raise RuntimeError(f'Child exited with code {returncode}; log: {log_path}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def handle_termination(signum, frame):
    raise KeyboardInterrupt()


def execute(args, plan, identity, *, run_job=run_child, verify=verify_artifacts):
    output = args.output
    state_path = output / 'queue.json'
    if args.resume:
        state = read_json(state_path)
        if state['plan'] != plan or state['identity'] != identity:
            raise ValueError('Resume requires the original queue options, data, checkpoint, and source files')
    else:
        if output.exists():
            raise FileExistsError(f'Queue output exists; use --resume for this queue or choose a new --output: {output}')
        output.mkdir(parents=True)
        state = {'schema': 'pufm-sequential-queue-v1', 'plan': plan, 'identity': identity,
                 'status': 'pending', 'tasks': {t['name']: {'status': 'pending', 'attempts': 0} for t in plan}}
    with queue_lock(output, args.resume):
        (output / 'configs').mkdir(exist_ok=True)
        (output / 'logs').mkdir(exist_ok=True)
        for task in plan:
            config_path = Path(task['config_path'])
            if args.resume and config_path.exists() and read_json(config_path) != task['config']:
                raise ValueError('A frozen queue configuration was modified')
            write_json(config_path, task['config'])
        state['status'] = 'running'
        write_json(state_path, state)
        active = None
        try:
            for index, task in enumerate(plan, 1):
                active = state['tasks'][task['name']]
                if active['status'] == 'completed':
                    verify(task, identity)
                    print(f'[{index}/{len(plan)}] SKIP completed: {task["name"]}', flush=True)
                    continue
                command = list(task['argv'])
                task_output = Path(task['output'])
                if task_output.exists():
                    if not args.resume:
                        raise FileExistsError(f'Task output already exists: {task_output}')
                    if task['command'] == 'train':
                        checkpoint = task_output / 'last.pt'
                        if not checkpoint.is_file():
                            raise RuntimeError(f'Incomplete task has no resumable last.pt; preserve this output and inspect it: {task_output}')
                        command += ['--resume', str(checkpoint)]
                    else:
                        verify(task, identity)
                        active['status'] = 'completed'
                        write_json(state_path, state)
                        print(f'[{index}/{len(plan)}] Recovered completed output: {task["name"]}', flush=True)
                        continue
                active['attempts'] += 1
                active.update(status='running', started_unix=time.time())
                log_path = output / 'logs' / f'{task["name"]}.attempt_{active["attempts"]:03d}.log'
                active['log'] = str(log_path)
                write_json(state_path, state)
                print(f'[{index}/{len(plan)}] START {task["name"]}: {log_path}', flush=True)
                run_job(command, log_path)
                verify(task, identity)
                active.update(status='completed', completed_unix=time.time())
                write_json(state_path, state)
                print(f'[{index}/{len(plan)}] COMPLETED {task["name"]}', flush=True)
            state['status'] = 'completed'
            write_json(state_path, state)
            print(f'ALL COMPLETED: {output}', flush=True)
        except BaseException as error:
            state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            if active is not None and active['status'] != 'completed':
                active.update(status=state['status'], error=str(error))
            write_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('V3_DATA_ROOT', REPO / 'adaptive_resampling_data_v3')))
    parser.add_argument('--baseline-checkpoint', type=Path, default=Path(os.environ.get('PUFM_BASELINE_CHECKPOINT', REPO / 'server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt')))
    parser.add_argument('--output', type=Path, default=REPO / 'server_runs/patch32/module_screening/sequential_v1')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--assignment-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--with-capacity-controls', action='store_true')
    parser.add_argument('--tasks', nargs='+', choices=[t[0] for t in DEFAULT_TASKS + CONTROLS])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print commands only, without creating files or loading data')
    parser.add_argument('--limit-batches', type=int, default=0, help='Debug only; marks training and ATS outputs as partial')
    parser.add_argument('--limit-samples', type=int, default=0, help='Debug only; limits validation samples per condition')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.assignment_workers) < 1 or min(args.workers, args.limit_batches, args.limit_samples) < 0:
        parser.error('Invalid epoch/batch/worker/debug budget')
    args.root = args.root.resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.resolve()
    args.output = args.output.resolve()
    plan = build_plan(args)
    if args.dry_run:
        for task in plan:
            print(task['name'] + ': ' + shlex.join(task['argv']))
        return
    identity = preflight(args, plan)
    old_handler = signal.signal(signal.SIGTERM, handle_termination)
    try:
        execute(args, plan, identity)
    finally:
        signal.signal(signal.SIGTERM, old_handler)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Queue interrupted; the active child was stopped. Use the same options plus --resume.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f'QUEUE FAILED: {error}', file=sys.stderr)
        raise SystemExit(1)
