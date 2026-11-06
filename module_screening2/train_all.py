"""One-click sequential 15-epoch independent PUFM module screening."""
from pathlib import Path
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from module_screening2.protocol import DEFAULT_TASKS, FOLDERS, validate_config
from module_screening2.report import collect


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
    if args.tasks:
        selected = set(args.tasks) | {'b0'}
        for candidate, control in [('pointsp_wrs', 'uniform_random'),
                                   ('frozen_pointsp_wrs', 'frozen_uniform_random'),
                                   ('dwc_nmm', 'dwc_no_contrastive')]:
            if candidate in selected:
                selected.add(control)
        if any(t[0] in selected and t[3] == 'train' for t in tasks):
            selected.add('b_extra')
        tasks = [task for task in tasks if task[0] in selected]
    plan = []
    if not args.skip_checks:
        command = [sys.executable, '-u', '-B', str(REPO / 'module_screening2/check.py'),
                   '--real-patch', '--whole-shape', '--device', args.device,
                   '--cpu-threads', str(args.cpu_threads), '--root', str(args.root),
                   '--baseline-checkpoint', str(args.baseline_checkpoint),
                   '--output', str(args.output / 'checks')]
        if args.fps_backend:
            command += ['--fps-backend', args.fps_backend]
        plan.append({'name': 'interface_checks', 'experiment': 'all', 'command': 'check',
                     'config': {}, 'config_path': str(args.output / 'configs/interface_checks.json'),
                     'output': str(args.output / 'checks'), 'argv': command, 'expected_debug': False})
    for name, experiment, variant, action in tasks:
        config = read_json(REPO / 'module_screening2' / FOLDERS[experiment] / 'config.json')
        config.update(variant=variant, epochs=args.epochs, batch_size=args.batch_size,
                      workers=args.workers, assignment_workers=args.assignment_workers)
        validate_config(config)
        config_path = args.output / 'configs' / (name + '.json')
        output = args.output / 'runs' / name
        entry = REPO / 'module_screening2' / FOLDERS[experiment] / 'PUFM-main/screening.py'
        command = [sys.executable, '-u', '-B', str(entry), action,
                   '--config', str(config_path), '--root', str(args.root), '--baseline-checkpoint',
                   str(args.baseline_checkpoint), '--device', args.device, '--cpu-threads', str(args.cpu_threads),
                   '--output', str(output)]
        if args.fps_backend:
            command += ['--fps-backend', args.fps_backend]
        if args.fused_optimizer and action == 'train':
            command.append('--fused-optimizer')
        if args.limit_batches and action == 'train':
            command += ['--limit-batches', str(args.limit_batches)]
        if args.limit_samples:
            command += ['--limit-samples', str(args.limit_samples)]
        plan.append({'name': name, 'experiment': experiment, 'command': action, 'config': config,
                     'config_path': str(config_path), 'output': str(output), 'argv': command,
                     'expected_debug': bool(args.limit_samples or (args.limit_batches and action == 'train'))})
    return plan


def preflight(args, plan):
    from module_screening2.protocol import BASELINE_SHA256
    for name in ('base_manifest.json', 'generation_config.json', 'dataset_splits.json'):
        if not (args.root / 'metadata' / name).is_file():
            raise FileNotFoundError(f'V3 metadata missing: {name}')
    baseline_digest = file_hash(args.baseline_checkpoint)
    if baseline_digest != BASELINE_SHA256:
        raise ValueError('Pinned original baseline SHA256 mismatch')
    budgets = {(t['config']['epochs'], t['config']['batch_size'], t['config']['lr'],
                t['config']['seed'], t['config']['validation_every']) for t in plan if t['command'] == 'train'}
    if len(budgets) > 1:
        raise ValueError('All trained candidates/control require identical budgets/LR/seed/selection frequency')
    # Hash actual validation H5 files, addressing the old metadata-only limitation.
    manifest = read_json(args.root / 'metadata/base_manifest.json')
    validation = [args.root / 'processed/validation' / Path(e['path']).name for e in manifest['groups']['validation']]
    for path in validation:
        if not path.is_file():
            raise FileNotFoundError(path)
    sources = list((REPO / 'module_screening2').rglob('*.py'))
    sources += list((REPO / 'adaptive_resampling').glob('*.py')) + [REPO / 'run_subset_eval.py']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output.parent).free
    minimum = 1024**3 if args.limit_batches or args.limit_samples else 12 * 1024**3
    if free < minimum:
        raise RuntimeError(f'Insufficient output filesystem space: {free / 1024**3:.2f} GiB; need {minimum / 1024**3:g} GiB')
    import torch
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable')
    return {'baseline_sha256': baseline_digest,
            'metadata_sha256': {name: file_hash(args.root / 'metadata' / name) for name in
                ('base_manifest.json', 'generation_config.json', 'dataset_splits.json')},
            'validation_files_sha256': {p.relative_to(args.root).as_posix(): file_hash(p) for p in validation},
            'sources_sha256': {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(sources)},
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
            'fps_backend': args.fps_backend or ('cuda' if args.device.startswith('cuda') else 'torch')}


def verify_artifacts(task, identity):
    output = Path(task['output'])
    if task['command'] == 'check':
        report = read_json(output / 'summary.json')
        if report['status'] != 'passed' or not report['real_patch'] or not report['whole_shape'] or not report['baseline_loaded']:
            raise ValueError('Real-patch/whole-shape/pretrained-baseline checks did not pass')
        if set(report['copies']) != set(FOLDERS) or len(report['reports']) != 14:
            raise ValueError('Interface checks must cover all copies and registered variants')
        return
    if task['command'] == 'train':
        import torch
        for name in ('last.pt', 'best.pt'):
            saved = torch.load(output / name, map_location='cpu', weights_only=False)
            if (saved.get('schema') != 'pufm-module2-checkpoint-v1' or saved.get('config') != task['config']
                    or not saved.get('training_complete')
                    or saved.get('selection_completed_epochs') != task['config']['epochs']
                    or saved.get('partial_debug_run') != task['expected_debug']
                    or saved.get('baseline_sha256') != identity['baseline_sha256']
                    or saved['identity']['metadata_sha256'] != identity['metadata_sha256']):
                raise ValueError(f'{task["name"]}: checkpoint completion/identity check failed')
            if name == 'last.pt' and saved['completed_epochs'] != task['config']['epochs']:
                raise ValueError('Training stopped before the planned final epoch')
            if not all(torch.isfinite(v).all() for v in saved['model'].values()):
                raise ValueError('Non-finite model state')
            if any(identity['sources_sha256'].get(k) != v for k, v in saved['identity']['code_sha256'].items()):
                raise ValueError('Checkpoint source differs from the queue source snapshot')
        best = read_json(output / 'best.json')
        summary = read_json(output / f'validation/epoch_{best["epoch"]:04d}/summary.json')
    else:
        summary = read_json(output / 'summary.json')
    if (summary.get('config') != task['config'] or summary.get('split') != 'validation'
            or summary.get('partial_debug_run') != bool('--limit-samples' in task['argv'])
            or summary.get('baseline_sha256') != identity['baseline_sha256']
            or summary.get('metadata_sha256') != identity['metadata_sha256'] or len(summary['conditions']) != 7):
        raise ValueError(f'{task["name"]}: validation identity/completion check failed')
    if not summary['partial_debug_run'] and (summary['samples'] != 2800 or summary['objects'] != 100):
        raise ValueError('Formal screening must cover 100 validation objects x 4 patches x 7 conditions')
    if any(identity['sources_sha256'].get(k) != v for k, v in summary['code_sha256'].items()):
        raise ValueError('Validation source differs from the queue source snapshot')


def verify_sources(identity):
    for relative, digest in identity.get('sources_sha256', {}).items():
        if file_hash(REPO / relative) != digest:
            raise ValueError(f'Source modified while the queue was running: {relative}')


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
                verify_sources(identity)
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
                    elif (task_output / 'summary.json').is_file():
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
                verify_sources(identity)
                verify(task, identity)
                active.update(status='completed', completed_unix=time.time())
                write_json(state_path, state)
                print(f'[{index}/{len(plan)}] COMPLETED {task["name"]}', flush=True)
            collect(output)
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
    parser.add_argument('--output', type=Path, default=REPO / 'server_runs/patch32/module_screening2/sequential15_v1')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--assignment-workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fps-backend', choices=['cuda', 'torch'])
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--tasks', nargs='+', choices=[t[0] for t in DEFAULT_TASKS])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-checks', action='store_true', help='Use only after separately passing check.py in this exact environment')
    parser.add_argument('--limit-batches', type=int, default=0)
    parser.add_argument('--limit-samples', type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.assignment_workers, args.cpu_threads) < 1 or min(args.workers, args.limit_batches, args.limit_samples) < 0:
        parser.error('Invalid epoch/batch/worker budget')
    args.root, args.baseline_checkpoint, args.output = args.root.resolve(), args.baseline_checkpoint.resolve(), args.output.resolve()
    if (REPO / 'module_screening2') == args.output or (REPO / 'module_screening2') in args.output.parents:
        parser.error('Runtime results must stay outside the source directory')
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
        print('Queue interrupted. Resume with the original options plus --resume.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f'QUEUE FAILED: {error}', file=sys.stderr)
        raise SystemExit(1)

