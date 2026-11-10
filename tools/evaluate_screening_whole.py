"""Evaluate isolated screening candidates on the previously frozen whole objects.

All model source/checkpoint metadata is audited before loading. Each evaluation
runs in a separate interpreter, and all outputs remain in a new output folder.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import socket
from contextlib import contextmanager

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = Path(os.environ.get('V3_DATA_ROOT', REPO / 'adaptive_resampling_data_v3')).resolve()
HISTORY = Path(os.environ.get('SCREENING_HISTORY_ROOT', REPO / 'server_runs')).resolve()
RUNS = Path(os.environ.get('SCREENING_RUNS_ROOT', HISTORY / 'patch32')).resolve()
BASE = Path(os.environ.get('PUFM_BASELINE_CHECKPOINT', RUNS / 'pufm_baseline_b8_w0_a8_fused/epoch_0030.pt')).resolve()
OUTPUT = HISTORY / 'whole_module_eval80_20261110_v2'
OLD = RUNS / 'module_screening/sequential10_v1'
NEW = RUNS / 'module_screening2/sequential15_v1'
REFERENCE = REPO / 'tools/whole_object_reference_20261110.json'
default_whole = REPO / 'modelnet40_object80_seed21'
if not default_whole.is_dir():
    default_whole = DATA / 'supplements/modelnet40_object80_seed21'
WHOLE = Path(os.environ.get('MODELNET40_WHOLE_DATA_ROOT', default_whole)).resolve()
SPLITS = ('pu1k_object', 'modelnet40_object')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_reference(conditions):
    reference = read_json(REFERENCE)
    if reference['schema'] != 'fixed-v3-six-model-reference-v1':
        raise ValueError('Unsupported historical reference schema')
    models = ('pufm', 'pufmpp', 'spu', 'ours', 'ours1', 'ounet')
    expected = {(model, split, condition) for model in models for split in SPLITS for condition in conditions}
    rows = reference['historical_rows']
    keys = {(r['model'], r['split'], r['condition']) for r in rows}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError('Historical reference has missing or duplicate results')
    for row in rows:
        if row['objects'] != 80 or row['samples'] != 80 or not all(
                math.isfinite(float(row[m])) for m in ('cd_l1', 'cd_l2_squared', 'hausdorff_l2', 'f_score')):
            raise ValueError('Invalid historical result in reference')
    for split in SPLITS:
        objects = reference['selection'][split]
        if len(objects) != 80 or len({o['object_id'] for o in objects}) != 80:
            raise ValueError(f'Invalid historical object selection: {split}')
    return reference


def check_reference_plan(plan, split, reference):
    if plan['metadata_sha256'] != reference['metadata_sha256'][split]:
        raise ValueError(f'V3 metadata differs from the six-model test: {split}')
    if split == 'modelnet40_object' and plan['supplement_manifest_sha256'] != reference['supplement_manifest_sha256']:
        raise ValueError('ModelNet40 supplement differs from the six-model test')
    selection = [{'object_id': o['object_id'], 'dataset_indices': o['dataset_indices']}
                 for o in plan['selection']['objects']]
    if selection != reference['selection'][split]:
        raise ValueError(f'Whole-object selection differs from the previous six-model test: {split}')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def jobs():
    tasks = []
    for generation, root in [('old', OLD), ('new', NEW)]:
        queue = read_json(root / 'queue.json')
        if queue['status'] != 'completed':
            raise ValueError(f'Incomplete source queue: {root}')
        for task in queue['plan']:
            if task['command'] == 'train' or (generation == 'new' and task['command'] == 'validate'):
                tasks.append({'generation': generation, 'name': task['name'], 'folder': task['experiment'],
                              'config': task['config'], 'checkpoint': str(root / 'runs' / task['name'] / 'best.pt')
                              if task['command'] == 'train' else None})
        if generation == 'old':
            for name, folder in [('m7_uniform_validation', 'm7'), ('m7_ats_validation', 'm7')]:
                task = next(t for t in queue['plan'] if t['name'] == name)
                tasks.append({'generation': generation, 'name': name, 'folder': folder,
                              'config': task['config'], 'checkpoint': None,
                              'schedule': str(root / 'runs/m7_calibrate/schedule.json') if name == 'm7_ats_validation' else None})
    for task in tasks:
        task['checkpoint_sha256'] = sha(task['checkpoint']) if task['checkpoint'] else None
        task['schedule_sha256'] = sha(task['schedule']) if task.get('schedule') else None
    return tasks


def audit_sources(record):
    changed, missing, newline = [], [], 0
    for relative, expected in record.items():
        path = REPO / relative
        if not path.is_file():
            missing.append(relative)
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() == expected:
            continue
        if hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == expected:
            newline += 1
        else:
            changed.append(relative)
    if changed or missing:
        raise ValueError(f'Source mismatch: changed={changed}, missing={missing}')
    return {'files': len(record), 'line_endings_only': newline}


def audit_task(task, metadata, baseline_hash):
    import torch
    config = task['config']
    if baseline_hash != config['baseline_sha256']:
        raise ValueError('Baseline checkpoint hash differs from the recorded source')
    saved = None
    if task['checkpoint']:
        if sha(task['checkpoint']) != task['checkpoint_sha256']:
            raise ValueError(f"Checkpoint changed: {task['name']}")
        saved = torch.load(task['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
        expected = 'pufm-module-checkpoint-v1' if task['generation'] == 'old' else 'pufm-module2-checkpoint-v1'
        if (saved.get('schema') != expected or saved.get('config') != config or
                not saved.get('training_complete') or saved.get('partial_debug_run') or
                saved['identity']['baseline_sha256'] != baseline_hash or saved['identity']['metadata_sha256'] != metadata):
            raise ValueError(f"Checkpoint identity verification failed: {task['name']}")
        if not all(torch.isfinite(value).all() for value in saved['model'].values()):
            raise ValueError(f"Non-finite checkpoint: {task['name']}")
        audit = audit_sources(saved['identity']['code_sha256'])
    else:
        source = (OLD if task['generation'] == 'old' else NEW) / 'runs' / task['name'] / 'summary.json'
        recorded = read_json(source)
        if (recorded['baseline_sha256'] != baseline_hash or recorded['metadata_sha256'] != metadata
                or recorded.get('config') != config or recorded.get('partial_debug_run')):
            raise ValueError(f"Frozen baseline/data identity changed: {task['name']}")
        audit = audit_sources(recorded['code_sha256'])
    if task.get('schedule'):
        if sha(task['schedule']) != task['schedule_sha256']:
            raise ValueError('ATS schedule file changed')
        schedule = read_json(task['schedule'])
        if (schedule['baseline_sha256'] != baseline_hash or schedule['metadata_sha256'] != metadata
                or schedule.get('config') != config or schedule.get('partial_debug_run')):
            raise ValueError('ATS schedule identity changed')
        audit_sources(schedule['code_sha256'])
    return saved, audit


def preflight(tasks, output):
    from run_subset_eval import CONDITIONS, make_plan, validate_supplement
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable in the active environment')
    metadata = {name: sha(DATA / 'metadata' / name) for name in
                ('base_manifest.json', 'generation_config.json', 'dataset_splits.json')}
    baseline_hash = sha(BASE)
    for task in tasks:
        audit_task(task, metadata, baseline_hash)
        print(f"PREFLIGHT {task['generation']}:{task['name']} passed", flush=True)
    reference = load_reference(CONDITIONS)
    for split in SPLITS:
        args = SimpleNamespace(repo=REPO, root=DATA, output=output, split=split, objects=80,
            patches_per_object=None, seed=21, conditions=CONDITIONS, models=['pufm'], calibration_samples=256,
            fps_backend='cuda', patch_batch_size=1, schedule=None, p2f=False, device='cuda',
            pufm_checkpoint=BASE, whole_data_root=WHOLE)
        plan = make_plan(args)
        validate_supplement(plan)
        check_reference_plan(plan, split, reference)
    print(f'PREFLIGHT all {len(tasks)} candidates and both frozen test datasets passed', flush=True)


@contextmanager
def queue_lock(output):
    lock = output / 'queue.lock'
    if lock.exists() and os.name != 'nt':
        previous = read_json(lock)
        if previous.get('host') == socket.gethostname():
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                lock.unlink()
            except PermissionError:
                pass
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f'Evaluation queue is locked: {lock}') from None
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'host': socket.gethostname()}, handle)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def worker(task, split, output, limit):
    os.environ['PUFM_POINTOPS'] = 'torch'
    os.environ['V3_FPS_BACKEND'] = 'cuda'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    sys.path.insert(0, str(REPO))
    if task['generation'] == 'old':
        from module_screening.run import FOLDERS
        local = REPO / 'module_screening' / FOLDERS[task['folder']] / 'PUFM-main'
    else:
        local = REPO / 'module_screening2' / task['folder'] / 'PUFM-main'
    sys.path.insert(0, str(local))
    import screening as engine
    from models.screening_model import ScreeningPUFM, load_baseline
    from run_subset_eval import CONDITIONS, make_plan, validate_supplement
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    torch.manual_seed(21)
    torch.cuda.manual_seed_all(21)
    config = task['config']
    metadata = engine.data_identity(DATA)
    saved, audit = audit_task(task, metadata, sha(BASE))
    model = ScreeningPUFM(config).to('cuda').eval()
    if saved is None:
        load_baseline(model, torch.load(BASE, map_location='cpu', weights_only=False, mmap=True))
    else:
        model.load_state_dict(saved['model'], strict=True)
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError('Non-finite parameters')
    if task.get('schedule'):
        schedule = read_json(task['schedule'])
        from models.screening_model import validate_times
        model.times = validate_times(schedule['times'])
    args = SimpleNamespace(root=DATA, baseline_checkpoint=BASE, checkpoint=Path(task['checkpoint']) if task['checkpoint'] else None,
                           resume=None, device='cuda', whole_data_root=WHOLE, limit_samples=limit, fps_backend='cuda',
                           cpu_threads=8, refine_iterations=None, schedule=Path(task['schedule']) if task.get('schedule') else None)
    plan_args = SimpleNamespace(repo=REPO, root=DATA, output=output, split=split, objects=80,
        patches_per_object=None, seed=21, conditions=CONDITIONS, models=['pufm'], calibration_samples=256,
        fps_backend='cuda', patch_batch_size=1, schedule=None, p2f=False, device='cuda',
        pufm_checkpoint=BASE, whole_data_root=WHOLE)
    plan = make_plan(plan_args)
    validate_supplement(plan)
    check_reference_plan(plan, split, load_reference(CONDITIONS))
    from adaptive_resampling import v3_metrics
    evaluate_sample = v3_metrics.evaluate_sample
    progress = {'count': 0}

    def evaluate_with_progress(prediction, sample, **kwargs):
        result = evaluate_sample(prediction, sample, **kwargs)
        progress['count'] += 1
        if progress['count'] % 80 == 0:
            print(f"{task['name']} {split}: {progress['count']}/560 samples", flush=True)
        return result

    v3_metrics.evaluate_sample = evaluate_with_progress
    started = time.time()
    if task['generation'] == 'old':
        result = engine.evaluate(model, args, output, split=split, save_predictions=False)
    else:
        result = engine.evaluate(model, args, output, split=split, save_predictions=False)
    expected = 7 * (min(80, limit) if limit else 80)
    if result['samples'] != expected or len(result['conditions']) != 7:
        raise ValueError('Evaluation sample count is incomplete')
    write_json(output / 'local_execution_manifest.json', {
        'task': task, 'split': split, 'partial_debug_run': bool(limit), 'source_audit': audit,
        'baseline_sha256': sha(BASE), 'checkpoint_sha256': sha(task['checkpoint']) if task['checkpoint'] else None,
        'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(), 'fps_backend': 'cuda', 'pointops': 'portable_torch',
                        'matmul_tf32': False, 'cudnn_tf32': False, 'deterministic_algorithms': True},
        'data_metadata_sha256': metadata, 'same_object_selection_as_original_six_models': True,
        'historical_reference_sha256': sha(REFERENCE),
        'elapsed_seconds': time.time() - started})
    print(json.dumps({'task': task['name'], 'split': split, 'samples': result['samples'],
                      'cd_l1': result['macro']['cd_l1'], 'elapsed': time.time() - started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit-samples', type=int, default=0)
    parser.add_argument('--jobs', nargs='+')
    parser.add_argument('--worker-task', type=Path)
    parser.add_argument('--split', choices=SPLITS)
    args = parser.parse_args()
    if args.worker_task:
        worker(read_json(args.worker_task), args.split, args.output, args.limit_samples)
        return
    if args.limit_samples < 0:
        parser.error('Invalid sample limit')
    all_jobs = jobs()
    if args.jobs:
        all_jobs = [job for job in all_jobs if job['generation'] + ':' + job['name'] in args.jobs]
        if len(all_jobs) != len(set(args.jobs)):
            parser.error('Unknown or duplicate job names')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'queue.json'
    config = {'jobs': all_jobs, 'splits': SPLITS, 'limit_samples': args.limit_samples,
              'data': str(DATA), 'whole_data_root': str(WHOLE), 'history_root': str(HISTORY),
              'baseline_sha256': sha(BASE), 'script_sha256': sha(Path(__file__)),
              'summary_script_sha256': sha(REPO / 'tools/summarize_screening_whole.py'),
              'historical_reference_sha256': sha(REFERENCE)}
    if manifest_path.exists():
        state = read_json(manifest_path)
        if state['config'] != json.loads(json.dumps(config)):
            raise ValueError('Use a new output directory when the evaluation plan changes')
    else:
        state = {'config': config, 'status': 'pending', 'tasks': {}}
    with queue_lock(args.output):
        execute(args, state, all_jobs, manifest_path)


def execute(args, state, all_jobs, manifest_path):
    try:
        preflight(all_jobs, args.output)
        state['status'] = 'running'
        write_json(manifest_path, state)
        for task in all_jobs:
            name = task['generation'] + '_' + task['name']
            taskfile = args.output / 'configs' / (name + '.json')
            write_json(taskfile, task)
            for split in SPLITS:
                key = name + '/' + split
                entry = state['tasks'].setdefault(key, {'attempts': 0, 'status': 'pending'})
                if entry['status'] == 'completed':
                    result = read_json(Path(entry['output']) / 'summary.json')
                    if result['samples'] != 7 * (min(80, args.limit_samples) if args.limit_samples else 80):
                        raise ValueError('Previously completed task is incomplete')
                    continue
                entry['attempts'] += 1
                output = args.output / 'runs' / name / split / f"attempt_{entry['attempts']:03d}"
                log = args.output / 'logs' / (name + '_' + split + f"_{entry['attempts']:03d}.log")
                log.parent.mkdir(parents=True, exist_ok=True)
                entry.update(status='running', output=str(output), log=str(log))
                write_json(manifest_path, state)
                command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--worker-task', str(taskfile),
                           '--split', split, '--output', str(output), '--limit-samples', str(args.limit_samples)]
                print(f'START {key}', flush=True)
                with log.open('w', encoding='utf-8') as handle:
                    subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=True)
                entry['status'] = 'completed'
                write_json(manifest_path, state)
                print(f'COMPLETED {key}', flush=True)
        if not args.limit_samples and len(all_jobs) == 32:
            from summarize_screening_whole import summarize
            summarize(args.output, state)
        state['status'] = 'completed'
        write_json(manifest_path, state)
    except BaseException as error:
        state['status'] = 'failed'
        if 'entry' in locals():
            entry.update(status='failed', error=str(error))
        write_json(manifest_path, state)
        raise


if __name__ == '__main__':
    main()
