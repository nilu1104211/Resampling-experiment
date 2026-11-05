"""Isolated V3 module screening: train, validate, calibrate, predict, or smoke.

Run this copy in its own Python process (or use module_screening/run.py).
The historical training/evaluation programs are never invoked by this entry.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time

LOCAL = Path(__file__).resolve().parent
REPO = LOCAL.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LOCAL))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def code_identity():
    paths = list(LOCAL.rglob('*.py')) + list((REPO / 'adaptive_resampling').glob('*.py'))
    paths += [REPO / 'run_subset_eval.py']
    return {p.relative_to(REPO).as_posix(): file_hash(p) for p in paths if '__pycache__' not in p.parts}


def seed_all(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def read_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def data_identity(root):
    return {name: file_hash(Path(root) / 'metadata' / name) for name in
            ('base_manifest.json', 'generation_config.json', 'dataset_splits.json')}


def baseline_path(args):
    path = Path(args.baseline_checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Original V3 PUFM checkpoint missing: {path}; pass --baseline-checkpoint')
    return path


def make_model(args, config, *, initialize=True):
    from models.screening_model import ScreeningPUFM, load_baseline, validate_times
    model = ScreeningPUFM(config).to(args.device)
    saved = None
    if initialize and file_hash(baseline_path(args)) != config['baseline_sha256']:
        raise ValueError('All seven copies must start from the pinned epoch_0030.pt baseline SHA256')
    if args.checkpoint or args.resume:
        path = Path(args.resume or args.checkpoint)
        saved = read_checkpoint(path)
        if saved.get('schema') != 'pufm-module-checkpoint-v1' or saved.get('config') != config:
            raise ValueError('Checkpoint must belong to this exact experiment, variant, and configuration')
        if not args.resume and (saved.get('partial_debug_run') or not saved.get('training_complete')) and not args.allow_debug:
            raise ValueError('Incomplete/debug checkpoints require --allow-debug')
        model.load_state_dict(saved['model'], strict=True)
    elif initialize:
        path = baseline_path(args)
        load_baseline(model, read_checkpoint(path))
    if saved is not None and not args.resume:
        if saved['identity']['baseline_sha256'] != file_hash(baseline_path(args)):
            raise ValueError('Candidate was initialized from a different baseline checkpoint')
        if saved['identity']['metadata_sha256'] != data_identity(args.root):
            raise ValueError('Candidate checkpoint belongs to different V3 data metadata')
        if saved['identity']['code_sha256'] != code_identity():
            raise ValueError('Candidate source changed since training; preserve the trained source snapshot')
    if args.refine_iterations is not None:
        if not config['variant'].startswith('m6') or args.command not in {'validate', 'predict', 'smoke'}:
            raise ValueError('--refine-iterations is an M6 inference ablation only')
        model.candidate.inference_steps = args.refine_iterations
    if config['variant'] == 'm7b' and args.command not in {'calibrate', 'smoke'}:
        if not args.schedule:
            raise ValueError('M7b needs --schedule calibrated with this original PUFM on training data')
        schedule = json.loads(Path(args.schedule).read_text(encoding='utf-8'))
        if schedule.get('split') != 'train' or schedule.get('baseline_sha256') != file_hash(baseline_path(args)):
            raise ValueError('ATS schedule must be trained-data calibration bound to this PUFM checkpoint')
        if schedule.get('partial_debug_run') and not args.allow_debug:
            raise ValueError('Debug ATS calibration requires --allow-debug')
        if schedule.get('metadata_sha256') != data_identity(args.root):
            raise ValueError('ATS schedule belongs to different V3 metadata')
        if schedule.get('config') != config or schedule.get('code_sha256') != code_identity():
            raise ValueError('ATS schedule configuration/source has changed; recalibrate in a new run')
        model.times = validate_times(schedule['times'])
    return model, saved


def training_dataset(args, config, *, supervision=False):
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    return AdaptiveResamplingDataset(args.root, 'train', training_mode='mixed',
        include_supervision=supervision, augment_rotation=True,
        train_patches_per_object=config['train_patches_per_object'])


def batch_to(batch, device):
    import torch
    if isinstance(batch, dict):
        return {k: batch_to(v, device) for k, v in batch.items()}
    return batch.to(device).float() if isinstance(batch, torch.Tensor) else batch


def aggregate(rows):
    import numpy as np
    if not rows:
        raise ValueError('No samples were evaluated')
    keys = sorted({key for row in rows for key, val in row.items()
                   if isinstance(val, (int, float)) and key != 'sample_index'})
    by_object = {}
    for row in rows:
        by_object.setdefault((row['condition'], row['object_id']), []).append(row)
    objects = [{'condition': c, 'object_id': o, **{k: float(np.mean([r[k] for r in group if k in r]))
                for k in keys if any(k in r for r in group)}}
               for (c, o), group in sorted(by_object.items())]
    conditions = {}
    for condition in sorted({r['condition'] for r in objects}):
        group = [r for r in objects if r['condition'] == condition]
        conditions[condition] = {k: float(np.mean([r[k] for r in group if k in r]))
                                 for k in keys if any(k in r for r in group)}
    macro = {k: float(np.mean([r[k] for r in conditions.values() if k in r])) for k in keys}
    return {'aggregation': 'patches within object, objects within condition, seven conditions equally weighted',
            'samples': len(rows), 'objects': len({r['object_id'] for r in rows}),
            'conditions': conditions, 'macro': macro,
            'metric_condition_counts': {k: sum(k in r for r in conditions.values()) for k in keys}}, objects


def evaluate(model, args, output, *, split='validation', save_predictions=False):
    import numpy as np
    import torch
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    from adaptive_resampling.v3_metrics import evaluate_sample
    from run_subset_eval import CONDITIONS
    from types import SimpleNamespace
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Evaluation output already exists: {output}')
    output.mkdir(parents=True)
    model.eval()
    plan = None
    if split != 'validation':
        from run_subset_eval import make_plan, validate_supplement, evaluation_dataset, flatten
        plan_args = SimpleNamespace(repo=REPO, root=Path(args.root), output=output, split=split,
            objects=80, patches_per_object=None, seed=21, conditions=CONDITIONS, models=['pufm'],
            calibration_samples=256, fps_backend=os.environ.get('V3_FPS_BACKEND', 'auto'),
            patch_batch_size=1, schedule=None, p2f=False, device=args.device,
            pufm_checkpoint=baseline_path(args), whole_data_root=args.whole_data_root)
        plan = make_plan(plan_args)
        validate_supplement(plan)
    rows, diagnostics = [], []
    for condition in CONDITIONS:
        if plan is None:
            ds = AdaptiveResamplingDataset(args.root, 'validation', condition=condition,
                include_supervision=True, augment_rotation=False)
            indices = list(range(len(ds)))
        else:
            ds = evaluation_dataset(plan, condition, include_supervision=True)
            indices = [i for i, _ in flatten(plan['selection'])]
        if args.limit_samples:
            indices = indices[:args.limit_samples]
        predictions = []
        ids = []
        for index in indices:
            sample = ds[index]
            observed = torch.as_tensor(sample['input_xyz'], device=args.device).float().unsqueeze(0)
            if observed.shape[1] != (256 if split == 'validation' else 2048):
                raise ValueError('Unexpected V3 input point budget')
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                prediction, diag = model(observed, return_diagnostics=True)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {'condition': condition, 'object_id': sample['object_id'], 'sample_id': sample['sample_id'],
                   'sample_index': index, **evaluate_sample(prediction[0], sample), 'prediction_seconds': elapsed}
            rows.append(row)
            ids.append(sample['sample_id'])
            if save_predictions:
                from adaptive_resampling.v3_dataset import inverse_prediction
                predictions.append(inverse_prediction(prediction[0], sample).astype('f4'))
            diagnostics.append({'sample_id': sample['sample_id'], **{k: v.detach().cpu().tolist() if hasattr(v, 'detach') else v for k, v in diag.items()}})
        if save_predictions:
            np.savez_compressed(output / f'{condition}.npz', prediction=np.stack(predictions), sample_id=np.asarray(ids))
        ds.close()
    summary, objects = aggregate(rows)
    summary.update(split=split, partial_debug_run=bool(args.limit_samples),
                   metadata_sha256=data_identity(args.root), code_sha256=code_identity(),
                   config=model.config, times=model.times,
                   refine_iterations=args.refine_iterations,
                   baseline_sha256=file_hash(baseline_path(args)),
                   checkpoint_sha256=file_hash(args.checkpoint or args.resume) if (args.checkpoint or args.resume) else None,
                   schedule_sha256=file_hash(args.schedule) if args.schedule else None,
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   extra_parameter_count=sum(p.numel() for p in model.candidate.parameters()) if model.candidate is not None else 0,
                   timing_scope='prediction only; includes preprocessing/refinement, excludes metrics/loading; no standardized warmup')
    for name, records in [('per_sample.csv', rows), ('per_object.csv', objects)]:
        with (output / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
            writer.writeheader()
            writer.writerows(records)
    write_json(output / 'summary.json', summary)
    write_json(output / 'diagnostics.json', diagnostics)
    if plan is not None:
        write_json(output / 'frozen_plan.json', plan)
    return summary


def train(model, saved, args, config):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from adaptive_resampling.v3_dataset import EpochSampler
    from adaptive_resampling.v3_optimizer import make_optimizer
    from adaptive_resampling.v3_training import require_training_profile
    if config['variant'] in {'b0', 'm7a', 'm7b'}:
        raise ValueError('B0/M7 do not train a network')
    ds = training_dataset(args, config, supervision=config['variant'].startswith('m5'))
    sampler = EpochSampler(ds)
    loader = DataLoader(ds, sampler=sampler, batch_size=config['batch_size'], num_workers=config['workers'])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(torch.optim.Adam, params, lr=config['lr'], fused=args.fused_optimizer)
    output = Path(args.output)
    identity = {'baseline_sha256': file_hash(baseline_path(args)), 'metadata_sha256': data_identity(args.root),
                'code_sha256': code_identity(), 'config': config}
    start, best, history = 0, float('inf'), []
    if args.resume:
        if saved['identity'] != identity:
            raise ValueError('Resume requires identical baseline, data, code, and configuration')
        require_training_profile(saved, ds.training_profile)
        if Path(saved['output']).resolve() != output.resolve():
            raise ValueError('Resume into the original run directory; do not mix output histories')
        optimizer.load_state_dict(saved['optimizer'])
        start, best, history = saved['completed_epochs'], saved['best_score'], saved['history']
        torch.set_rng_state(saved['rng']['torch'].cpu())
        np.random.set_state(saved['rng']['numpy'])
        random.setstate(saved['rng']['python'])
        if torch.cuda.is_available() and saved['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['rng']['cuda']])
    elif output.exists():
        raise FileExistsError(f'Run already exists: {output}. Choose a new --output or --resume.')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', {**identity, 'training_profile': ds.training_profile,
        'device': args.device, 'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        'torch_version': torch.__version__, 'partial_debug_run': bool(args.limit_batches or args.limit_samples),
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in params),
        'trainable_names': [name for name, p in model.named_parameters() if p.requires_grad],
        'fused_optimizer': args.fused_optimizer})
    for epoch in range(start, config['epochs']):
        model.train()
        sampler.set_epoch(epoch)
        total, batches, seen = 0., 0, 0
        started = time.perf_counter()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            batch = batch_to(batch, args.device)
            if batch['input_xyz'].shape[1:] != (256, 3) or batch['gt_xyz'].shape[1:] != (1024, 3):
                raise ValueError('Training must retain the public 256 -> 1024 patch protocol')
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=True)
            if grad_norm == 0:
                raise RuntimeError('All trainable gradients are zero')
            optimizer.step()
            batches += 1
            seen += len(batch['input_xyz'])
            total += float(loss.detach()) * len(batch['input_xyz'])
            if args.limit_batches and batches >= args.limit_batches:
                break
        elapsed = time.perf_counter() - started
        record = {'epoch': epoch + 1, 'loss': total / seen, 'batches': batches, 'seen_patches': seen,
                  'optimizer_steps': batches, 'training_seconds': elapsed,
                  'peak_cuda_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None}
        selection = None
        if (epoch + 1) % config['validation_every'] == 0 or epoch + 1 == config['epochs']:
            selection = evaluate(model, args, output / f'validation/epoch_{epoch+1:04d}')
            record['validation'] = selection['macro']
        history.append(record)
        score = selection['macro']['cd_l1'] if selection else float('inf')
        improved = selection is not None and selection['macro']['f_score'] >= config['fscore_floor'] and score < best
        if improved:
            best = score
        payload = {'schema': 'pufm-module-checkpoint-v1', 'config': config, 'identity': identity,
            'baseline_sha256': identity['baseline_sha256'], 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'completed_epochs': epoch + 1,
            'training_complete': epoch + 1 == config['epochs'], 'selection_completed_epochs': epoch + 1,
            'partial_debug_run': bool(args.limit_batches or args.limit_samples),
            'training_profile': ds.training_profile, 'best_score': best, 'history': history, 'output': str(output.resolve()),
            'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        torch.save(payload, output / 'last.pt')
        if improved:
            torch.save(payload, output / 'best.pt')
            write_json(output / 'best.json', {'epoch': epoch + 1, 'validation': selection['macro']})
        with (output / 'train.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    if (output / 'best.pt').is_file():
        chosen = read_checkpoint(output / 'best.pt')
        chosen.update(training_complete=True, selection_completed_epochs=config['epochs'])
        torch.save(chosen, output / 'best.pt')
    ds.close()


def calibrate(model, args, config):
    import numpy as np
    import torch
    from adaptive_resampling.v3_transport import assignment_indices
    from models.adaptive_ops import gather
    if config['experiment'] != 'M7' or config['variant'] != 'm7b':
        raise ValueError('Only M7b calibrates ATS; no other copy adds a time schedule')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Calibration output already exists: {output}')
    ds = training_dataset(args, config)
    profile = ds.training_profile
    count = min(config['calibration_samples'], len(ds))
    indices = sorted(sorted(range(len(ds)), key=lambda i: signature([config['seed'], 'calibration', i]))[:count])
    grid = np.linspace(0, 1, config['schedule_grid'])
    sums = np.zeros(len(grid))
    actual = []
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, count, config['batch_size']):
            selected = indices[batch_start:batch_start + config['batch_size']]
            samples = [ds[(0, index)] for index in selected]
            points = torch.as_tensor(np.stack([s['input_xyz'] for s in samples]), device=args.device).float()
            target = torch.as_tensor(np.stack([s['gt_xyz'] for s in samples]), device=args.device).float()
            state = model.prepare(points)
            initial = state['initial'] + .01 * torch.randn_like(state['initial'])
            matched = gather(target, assignment_indices(initial, target))
            for j, t in enumerate(grid):
                query = (1 - t) * initial + t * matched
                velocity = model.velocity(query, query.new_full((len(query),), float(t)), state)
                sums[j] += float((velocity - (matched - initial)).square().mean()) * len(query)
            actual.extend(selected)
            print(json.dumps({'calibrated_samples': len(actual), 'planned_samples': count}), flush=True)
            if args.limit_batches and len(actual) >= args.limit_batches * config['batch_size']:
                break
    from models.candidate import build_schedule
    errors = sums / len(actual)
    times, cdf = build_schedule(grid, errors, floor=config['schedule_density_floor'])
    output.mkdir(parents=True)
    write_json(output / 'schedule.json', {'schema': 'pufm-ats-v1', 'split': 'train',
        'times': times, 'grid': grid.tolist(), 'mse': errors.tolist(), 'cdf': cdf,
        'config': config, 'baseline_sha256': file_hash(baseline_path(args)),
        'training_profile': profile, 'metadata_sha256': data_identity(args.root),
        'code_sha256': code_identity(), 'dataset_indices': actual,
        'selection_sha256': signature(actual), 'sample_ids': [ds[(0, i)]['sample_id'] for i in actual],
        'partial_debug_run': bool(args.limit_batches), 'network_training': False})
    ds.close()


def smoke(model, args, config):
    import torch
    from models.screening_model import ScreeningPUFM
    from models.utils import midpoint_interpolate
    from types import SimpleNamespace
    seed_all(config['seed'])
    model.eval()
    if args.real_patch:
        ds = training_dataset(args, config, supervision=True)
        sample = ds[(0, 0)]
        batch = batch_to({k: torch.as_tensor(sample[k])[None] for k in ('input_xyz', 'gt_xyz')}, args.device)
        batch['supervision'] = {'clean_input_xyz': torch.as_tensor(sample['supervision']['clean_input_xyz'], device=args.device)[None]}
        ds.close()
    else:
        points = torch.randn(1, 256, 3, device=args.device) * .2
        batch = {'input_xyz': points, 'gt_xyz': points.repeat_interleave(4, 1) + .001 * torch.randn(1, 1024, 3, device=args.device),
                 'supervision': {'clean_input_xyz': points}}
    # Compare the disabled candidate with the original direct five-step call.
    base_config = {**config, 'variant': 'b0'}
    baseline = ScreeningPUFM(base_config).to(args.device).eval()
    baseline.flow.load_state_dict(model.flow.state_dict(), strict=True)
    with torch.no_grad():
        initial = midpoint_interpolate(SimpleNamespace(up_rate=4), batch['input_xyz'].transpose(1, 2).contiguous())
        query = initial.clone()
        for i in range(5):
            query = query + baseline.flow(query, initial, query.new_full((len(query),), i / 5)) / 5
        disabled = baseline(batch['input_xyz'])
        torch.testing.assert_close(disabled, query.transpose(1, 2), atol=2e-6, rtol=2e-6)
        output, diag = model(batch['input_xyz'], return_diagnostics=True)
    params = [p for p in model.parameters() if p.requires_grad]
    loss = None
    if config['variant'] not in {'b0', 'm7a', 'm7b'}:
        frozen_before = {k: v.detach().clone() for k, v in model.flow.state_dict().items()} if model.freeze_flow else None
        model.train()
        optimizer = torch.optim.Adam(params, lr=config['lr'])
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(batch)
        loss.backward()
        gradients = [p.grad for p in params if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
            raise AssertionError('Candidate has no finite nonzero training gradient')
        if model.candidate is not None and not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.candidate.parameters()):
            raise AssertionError('Added module parameters did not receive a gradient')
        optimizer.step()
        if frozen_before is not None:
            if any(not torch.equal(v, model.flow.state_dict()[k]) for k, v in frozen_before.items()):
                raise AssertionError('Frozen PUFM parameters or BatchNorm buffers changed')
    result = {'experiment': config['experiment'], 'variant': config['variant'], 'status': 'passed',
        'input': list(batch['input_xyz'].shape), 'output': list(output.shape), 'finite': bool(torch.isfinite(output).all()),
        'baseline_equivalence': True, 'loss': float(loss.detach()) if loss is not None else None,
        'optimizer_step': loss is not None, 'frozen_flow_unchanged': model.freeze_flow,
        'baseline_loaded': bool(args.baseline_checkpoint and Path(args.baseline_checkpoint).is_file()),
        'real_v3_patch': args.real_patch, 'device': args.device, 'torch': torch.__version__}
    print(json.dumps(result), flush=True)
    if args.report:
        write_json(args.report, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'validate', 'predict', 'calibrate', 'smoke'])
    parser.add_argument('--config', type=Path, default=LOCAL.parent / 'config.json')
    parser.add_argument('--variant')
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('V3_DATA_ROOT', REPO / 'adaptive_resampling_data_v3')))
    parser.add_argument('--baseline-checkpoint', type=Path, default=Path(os.environ.get('PUFM_BASELINE_CHECKPOINT', REPO / 'server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt')))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--schedule', type=Path)
    parser.add_argument('--refine-iterations', type=int, choices=[1, 3], help='M6 inference ablation using the same trained distance model')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=['validation', 'pu1k_object', 'modelnet40_object'], default='validation')
    parser.add_argument('--whole-data-root', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit-batches', type=int, default=0)
    parser.add_argument('--limit-samples', type=int, default=0)
    parser.add_argument('--allow-debug', action='store_true')
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--real-patch', action='store_true', help='smoke only: load one actual training patch')
    parser.add_argument('--report', type=Path, help='smoke JSON report path')
    for name, typ in [('epochs', int), ('batch-size', int), ('lr', float), ('workers', int),
                      ('assignment-workers', int), ('validation-every', int)]:
        parser.add_argument('--' + name, type=typ)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    owner = json.loads((LOCAL.parent / 'config.json').read_text(encoding='utf-8'))['experiment']
    if config['experiment'] != owner:
        parser.error('Configuration belongs to another independent PUFM copy')
    if args.variant:
        config['variant'] = args.variant
    if config['variant'] not in config['allowed_variants']:
        parser.error('This variant belongs to another independent experiment')
    for name in ['epochs', 'batch_size', 'lr', 'workers', 'assignment_workers', 'validation_every']:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.resume and args.checkpoint:
        parser.error('Use either --resume or --checkpoint')
    if args.resume and args.command != 'train':
        parser.error('--resume is only valid for continuing the same training run')
    if args.command == 'train' and args.checkpoint:
        parser.error('Training starts from --baseline-checkpoint or continues with --resume')
    if args.command in {'validate', 'predict'} and config['variant'] not in {'b0', 'm7a', 'm7b'} and not args.checkpoint:
        parser.error('Candidate evaluation requires its own --checkpoint')
    if args.command == 'calibrate' and (args.checkpoint or args.resume):
        parser.error('M7 calibrates the original baseline checkpoint without network fine-tuning')
    if args.limit_batches < 0 or args.limit_samples < 0 or config['batch_size'] < 1 or config['epochs'] < 1 or config['validation_every'] < 1 or config['workers'] < 0 or config['lr'] <= 0:
        parser.error('Invalid training/evaluation budget')
    fixed = {'input_points': 256, 'output_points': 1024, 'flow_steps': 5,
             'train_patches_per_object': 32, 'training_mode': 'mixed', 'backbone': 'pufm_w_attn'}
    if any(config.get(k) != v for k, v in fixed.items()):
        parser.error('This screening preserves the pinned attention PUFM and V3 point/data budgets')
    if config['schedule_grid'] < 2 or config['calibration_samples'] < 1 or config['denoise_steps'] < 1 or config['refine_steps'] < 1 or config['geometry_neighbors'] < 3:
        parser.error('Invalid candidate/calibration configuration')
    if args.command in {'train', 'calibrate'} and args.split != 'validation':
        parser.error('Training/calibration do not use frozen test objects')
    import torch
    from adaptive_resampling.v3_transport import configure_assignment_workers
    configure_assignment_workers(config['assignment_workers'])
    seed_all(config['seed'])
    # A smoke run may use random weights; all other commands require the baseline.
    initialize = args.command != 'smoke' or args.baseline_checkpoint.is_file()
    model, saved = make_model(args, config, initialize=initialize)
    # Added random branch initialization must not shift the shared FM noise RNG.
    seed_all(config['seed'])
    if args.command == 'smoke':
        smoke(model, args, config)
        return
    if args.output is None:
        args.output = REPO / 'server_runs/patch32/module_screening' / config['experiment'].lower() / config['variant'] / args.command
    if args.command == 'train':
        train(model, saved, args, config)
    elif args.command == 'calibrate':
        calibrate(model, args, config)
    else:
        if args.command == 'validate' and args.split != 'validation':
            parser.error('Use predict explicitly for the final frozen subset')
        print(json.dumps(evaluate(model, args, args.output, split=args.split, save_predictions=args.command == 'predict')['macro']), flush=True)


if __name__ == '__main__':
    main()
