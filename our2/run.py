"""Independent our2 training, validation and frozen whole-object evaluation."""
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
REPO = LOCAL.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LOCAL / 'PUFM-main'))
os.environ['PUFM_POINTOPS'] = 'torch'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    import torch
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def code_identity():
    paths = list(LOCAL.rglob('*.py')) + list((LOCAL / 'configs').glob('*.json')) + list((REPO / 'adaptive_resampling').glob('*.py'))
    paths += [REPO / 'run_subset_eval.py', LOCAL / 'config.py']
    return {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(paths)}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model):
    return signature({k: tensor_hash(v) for k, v in sorted(model.state_dict().items())})


def seed_all(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def configure_runtime(args):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.cpu_threads)
    os.environ['V3_FPS_BACKEND'] = args.fps_backend or ('cuda' if args.device.startswith('cuda') else 'torch')


def runtime_identity(args):
    import platform
    import torch
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
            'device': args.device, 'fps_backend': os.environ['V3_FPS_BACKEND'],
            'pointops': 'portable_torch', 'tf32': False, 'deterministic_algorithms': True,
            'cpu_threads': args.cpu_threads}


def read_checkpoint(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def data_identity(root):
    return {name: file_hash(Path(root) / 'metadata' / name) for name in
            ('base_manifest.json', 'generation_config.json', 'dataset_splits.json')}


def make_model(args, config, initialize=True):
    from models.dual import DualPUFM, load_baseline
    model = DualPUFM(config).to(args.device)
    saved = None
    if initialize and file_hash(args.baseline_checkpoint) != config['baseline_sha256']:
        raise ValueError('All copies require the pinned original epoch_0030.pt SHA256')
    if args.checkpoint or args.resume:
        saved = read_checkpoint(args.resume or args.checkpoint)
        if saved.get('schema') != 'our2-checkpoint-v1' or saved.get('config') != config:
            raise ValueError('Checkpoint belongs to a different copy/configuration')
        if not args.resume and (saved.get('partial_debug_run') or not saved.get('training_complete')) and not args.allow_debug:
            raise ValueError('Incomplete/debug checkpoints require --allow-debug')
        model.load_state_dict(saved['model'], strict=True)
        if saved['identity']['baseline_sha256'] != file_hash(args.baseline_checkpoint):
            raise ValueError('Candidate initialized from another baseline')
        if saved['identity']['metadata_sha256'] != data_identity(args.root):
            raise ValueError('Candidate data metadata changed')
        if saved['identity']['code_sha256'] != code_identity():
            raise ValueError('Candidate source changed since training')
        if saved['identity']['runtime'] != runtime_identity(args):
            raise ValueError('Runtime changed; use the exact recorded environment/backend')
    elif initialize:
        load_baseline(model, read_checkpoint(args.baseline_checkpoint))
    if args.command == 'train' and not args.resume:
        initialize_stage(model, args, config)
    return model, saved


def training_dataset(args, config):
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    return AdaptiveResamplingDataset(args.root, 'train', training_mode='mixed', include_supervision=True,
        augment_rotation=True, train_patches_per_object=config['train_patches_per_object'])


def batch_to(batch, device):
    import torch
    if isinstance(batch, dict):
        return {k: batch_to(v, device) for k, v in batch.items()}
    return batch.to(device).float() if isinstance(batch, torch.Tensor) else batch


def aggregate(rows):
    import numpy as np
    keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (float, int)) and k != 'sample_index'})
    grouped = {}
    for row in rows:
        grouped.setdefault((row['condition'], row['object_id']), []).append(row)
    objects = [{'condition': c, 'object_id': o, **{k: float(np.mean([r[k] for r in group if k in r]))
                for k in keys if any(k in r for r in group)}} for (c, o), group in sorted(grouped.items())]
    conditions = {}
    for condition in sorted({r['condition'] for r in objects}):
        group = [r for r in objects if r['condition'] == condition]
        conditions[condition] = {k: float(np.mean([r[k] for r in group if k in r]))
                                 for k in keys if any(k in r for r in group)}
    if len(conditions) != 7:
        raise ValueError('Selection requires all seven conditions')
    macro = {k: float(np.mean([r[k] for r in conditions.values() if k in r])) for k in keys}
    return {'aggregation': 'patches/object, objects/condition, seven conditions equally weighted',
            'samples': len(rows), 'objects': len({r['object_id'] for r in rows}),
            'conditions': conditions, 'macro': macro,
            'metric_condition_counts': {k: sum(k in r for r in conditions.values()) for k in keys}}, objects


def evaluate(model, args, output, split='validation', save_predictions=False):
    import numpy as np
    import torch
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset, inverse_prediction
    from adaptive_resampling.v3_metrics import evaluate_sample
    from run_subset_eval import CONDITIONS
    from types import SimpleNamespace
    output = Path(output)
    model.eval()
    identity = {'model_sha256': model_hash(model), 'config': model.config, 'metadata_sha256': data_identity(args.root),
                'code_sha256': code_identity(), 'runtime': runtime_identity(args), 'split': split,
                'limit_samples': args.limit_samples, 'save_predictions': save_predictions}
    marker = output / 'evaluation_identity.json'
    if output.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != identity:
            raise FileExistsError(f'Evaluation output belongs to another model/configuration: {output}')
        if (output / 'summary.json').is_file():
            return json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    write_json(marker, identity)
    plan = None
    if split != 'validation':
        from run_subset_eval import make_plan, validate_supplement, evaluation_dataset, flatten
        plan_args = SimpleNamespace(repo=REPO, root=args.root, output=output, split=split,
            objects=80, patches_per_object=None, seed=21, conditions=CONDITIONS, models=['pufm'],
            calibration_samples=256, fps_backend=os.environ['V3_FPS_BACKEND'], patch_batch_size=1,
            schedule=None, p2f=False, device=args.device, pufm_checkpoint=args.baseline_checkpoint,
            whole_data_root=args.whole_data_root)
        plan = make_plan(plan_args)
        validate_supplement(plan)
    rows, diagnostics = [], []
    warmed = False
    for condition in CONDITIONS:
        ds = (AdaptiveResamplingDataset(args.root, 'validation', condition=condition, include_supervision=True,
              augment_rotation=False) if plan is None else evaluation_dataset(plan, condition, include_supervision=True))
        indices = list(range(len(ds))) if plan is None else [i for i, _ in flatten(plan['selection'])]
        if args.limit_samples:
            indices = indices[:args.limit_samples]
        predictions, ids = [], []
        for index in indices:
            sample = ds[index]
            observed = torch.as_tensor(sample['input_xyz'], device=args.device).float().unsqueeze(0)
            if observed.shape[1] != (256 if split == 'validation' else 2048):
                raise ValueError('Input point budget changed')
            # Evaluation has its own RNG scope and cannot shift training RNG states.
            devices = [observed.device.index] if observed.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(signature([model.config['seed'], sample['sample_id']])[:8], 16))
                with torch.no_grad():
                    if not warmed:
                        model(observed)
                        warmed = True
                    if observed.is_cuda:
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    prediction, diag = model(observed, return_diagnostics=True)
                    if observed.is_cuda:
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
            row = {'condition': condition, 'object_id': sample['object_id'], 'sample_id': sample['sample_id'],
                   'sample_index': index, 'input_sha256': tensor_hash(observed), 'prediction_sha256': tensor_hash(prediction),
                   **evaluate_sample(prediction[0], sample), 'prediction_seconds': elapsed}
            if model.stage in {'m1_rel','m1_hole','m2_soft','joint_condition'}:
                batch = {'input_xyz':observed,'gt_xyz':torch.as_tensor(sample['gt_xyz'],device=args.device)[None],
                         'supervision':{k:torch.as_tensor(v,device=args.device).float()[None] for k,v in sample['supervision'].items()}}
                with torch.no_grad(): row['auxiliary_selection'] = float(model.training_loss(batch))
            rows.append(row)
            ids.append(sample['sample_id'])
            diagnostics.append({'sample_id': sample['sample_id'], **{k: v.detach().cpu().tolist()
                                if hasattr(v, 'detach') else v for k, v in diag.items()}})
            if save_predictions:
                predictions.append(inverse_prediction(prediction[0], sample).astype('f4'))
        if save_predictions:
            np.savez_compressed(output / f'{condition}.npz', prediction=np.stack(predictions), sample_id=np.asarray(ids))
        ds.close()
    summary, objects = aggregate(rows)
    summary.update(**identity, partial_debug_run=bool(args.limit_samples), baseline_sha256=file_hash(args.baseline_checkpoint),
                   checkpoint_sha256=file_hash(args.checkpoint or args.resume) if args.checkpoint or args.resume else None,
                   input_selection_sha256=signature([[r['condition'], r['sample_id'], r['input_sha256']] for r in rows]),
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   extra_parameter_count=sum(p.numel() for m in (model.m1,model.m2) if m is not None for p in m.parameters()),
                   timing_scope='prediction including adapter; first sample warmup excluded; loading/metrics excluded')
    for name, records in [('per_sample.csv', rows), ('per_object.csv', objects)]:
        with (output / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
            writer.writeheader()
            writer.writerows(records)
    write_json(output / 'diagnostics.json', diagnostics)
    if plan is not None:
        write_json(output / 'frozen_plan.json', plan)
    # Completion marker is written last, so interrupted evaluation can be replayed.
    write_json(output / 'summary.json', summary)
    return summary


def train(model, saved, args, config):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from adaptive_resampling.v3_dataset import EpochSampler
    from adaptive_resampling.v3_optimizer import make_optimizer
    from adaptive_resampling.v3_training import require_training_profile
    ds = training_dataset(args, config)
    loader = DataLoader(ds, sampler=EpochSampler(ds), batch_size=config['batch_size'], num_workers=config['workers'])
    sampler = loader.sampler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_stage_optimizer(model, args, config)
    output = args.output
    identity = {'baseline_sha256': file_hash(args.baseline_checkpoint), 'metadata_sha256': data_identity(args.root),
                'code_sha256': code_identity(), 'runtime': runtime_identity(args), 'config': config,
                'parent_checkpoints': parent_identity(args),
                'debug_limits': [args.limit_batches,args.limit_samples]}
    start, best, history = 0, float('inf'), []
    if args.resume:
        if saved['identity'] != identity or Path(saved['output']).resolve() != output.resolve():
            raise ValueError('Resume requires original source/data/runtime/options/output')
        require_training_profile(saved, ds.training_profile)
        optimizer.load_state_dict(saved['optimizer'])
        start, best, history = saved['completed_epochs'], saved['best_score'], saved['history']
        torch.set_rng_state(saved['rng']['torch'].cpu())
        np.random.set_state(saved['rng']['numpy'])
        random.setstate(saved['rng']['python'])
        if torch.cuda.is_available() and saved['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['rng']['cuda']])
    elif output.exists():
        raise FileExistsError(f'Use a fresh output or --resume: {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    write_json(output / 'manifest.json', {**identity, 'training_profile': ds.training_profile,
        'partial_debug_run': bool(args.limit_batches or args.limit_samples),
        'parameters': sum(p.numel() for p in model.parameters()), 'fused_optimizer': args.fused_optimizer,
        'trainable_parameters': sum(p.numel() for p in params)})
    def snapshot(completed):
        names = {id(p): name for name,p in model.named_parameters()}
        return {'schema': 'our2-checkpoint-v1', 'config': config, 'identity': identity,
            'baseline_sha256': identity['baseline_sha256'], 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_parameter_names': [[names[id(p)] for p in g['params']] for g in optimizer.param_groups],
            'completed_epochs': completed,
            'training_complete': completed == config['epochs'], 'selection_completed_epochs': completed,
            'partial_debug_run': bool(args.limit_batches or args.limit_samples),
            'training_profile': ds.training_profile, 'best_score': best, 'history': history, 'output': str(output.resolve()),
            'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    if not args.resume:
        initial_selection = evaluate(model,args,output / 'validation/epoch_0000')
        best = selection_score(initial_selection,config)
        save_checkpoint(output / 'best.pt',snapshot(0))
        write_json(output / 'best.json',{'epoch':0,'validation':initial_selection['macro']})
        # An interrupted first epoch can restart exactly; no extra checkpoint file.
        save_checkpoint(output / 'last.pt', snapshot(0))
    for epoch in range(start, config['epochs']):
        model.train()
        sampler.set_epoch(config['data_epoch_offset'] + epoch)
        total, batches, seen, components = 0., 0, 0, {}
        started = time.perf_counter()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            batch = batch_to(batch, args.device)
            if batch['input_xyz'].shape[1:] != (256, 3) or batch['gt_xyz'].shape[1:] != (1024, 3):
                raise ValueError('Training budget must remain 256 -> 1024')
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, config['gradient_clip'], error_if_nonfinite=True)
            if norm == 0:
                raise RuntimeError('All trainable gradients are zero')
            optimizer.step()
            batches += 1
            seen += len(batch['input_xyz'])
            total += float(loss.detach()) * len(batch['input_xyz'])
            for k, v in model.loss_diagnostics.items():
                components[k] = components.get(k, 0.) + v * len(batch['input_xyz'])
            if args.limit_batches and batches >= args.limit_batches:
                break
        record = {'epoch': epoch + 1, 'loss': total / seen, 'batches': batches, 'seen_patches': seen,
                  'optimizer_steps': batches, 'training_seconds': time.perf_counter() - started,
                  'peak_cuda_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                  **{k: v / seen for k, v in components.items()}}
        selection = None
        if (epoch + 1) % config['validation_every'] == 0 or epoch + 1 == config['epochs']:
            selection = evaluate(model, args, output / f'validation/epoch_{epoch + 1:04d}')
            record['validation'] = selection['macro']
        history.append(record)
        score = selection_score(selection,config) if selection else float('inf')
        improved = selection is not None and score < best
        if improved:
            best = score
        payload = snapshot(epoch + 1)
        if improved:
            save_checkpoint(output / 'best.pt', payload)
            write_json(output / 'best.json', {'epoch': epoch + 1, 'validation': selection['macro']})
        # Commit last only after best, so a crash cannot lose the selected state.
        save_checkpoint(output / 'last.pt', payload)
        # Rewrite from checkpoint history; interrupted epoch replay cannot duplicate rows.
        (output / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in history), encoding='utf-8')
        print(json.dumps(record), flush=True)
    if (output / 'best.pt').is_file():
        chosen = read_checkpoint(output / 'best.pt')
        chosen.update(training_complete=True, selection_completed_epochs=config['epochs'])
        save_checkpoint(output / 'best.pt', chosen)
    (output / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in history), encoding='utf-8')
    ds.close()


def selection_score(selection, config):
    key = 'auxiliary_selection' if config['stage'] in {'m1_rel','m1_hole','m2_soft','joint_condition'} else 'cd_l1'
    return selection['macro'][key]


def parent_identity(args):
    return {name: file_hash(path) for name in ('init','m1_checkpoint','m2_checkpoint')
            if (path := getattr(args,name,None)) is not None}


def initialize_stage(model,args,config):
    expected = {'m1_hole':'m1_rel','m1_refine':'m1_hole','m2_hard':'m2_soft','joint_finetune':'joint_condition'}
    if config['stage'] in expected and not args.init:
        raise ValueError(f"{config['stage']} requires --init from {expected[config['stage']]}")
    if config['stage'] == 'joint_condition' and not (args.m1_checkpoint and args.m2_checkpoint):
        raise ValueError('Joint requires both independent module checkpoints')
    for option,module in [('init',None),('m1_checkpoint',model.m1),('m2_checkpoint',model.m2)]:
        path = getattr(args,option)
        if path is None: continue
        saved = read_checkpoint(path)
        if saved.get('schema')!='our2-checkpoint-v1' or saved['baseline_sha256']!=config['baseline_sha256']:
            raise ValueError('Incompatible stage checkpoint')
        if saved['identity']['metadata_sha256'] != data_identity(args.root): raise ValueError('Parent data mismatch')
        if (saved.get('partial_debug_run') or not saved.get('training_complete')) and not args.allow_debug:
            raise ValueError('Debug/incomplete parent requires --allow-debug')
        previous = saved['config']
        for name in ('width','region_divisor','max_move','minimum','maximum','temperature'):
            if previous[name]!=config[name]: raise ValueError(f'Parent geometry changed: {name}')
        if option=='init':
            if config['stage'] in expected and previous['stage']!=expected[config['stage']]:
                raise ValueError('Unexpected predecessor stage')
            model.load_state_dict(saved['model'],strict=True)
        else:
            prefix = 'm1.' if option=='m1_checkpoint' else 'm2.'
            required_stage = 'm1_refine' if prefix=='m1.' else 'm2_hard'
            if previous['stage']!=required_stage or module is None: raise ValueError('Wrong independent module stage')
            module.load_state_dict({k[len(prefix):]:v for k,v in saved['model'].items() if k.startswith(prefix)},strict=True)
    # Joint starts from original flow and the two independently learned heads;
    # incompatible independently fine-tuned flow weights are never silently merged.


def make_stage_optimizer(model,args,config):
    import copy
    import torch
    groups=[]
    flow=[p for p in model.flow.parameters() if p.requires_grad]
    if flow: groups.append({'params':flow,'lr':config['lr'],'name':'flow'})
    branch=[p for name,p in model.named_parameters() if not name.startswith('flow.') and p.requires_grad]
    if branch: groups.append({'params':branch,'lr':config['module_lr'],'name':'modules'})
    optimizer=torch.optim.Adam(groups, fused=args.fused_optimizer)
    if flow and config['restore_baseline_optimizer'] and not args.resume:
        baseline=read_checkpoint(args.baseline_checkpoint)
        original=baseline['optimizer']
        ids=original['param_groups'][0]['params']
        if len(original['param_groups'])!=1 or len(ids)!=len(flow): raise ValueError('Baseline optimizer layout differs')
        # A staged parent may have a frozen or fine-tuned flow. Prefer its saved
        # flow moments when present; otherwise original baseline moments are used.
        if args.init:
            previous=read_checkpoint(args.init)
            named=previous.get('optimizer_parameter_names',[])
            lookup={name:pid for names,g in zip(named,previous['optimizer']['param_groups']) for name,pid in zip(names,g['params'])}
            state_by_name={name:previous['optimizer']['state'].get(pid) for name,pid in lookup.items()}
        else: state_by_name={}
        for (name,p),pid in zip(model.flow.named_parameters(),ids):
            source=state_by_name.get('flow.'+name) or original['state'].get(pid)
            if source:
                optimizer.state[p]={k:v.detach().clone().to(p.device if k!='step' or args.fused_optimizer else 'cpu')
                                    if torch.is_tensor(v) else copy.deepcopy(v) for k,v in source.items()}
    return optimizer


def main():
    from our2.config import DEFAULT,validate_config
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','validate','predict'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('V3_DATA_ROOT',REPO/'adaptive_resampling_data_v3')))
    parser.add_argument('--baseline-checkpoint',type=Path,default=Path(os.environ.get('PUFM_BASELINE_CHECKPOINT',REPO/'server_runs/patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt')))
    for key in ('checkpoint','resume','init','m1-checkpoint','m2-checkpoint','output','whole-data-root'):
        parser.add_argument('--'+key,type=Path)
    parser.add_argument('--split',choices=['validation','pu1k_object','modelnet40_object'],default='validation')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--fps-backend',choices=['cuda','torch'])
    parser.add_argument('--cpu-threads',type=int,default=8)
    parser.add_argument('--limit-batches',type=int,default=0)
    parser.add_argument('--limit-samples',type=int,default=0)
    parser.add_argument('--allow-debug',action='store_true')
    parser.add_argument('--fused-optimizer',action='store_true')
    for key in ('epochs','batch-size','seed'):
        parser.add_argument('--'+key,type=int)
    args=parser.parse_args()
    config={**DEFAULT,**json.loads(args.config.read_text(encoding='utf-8'))}
    for key in ('epochs','batch_size','seed'):
        if getattr(args,key) is not None: config[key]=getattr(args,key)
    validate_config(config)
    if args.command=='train' and (args.checkpoint or args.split!='validation'): parser.error('Training uses train/validation only')
    if args.command=='validate' and args.split!='validation': parser.error('Use predict for final object tests')
    if args.resume and (args.command!='train' or args.checkpoint): parser.error('Invalid resume')
    if args.command!='train' and not args.checkpoint and (config['enable_m1'] or config['enable_m2']): parser.error('Trained module checkpoint required')
    if min(args.limit_samples,args.limit_batches)<0 or args.cpu_threads<1: parser.error('Invalid limits')
    args.root,args.baseline_checkpoint=args.root.resolve(),args.baseline_checkpoint.resolve()
    args.output=(args.output or REPO/'server_runs/our2'/config['stage']/args.command).resolve()
    if args.output.is_relative_to(LOCAL): parser.error('Keep runtime output outside the our2 source directory')
    configure_runtime(args)
    from adaptive_resampling.v3_transport import configure_assignment_workers
    configure_assignment_workers(config['assignment_workers'])
    seed_all(config['seed'])
    model,saved=make_model(args,config)
    seed_all(config['seed'])
    if args.command=='train': train(model,saved,args,config)
    else: print(json.dumps(evaluate(model,args,args.output,args.split,args.command=='predict')['macro']),flush=True)


if __name__=='__main__': main()

