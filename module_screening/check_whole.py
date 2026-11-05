"""Check 2048->8192 inference shapes and five flow calls, without test data/GT.

Auxiliary modules are randomly initialized: this checks interfaces, not quality.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys


def worker(args):
    import hashlib
    import numpy as np
    import torch
    project = Path(__file__).resolve().parent / args.worker / 'PUFM-main'
    sys.path.insert(0, str(project.parents[2]))
    sys.path.insert(0, str(project))
    from models.screening_model import ScreeningPUFM, load_baseline, validate_times
    config = json.loads((project.parent / 'config.json').read_text(encoding='utf-8'))
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    model = ScreeningPUFM(config).to(args.device).eval()
    with args.baseline_checkpoint.open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
        if digest.hexdigest() != config['baseline_sha256']:
            raise ValueError('Baseline SHA mismatch')
    load_baseline(model, torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=False))
    if config['experiment'] == 'M7':
        model.times = validate_times([0., .05, .15, .45, .75, 1.])
    calls = []
    hook = model.flow.register_forward_pre_hook(lambda module, inputs: calls.append(1))
    points = torch.randn(1, 2048, 3, generator=torch.Generator().manual_seed(21)).to(args.device) * .2
    with torch.no_grad():
        prediction = model(points)
    hook.remove()
    if prediction.shape != (1, 8192, 3) or not torch.isfinite(prediction).all() or len(calls) != 5:
        raise AssertionError('Whole-object shape/finite/flow-call check failed')
    result = {'experiment': config['experiment'], 'input': list(points.shape), 'output': list(prediction.shape),
        'finite': True, 'flow_evaluations': len(calls), 'input_sha256': hashlib.sha256(points.cpu().numpy().tobytes()).hexdigest(),
        'test_data_used': False, 'ground_truth_used': False, 'auxiliary_weights_trained': False,
        'device': args.device, 'status': 'passed'}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        if args.output.exists():
            parser.error('Use a new output directory')
        args.output.mkdir(parents=True)
        for folder in sorted(Path(__file__).resolve().parent.iterdir()):
            if (folder / 'PUFM-main').is_dir():
                subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', folder.name,
                    '--device', args.device, '--baseline-checkpoint', str(args.baseline_checkpoint.resolve()),
                    '--output', str((args.output / (folder.name + '.json')).resolve())], check=True)
