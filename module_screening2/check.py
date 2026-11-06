"""Check physical copies, real patch gradients, repeatability and whole shapes."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from module_screening2.protocol import FOLDERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--baseline-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real-patch', action='store_true')
    parser.add_argument('--whole-shape', action='store_true')
    parser.add_argument('--fps-backend', choices=['cuda', 'torch'])
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--experiments', nargs='+', choices=FOLDERS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / folder / 'PUFM-main/models/diffusion.py' for folder in FOLDERS]
    if len({p.stat().st_ino for p in paths}) != len(paths) or any(p.is_symlink() for p in paths):
        raise AssertionError('PUFM copies must be physical files, not hard/symbolic links')
    reports = []
    selected = args.experiments or list(FOLDERS)
    variants = [(name, None) for name in selected]
    if not args.experiments:
        variants += [('pointsp_wrs', 'uniform_random'), ('dy_density', 'dy_density_high'),
                     ('dwc_nmm', 'dwc_no_contrastive')]
    for experiment, variant in variants:
        name = variant or experiment
        report = args.output / (name + '.json')
        command = [sys.executable, '-B', str(ROOT / experiment / 'PUFM-main/screening.py'), 'smoke',
                   '--device', args.device, '--report', str(report.resolve()), '--cpu-threads', str(args.cpu_threads)]
        if variant:
            command += ['--variant', variant]
        for flag in ['root', 'baseline_checkpoint', 'fps_backend']:
            value = getattr(args, flag)
            if value is not None:
                command += ['--' + flag.replace('_', '-'), str(value)]
        if args.real_patch:
            command.append('--real-patch')
        if args.whole_shape:
            command.append('--whole-shape')
        subprocess.run(command, check=True)
        reports.append(json.loads(report.read_text(encoding='utf-8')))
    if len({r['input_sha256'] for r in reports}) != 1 or len({r['baseline_prediction_sha256'] for r in reports}) != 1:
        raise AssertionError('Disabled modules differ between copies on the same baseline/input')
    summary = {'status': 'passed', 'reports': reports, 'copies': selected,
               'real_patch': args.real_patch, 'whole_shape': args.whole_shape,
               'baseline_loaded': all(r['baseline_loaded'] for r in reports)}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'{len(reports)} checks passed; all disabled copies give the same baseline output', flush=True)


if __name__ == '__main__':
    main()
