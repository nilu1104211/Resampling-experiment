"""Run each candidate's finite/gradient/isolation smoke check sequentially."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--baseline-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real-patch', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory for each verification run')
    args.output.mkdir(parents=True)
    results = []
    for name in ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']:
        report = (args.output / f'{name}.json').resolve()
        command = [sys.executable, '-B', str(Path(__file__).with_name('run.py')), name, 'smoke',
                   '--device', args.device, '--report', str(report)]
        if args.root:
            command += ['--root', str(args.root.resolve())]
        if args.baseline_checkpoint:
            command += ['--baseline-checkpoint', str(args.baseline_checkpoint.resolve())]
        if args.real_patch:
            command.append('--real-patch')
        subprocess.run(command, check=True)
        results.append(json.loads(report.read_text(encoding='utf-8')))
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print('All seven independent checks passed', flush=True)
