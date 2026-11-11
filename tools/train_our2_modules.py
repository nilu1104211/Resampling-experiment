"""Train the seven our2 module stages with the existing PUFM checkpoint."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from our2.config import DEFAULT
from our2.run import code_identity, data_identity, file_hash, write_json


BASELINE_EPOCH_OFFSETS = {
    DEFAULT['baseline_sha256']: DEFAULT['data_epoch_offset'],
    # pufm_common_b8_w0_a8_fused/epoch_0060.pt
    'a5051af124fba4112bb118196e2f353cfbc2f43ec8385ef1b2f1cb8cb1e71599': 60,
}

STAGES = (
    ('m1_rel', 0, {}),
    ('m1_hole', 1, {'--init': 'm1_rel'}),
    ('m1_refine', 2, {'--init': 'm1_hole'}),
    ('m2_soft', 0, {}),
    ('m2_hard', 1, {'--init': 'm2_soft'}),
    ('joint_condition', 5, {'--m1-checkpoint': 'm1_refine', '--m2-checkpoint': 'm2_hard'}),
    ('joint_finetune', 6, {'--init': 'joint_condition'}),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fps-backend', choices=('cuda', 'torch'), default='cuda')
    parser.add_argument('--epochs', type=int, default=DEFAULT['epochs'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT['batch_size'])
    parser.add_argument('--seed', type=int, default=DEFAULT['seed'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')

    root = args.root.resolve()
    baseline = args.baseline_checkpoint.resolve()
    output = args.output.resolve()
    if output.is_relative_to(REPO / 'our2'):
        parser.error('Keep outputs outside our2')
    baseline_hash = file_hash(baseline)
    if baseline_hash not in BASELINE_EPOCH_OFFSETS:
        parser.error('Expected the trained PUFM epoch_0060.pt or the epoch_0030.pt used by P0')
    data_epoch_offset = BASELINE_EPOCH_OFFSETS[baseline_hash]

    identity = {
        'root': str(root), 'baseline_checkpoint': str(baseline), 'output': str(output),
        'device': args.device, 'fps_backend': args.fps_backend, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'seed': args.seed, 'baseline_sha256': baseline_hash,
        'data_sha256': data_identity(root), 'training_source_sha256': code_identity(),
        'launcher_sha256': file_hash(__file__),
    }
    queue = output / 'queue.json'
    if args.resume:
        if not queue.is_file():
            parser.error(f'Cannot resume without {queue}')
    else:
        output.mkdir(parents=True)

    lock = output / 'queue.lock'
    lock_handle = lock.open('x', encoding='utf-8')
    try:
        with lock_handle:
            json.dump({'pid': os.getpid(), 'host': socket.gethostname()}, lock_handle)
        if args.resume:
            state = json.loads(queue.read_text(encoding='utf-8'))
            if state['identity'] != identity:
                raise ValueError('Resume requires the original paths, data, code and options')
        else:
            state = {'identity': identity, 'stages': [name for name, _, _ in STAGES], 'completed': []}
        state.update(status='running', current=None)
        write_json(queue, state)

        try:
            baseline_config = json.loads((REPO / 'our2' / 'configs' / 'baseline.json').read_text(encoding='utf-8'))
            baseline_config.update(baseline_sha256=baseline_hash, data_epoch_offset=data_epoch_offset,
                                   epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
            write_json(output / 'configs' / 'baseline.json', baseline_config)

            for number, (stage, offset, parents) in enumerate(STAGES, start=1):
                stage_output = output / stage
                if stage in state['completed']:
                    if not (stage_output / 'best.pt').is_file():
                        raise FileNotFoundError(f'Completed stage is missing {stage_output / "best.pt"}')
                    continue

                config = json.loads((REPO / 'our2' / 'configs' / f'{stage}.json').read_text(encoding='utf-8'))
                config.update(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                              baseline_sha256=baseline_hash,
                              data_epoch_offset=data_epoch_offset + offset * args.epochs)
                config_path = output / 'configs' / f'{stage}.json'
                write_json(config_path, config)
                command = [sys.executable, '-u', '-B', str(REPO / 'our2' / 'run.py'), 'train',
                           '--config', str(config_path), '--root', str(root),
                           '--baseline-checkpoint', str(baseline), '--output', str(stage_output),
                           '--device', args.device, '--fps-backend', args.fps_backend]
                for option, parent in parents.items():
                    command.extend((option, str(output / parent / 'best.pt')))
                if stage_output.exists():
                    last = stage_output / 'last.pt'
                    if not args.resume or not last.is_file():
                        raise FileExistsError(f'{stage_output} has no resumable last.pt; move it aside before retrying')
                    command.extend(('--resume', str(last)))

                state['current'] = stage
                write_json(queue, state)
                print(f'[{number}/{len(STAGES)}] {stage}: {args.epochs} epochs', flush=True)
                subprocess.run(command, cwd=REPO, check=True)
                for name in ('best.pt', 'last.pt'):
                    if not (stage_output / name).is_file():
                        raise FileNotFoundError(stage_output / name)
                state['completed'].append(stage)
                state['current'] = None
                write_json(queue, state)
        except BaseException:
            state['status'] = 'interrupted'
            write_json(queue, state)
            raise
        state['status'] = 'completed'
        write_json(queue, state)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
