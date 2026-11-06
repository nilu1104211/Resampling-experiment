"""Validate paired identities and summarize the queue's selected checkpoints."""
from pathlib import Path
import csv
import json


def collect(output):
    output = Path(output)
    state = json.loads((output / 'queue.json').read_text(encoding='utf-8'))
    summaries = {}
    for task in state['plan']:
        if task['command'] == 'check':
            continue
        root = Path(task['output'])
        if task['command'] == 'train':
            best = json.loads((root / 'best.json').read_text(encoding='utf-8'))
            root = root / f'validation/epoch_{best["epoch"]:04d}'
        summaries[task['name']] = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if not summaries:
        return
    b0 = summaries['b0']
    rows = []
    for task in state['plan']:
        name = task['name']
        if name not in summaries:
            continue
        summary = summaries[name]
        for key in ('input_selection_sha256', 'metadata_sha256', 'runtime', 'baseline_sha256', 'split'):
            if summary[key] != b0[key]:
                raise ValueError(f'{name}: unfair input/environment/protocol mismatch: {key}')
        reference = 'b0' if task['command'] == 'validate' or name == 'b_extra' else 'b_extra'
        control = summaries[reference]
        secondary = {'pointsp_wrs': 'uniform_random', 'frozen_pointsp_wrs': 'frozen_uniform_random',
                     'dwc_nmm': 'dwc_no_contrastive'}.get(name)
        secondary = secondary if secondary in summaries else None
        row = {'experiment': name, 'control': reference, 'cd_l1': summary['macro']['cd_l1'],
               'f_score': summary['macro']['f_score'], 'hausdorff_l2': summary['macro']['hausdorff_l2'],
               'cd_l1_gain_percent': (1 - summary['macro']['cd_l1'] / control['macro']['cd_l1']) * 100,
               'f_score_delta_percentage_points': (summary['macro']['f_score'] - control['macro']['f_score']) * 100,
               'parameters': summary['parameter_count'], 'extra_parameters': summary['extra_parameter_count'],
               'prediction_seconds': summary['macro']['prediction_seconds'],
               'secondary_control': secondary,
               'secondary_cd_l1_gain_percent': (1 - summary['macro']['cd_l1'] / summaries[secondary]['macro']['cd_l1']) * 100
                   if secondary else None,
               'partial_debug_run': task['expected_debug'] or summary['partial_debug_run']}
        rows.append(row)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
