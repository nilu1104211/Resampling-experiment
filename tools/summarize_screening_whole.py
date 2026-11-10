"""Summarize completed whole-object tests alongside the earlier six networks."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import statistics
import os

REPO = Path(__file__).resolve().parents[1]
HISTORY = Path(os.environ.get('SCREENING_HISTORY_ROOT', REPO / 'server_runs')).resolve()
SUITES = ('whole_object_eval80_20261030', 'whole_object_eval80_spu_20261102',
          'whole_object_eval80_ours_20261102', 'whole_object_eval80_ours1_joint_b8_20261105',
          'whole_object_eval80_ounet_20261105')
METRICS = ('cd_l1', 'cd_l2_squared', 'hausdorff_l2', 'f_score')


def summarize(output, state):
    output = Path(output)
    rows, samples = [], {}
    for key, task in state['tasks'].items():
        if task['status'] != 'completed':
            raise ValueError(f'Cannot summarize incomplete task: {key}')
        root = Path(task['output'])
        summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
        provenance = json.loads((root / 'local_execution_manifest.json').read_text(encoding='utf-8'))
        if summary['samples'] != 560 or summary['partial_debug_run'] or not provenance['same_object_selection_as_original_six_models']:
            raise ValueError(f'Invalid formal sample selection: {key}')
        row = {'model': key.split('/')[0], 'split': key.split('/')[1], 'source': 'current_run',
               **{metric: summary['macro'][metric] for metric in METRICS}}
        rows.append(row)
        with (root / 'per_sample.csv').open(encoding='utf-8-sig', newline='') as handle:
            entries = list(csv.DictReader(handle))
        sample_ids = {(x['condition'], x['sample_id']) for x in entries}
        if len(entries) != 560 or len(sample_ids) != 560:
            raise ValueError(f'Invalid sample IDs: {key}')
        samples[key] = sample_ids
    for split in ('pu1k_object', 'modelnet40_object'):
        expected = samples['new_b0/' + split]
        for key, ids in samples.items():
            if key.endswith('/' + split) and ids != expected:
                raise ValueError(f'Sample identities differ from the local baseline: {key}')
    previous = defaultdict(list)
    for name in SUITES:
        with (HISTORY / name / 'subset_summary.csv').open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                previous[(row['model'], row['split'])].append(row)
    for (model, split), conditions in sorted(previous.items()):
        if len(conditions) != 7 or len({r['condition'] for r in conditions}) != 7:
            raise ValueError(f'Historical seven-condition summary incomplete: {model}/{split}')
        if any(int(r['objects']) != 80 or int(r['samples']) != 80 for r in conditions):
            raise ValueError(f'Historical object/sample budget changed: {model}/{split}')
        rows.append({'model': model, 'split': split, 'source': 'historical_saved',
                     **{metric: statistics.mean(float(r[metric]) for r in conditions) for metric in METRICS}})
    baselines = {r['split']: r for r in rows if r['model'] == 'new_b0'}
    by_model = {(row['model'], row['split'], row['source']): row for row in rows}
    for row in rows:
        baseline = baselines[row['split']]
        row['cd_gain_vs_current_pufm_percent'] = (1 - row['cd_l1'] / baseline['cd_l1']) * 100
        row['f_score_delta_vs_current_pufm_percentage_points'] = (row['f_score'] - baseline['f_score']) * 100
        if row['source'] == 'historical_saved':
            control_name = 'pufm'
        elif row['model'].startswith('old_'):
            control_name = ('old_m7_uniform_validation' if row['model'] in {
                'old_b_extra', 'old_m7_uniform_validation', 'old_m7_ats_validation'} else 'old_b_extra')
        elif row['model'] in {'new_b0', 'new_b_extra'} or row['model'].startswith('new_frozen_'):
            control_name = 'new_b0'
        else:
            control_name = 'new_b_extra'
        control = by_model[(control_name, row['split'], row['source'])]
        row['matched_control'] = control_name
        row['cd_gain_vs_matched_control_percent'] = (1 - row['cd_l1'] / control['cd_l1']) * 100
        row['f_score_delta_vs_matched_control_percentage_points'] = (row['f_score'] - control['f_score']) * 100
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'comparison.json').write_text(json.dumps({'scope': 'fixed 80 whole objects per dataset, seven conditions',
        'note': 'Historical and current rows have separately recorded runtime/backend settings; current PUFM is re-evaluated',
        'rows': rows}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
