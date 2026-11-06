"""Pinned point budgets, independent folders, and the 15-epoch screening queue."""
BASELINE_SHA256 = 'c602029630079252fcd249078877ebaae1be083c99e829a82c8e15aafd01887e'
FOLDERS = {name: name for name in (
    'control', 'pointsp_wrs', 'pointsp_ffps', 'dy_density', 'dy_curvature',
    'pointsp_lgp', 'upp_rectify', 'dwc_nmm', 'upp_completion', 'spo_keypoint', 'spo_projection')}
VARIANTS = {
    'control': ['b0', 'b_extra'],
    'pointsp_wrs': ['pointsp_wrs', 'uniform_random', 'b0', 'b_extra'],
    'pointsp_ffps': ['pointsp_ffps', 'b0', 'b_extra'],
    'dy_density': ['dy_density_low', 'dy_density_high', 'b0', 'b_extra'],
    'dy_curvature': ['dy_curvature', 'b0', 'b_extra'],
    'pointsp_lgp': ['pointsp_lgp', 'b0', 'b_extra'],
    'upp_rectify': ['upp_rectify', 'b0', 'b_extra'],
    'dwc_nmm': ['dwc_nmm', 'dwc_no_contrastive', 'b0', 'b_extra'],
    'upp_completion': ['upp_completion', 'b0', 'b_extra'],
    'spo_keypoint': ['spo_keypoint', 'b0', 'b_extra'],
    'spo_projection': ['spo_projection', 'b0', 'b_extra'],
}
FROZEN = [('uniform_random', 'pointsp_wrs', 'uniform_random'),
          ('pointsp_wrs', 'pointsp_wrs', 'pointsp_wrs'),
          ('pointsp_ffps', 'pointsp_ffps', 'pointsp_ffps'),
          ('dy_density_low', 'dy_density', 'dy_density_low'),
          ('dy_density_high', 'dy_density', 'dy_density_high'),
          ('dy_curvature', 'dy_curvature', 'dy_curvature'),
          ('pointsp_lgp', 'pointsp_lgp', 'pointsp_lgp')]
TRAINED = [('b_extra', 'control', 'b_extra')] + FROZEN + [
    ('upp_rectify', 'upp_rectify', 'upp_rectify'),
    ('dwc_nmm', 'dwc_nmm', 'dwc_nmm'),
    ('dwc_no_contrastive', 'dwc_nmm', 'dwc_no_contrastive'),
    ('upp_completion', 'upp_completion', 'upp_completion'),
    ('spo_keypoint', 'spo_keypoint', 'spo_keypoint'),
    ('spo_projection', 'spo_projection', 'spo_projection'),
]
DEFAULT_TASKS = [('b0', 'control', 'b0', 'validate')] + [
    ('frozen_' + name, folder, variant, 'validate') for name, folder, variant in FROZEN
] + [(name, folder, variant, 'train') for name, folder, variant in TRAINED]
FIXED = {'schema': 'pufm-module2-config-v1', 'input_points': 256, 'output_points': 1024,
         'whole_input_points': 2048, 'whole_output_points': 8192, 'flow_steps': 5,
         'train_patches_per_object': 32, 'training_mode': 'mixed', 'backbone': 'pufm_w_attn',
         'baseline_sha256': BASELINE_SHA256, 'lr': 1e-5, 'seed': 21,
         'selection_metric': 'seven_condition_object_mean_cd_l1', 'external_pretraining': False}


def default_config(name):
    config = {**FIXED, 'experiment': name, 'variant': VARIANTS[name][0],
              'allowed_variants': VARIANTS[name], 'epochs': 15, 'batch_size': 8,
              'workers': 0, 'assignment_workers': 8, 'validation_every': 5,
              'neighbors': 20, 'branch_width': 64, 'filter_fraction': .05,
              'offset_spacing': 2., 'feature_scale': .1, 'contrastive_temperature': .1,
              'anchor_queries': 32, 'anchor_divisor': 8, 'anchor_offset': .5,
              'ranking_weight': .01, 'projection_neighbors': 5, 'auxiliary_weight': 0.}
    config['auxiliary_weight'] = {'upp_rectify': 1., 'dwc_nmm': .001,
                                  'upp_completion': .1, 'spo_keypoint': .1,
                                  'spo_projection': .1}.get(name, 0.)
    return config


def validate_config(config):
    if any(config.get(k) != v for k, v in FIXED.items()):
        raise ValueError('Pinned baseline/point/data/solver/optimizer/selection protocol changed')
    owner = config['experiment']
    if owner not in VARIANTS or config['variant'] not in VARIANTS[owner] or config['allowed_variants'] != VARIANTS[owner]:
        raise ValueError('Variant belongs to another physical PUFM copy')
    if min(config['epochs'], config['batch_size'], config['assignment_workers'], config['validation_every']) < 1 or config['workers'] < 0:
        raise ValueError('Invalid training budget')
    if config['neighbors'] != 20 or config['branch_width'] != 64 or config['anchor_queries'] != 32 or config['anchor_divisor'] != 8:
        raise ValueError('Change candidate architecture in a separately registered protocol')


def comparison_control(task):
    return 'b0' if task['command'] == 'validate' else 'b_extra'
