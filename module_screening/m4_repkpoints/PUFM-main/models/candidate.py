"""M4a: upstream RepKPU REM only, projected into PUFM's final 128-D feature.

No KP-Queries or displacement generator. See repk_rem.py and THIRD_PARTY.md.
"""
from types import SimpleNamespace
import torch
from torch import nn
from .repk_rem import REM


class RepKBranch(nn.Module):
    kind = 'features'

    def __init__(self, config):
        super().__init__()
        width = config['branch_width']
        self.embed = nn.Conv1d(3, width, 1)
        self.rem = REM(SimpleNamespace(conv_radius=.8, kernel_radius=.3,
            kernel_point_receptive_radius=.2, neighbor_limits=30, num_kernel_points=15,
            kp_dim=width, in_dim=width, trans_dim=width, rigid_scale=1., is_kp_bias=False, is_kp_bn=False))
        self.project = nn.Conv1d(width, 128, 1)
        nn.init.normal_(self.project.weight, std=.001)
        nn.init.zeros_(self.project.bias)

    def forward(self, points):
        pos = points.transpose(1, 2).contiguous()
        _, representation, _ = self.rem(pos, self.embed(pos))
        return self.project(representation[1].mean(-1)).transpose(1, 2)


class MLPControl(nn.Module):
    kind = 'features'

    def __init__(self, config):
        super().__init__()
        # Match the trainable REM branch capacity without kernel/neighbor inputs.
        reference = RepKBranch(config)
        budget = sum(p.numel() for p in reference.parameters() if p.requires_grad)
        width = max(1, round((budget - 128) / 132))
        self.net = nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, 128))
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, points):
        return self.net(points)


def build_candidate(config):
    if config['variant'] == 'm4a':
        return RepKBranch(config)
    if config['variant'] == 'mlp_control':
        return MLPControl(config)
    return None


def candidate_loss(prediction, target, config):
    raise RuntimeError('M4 uses the original FM loss; REM regularization is not added')
