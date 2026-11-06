"""UPP-inspired coordinate rectification, adapted to XYZ-only PUFM inputs.

This is a lightweight mechanism adaptation, not the pretrained Point-MAE model.
"""
import torch
from torch import nn
from .adapter_ops import PointEncoder, geometry


class Candidate(nn.Module):
    kind = 'input'

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config['branch_width']
        self.encoder = PointEncoder(width, config['neighbors'])
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def refine(self, points):
        with torch.no_grad():
            spacing = geometry(points, self.config['neighbors'])[-1]
        offset = self.head(self.encoder(points)).tanh() * spacing[..., None] * self.config['offset_spacing']
        return points + offset, {'input_offset_mean': offset.norm(dim=-1).mean().detach(),
                                 'input_offset_max': offset.norm(dim=-1).max().detach()}

    def auxiliary(self, batch, state):
        clean = batch['supervision']['clean_input_xyz']
        return (state['observed'] - clean).square().mean()


def build_candidate(config):
    return None if config['variant'] in {'b0', 'b_extra'} else Candidate(config)
