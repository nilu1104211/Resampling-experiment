"""UPP masked-query completion mechanism, adapted to fixed-budget PUFM anchors.

Uses no pretrained backbone, GT masks, labels, or normals during inference.
"""
import math
import torch
from torch import nn
from .adaptive_ops import gather, fps_ids
from .adapter_ops import PointEncoder, grid, anchor_loss


class Candidate(nn.Module):
    kind = 'anchors'

    def __init__(self, config):
        super().__init__()
        self.config = config
        w, q = config['branch_width'], config['anchor_queries']
        self.encoder = PointEncoder(w, config['neighbors'])
        self.mask_token = nn.Parameter(torch.zeros(1, q, w))
        self.predict_token_generator = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w))
        self.decoder = nn.MultiheadAttention(w, 4, batch_first=True)
        self.norm = nn.LayerNorm(w)
        self.coarse_pred = nn.Linear(w, 3)
        self.dense_pred = nn.Sequential(nn.Linear(w + 2, w), nn.GELU(), nn.Linear(w, 3))
        nn.init.normal_(self.coarse_pred.weight, std=.001)
        nn.init.zeros_(self.coarse_pred.bias)
        nn.init.normal_(self.dense_pred[-1].weight, std=.001)
        nn.init.zeros_(self.dense_pred[-1].bias)

    def initialize(self, points, original):
        q = self.config['anchor_queries']
        features = self.encoder(points)
        ids = fps_ids(points, q)
        center = gather(points, ids)
        token = self.predict_token_generator(gather(features, ids)) + self.mask_token
        token = self.norm(token + self.decoder(token, features, features, need_weights=False)[0])
        coarse = center + self.coarse_pred(token).tanh() * self.config['anchor_offset']
        budget = original.shape[1] // self.config['anchor_divisor']
        per = math.ceil(budget / q)
        plane = grid(per, points.device, points.dtype)[None, None].expand(len(points), q, -1, -1)
        dense = token[:, :, None].expand(-1, -1, per, -1)
        offset = self.dense_pred(torch.cat([dense, plane], -1)).tanh() * self.config['anchor_offset']
        anchors = (coarse[:, :, None] + offset).flatten(1, 2)[:, :budget]
        # Reserve anchor slots explicitly: no post-merge FPS can discard them.
        prediction = torch.cat([original[:, :original.shape[1] - budget], anchors], 1)
        return prediction, {'anchors': anchors, 'anchor_count': budget, 'anchors_retained': budget}

    def auxiliary(self, batch, state):
        return anchor_loss(state['anchor_cache']['anchors'], batch)


def build_candidate(config):
    return None if config['variant'] in {'b0', 'b_extra'} else Candidate(config)
