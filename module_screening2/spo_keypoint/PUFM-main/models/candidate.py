"""SPOFormer-inspired coarse keypoints and supervised candidate ranking.

The original argsort-only ranker is replaced by explicit training-only score
supervision. This is a lightweight mechanism adaptation, not full SPOFormer.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .adaptive_ops import gather, fps_ids
from .adapter_ops import PointEncoder, grid, anchor_loss, median_value


class Candidate(nn.Module):
    kind = 'anchors'

    def __init__(self, config):
        super().__init__()
        self.config = config
        w, q = config['branch_width'], config['anchor_queries']
        self.encoder = PointEncoder(w, config['neighbors'])
        self.coarse_pred = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, 2 * q * 3))
        self.query_ranking = nn.Sequential(nn.Linear(w + 3, w), nn.GELU(), nn.Linear(w, 1))
        self.dense_pred = nn.Sequential(nn.Linear(w + 5, w), nn.GELU(), nn.Linear(w, 3))
        nn.init.normal_(self.coarse_pred[-1].weight, std=.001)
        nn.init.zeros_(self.coarse_pred[-1].bias)
        nn.init.normal_(self.dense_pred[-1].weight, std=.001)
        nn.init.zeros_(self.dense_pred[-1].bias)

    def initialize(self, points, original):
        q = self.config['anchor_queries']
        features = self.encoder(points)
        global_feature = features.max(1).values
        base = gather(points, fps_ids(points, 2 * q))
        coarse = base + self.coarse_pred(global_feature).reshape(len(points), 2 * q, 3).tanh() * self.config['anchor_offset']
        expanded = global_feature[:, None].expand(-1, 2 * q, -1)
        logits = self.query_ranking(torch.cat([expanded, coarse], -1)).squeeze(-1)
        ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :q]
        selected = gather(coarse, ids)
        budget = original.shape[1] // self.config['anchor_divisor']
        per = math.ceil(budget / q)
        plane = grid(per, points.device, points.dtype)[None, None].expand(len(points), q, -1, -1)
        context = global_feature[:, None, None].expand(-1, q, per, -1)
        xyz = selected[:, :, None].expand(-1, -1, per, -1)
        offset = self.dense_pred(torch.cat([context, xyz, plane], -1)).tanh() * self.config['anchor_offset']
        anchors = (xyz + offset).flatten(1, 2)[:, :budget]
        prediction = torch.cat([original[:, :original.shape[1] - budget], anchors], 1)
        return prediction, {'anchors': anchors, 'all_coarse': coarse, 'rank_logits': logits,
                            'anchor_count': budget, 'anchors_retained': budget}

    def auxiliary(self, batch, state):
        cache = state['anchor_cache']
        coarse, logits = cache['all_coarse'], cache['rank_logits']
        with torch.no_grad():
            distance = torch.cdist(coarse, batch['gt_xyz']).min(-1).values
            threshold = median_value(distance, keepdim=True).clamp_min(1e-4)
            score = torch.exp(-distance / threshold).clamp(.01, .99)
        ranking = F.binary_cross_entropy_with_logits(logits, score)
        return anchor_loss(cache['anchors'], batch) + anchor_loss(coarse, batch) + self.config['ranking_weight'] * ranking


def build_candidate(config):
    return None if config['variant'] in {'b0', 'b_extra'} else Candidate(config)
