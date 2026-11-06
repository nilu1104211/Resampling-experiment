"""SPOFormer projection regularizer, batched and normalized over B,N,k,k.

Matches upstream neighbor-distance construction while replacing its sum reduction
with a mean. This changes the coefficient scale and is explicitly an adaptation.
"""
import torch
from torch import nn
from .adaptive_ops import gather, knn


def projection_loss(points, k=5):
    neighbors = gather(points, knn(points, points, k))
    center = neighbors.mean(2, keepdim=True)
    target = (center[:, :, :, None] - neighbors[:, :, None]).square().sum(-1)
    actual = (points[:, :, None, None] - neighbors[:, :, None]).square().sum(-1)
    # Broadcast k rows exactly as upstream, without explicit B,N,k,k XYZ tiles.
    actual = actual.expand(-1, -1, k, -1)
    target = target.expand_as(actual)
    weights = (actual.sum(-2, keepdim=True).square() + 1).reciprocal()
    return ((target - actual).abs() * weights).mean()


class Candidate(nn.Module):
    kind = 'loss'

    def __init__(self, config):
        super().__init__()
        self.k = config['projection_neighbors']

    def endpoint_loss(self, endpoint):
        return projection_loss(endpoint, self.k)


def build_candidate(config):
    return None if config['variant'] in {'b0', 'b_extra'} else Candidate(config)
