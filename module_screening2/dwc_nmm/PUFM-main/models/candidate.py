"""DWCNet NMM adaptation: clean attention/FFN and geometric noisy neighborhoods.

Noisy convolutions operate on distance-ordered kNN windows, not input point order.
Clean teacher features use the same indexed clean observations, train time only.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .adaptive_ops import knn, gather
from .adapter_ops import PointEncoder


class Candidate(nn.Module):
    kind = 'features'

    def __init__(self, config):
        super().__init__()
        self.config = config
        width = config['branch_width']
        self.encoder = PointEncoder(width, config['neighbors'])
        self.clean_attn = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.clean_ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.projection = nn.Linear(width, 128)
        nn.init.normal_(self.projection.weight, std=.001)
        nn.init.zeros_(self.projection.bias)
        self.contrastive = config['variant'] != 'dwc_no_contrastive'
        if self.contrastive:
            self.noisy_convs = nn.ModuleList([nn.Conv1d(width, width, k) for k in (1, 3, 5)])
            self.noisy_merge = nn.Linear(3 * width, width)

    def encode(self, points):
        feature = self.encoder(points)
        clean = self.norm1(feature + self.clean_attn(feature, feature, feature, need_weights=False)[0])
        clean = self.norm2(clean + self.clean_ffn(clean))
        return self.projection(clean) * self.config['feature_scale'], {'clean': clean, 'encoded': feature}

    def auxiliary(self, batch, state):
        if not self.contrastive:
            return state['features'].sum() * 0
        clean = state['feature_cache']['clean']
        encoded = state['feature_cache']['encoded']
        ids = knn(batch['input_xyz'], batch['input_xyz'], 5)
        local = gather(encoded, ids)
        b, n, _, width = local.shape
        noisy = []
        for conv, k in zip(self.noisy_convs, (1, 3, 5)):
            window = local[:, :, :k].reshape(b * n, k, width).transpose(1, 2)
            noisy.append(conv(window).squeeze(-1).reshape(b, n, width))
        noisy = self.noisy_merge(torch.cat(noisy, -1))
        # The clean_input_xyz indices correspond exactly to the observed indices.
        with torch.no_grad():
            teacher = self.encoder(batch['supervision']['clean_input_xyz'])
        clean, teacher, noisy = [F.normalize(x, dim=-1, eps=1e-6) for x in (clean, teacher, noisy)]
        positive = -(clean * teacher).sum(-1).mean()
        negative = clean @ noisy.transpose(-1, -2) / self.config['contrastive_temperature']
        mask = torch.eye(n, device=negative.device, dtype=torch.bool)
        negative = negative.masked_fill(mask, -torch.inf).logsumexp(-1).mean()
        return positive + negative


def build_candidate(config):
    return None if config['variant'] in {'b0', 'b_extra'} else Candidate(config)
