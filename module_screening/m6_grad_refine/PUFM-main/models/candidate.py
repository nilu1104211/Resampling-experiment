"""M6: Grad-PU P2PNet distance model, on frozen PUFM predictions only.

Train with training-only GT distances. Refinement sees prediction coordinates
only and computes gradients of the learned distance, never a test GT loss.
"""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .grad_p2p import P2PNet
from .adaptive_ops import knn, gather


class GradientRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.distance = P2PNet(SimpleNamespace(k=16, feat_dim=32, block_num=3, layer_num=3, growth_rate=32, bn_size=1))

    def training_loss(self, prediction, target):
        # Original predictions + nearby queries. Context remains fixed predictions.
        query = prediction + self.config['distance_jitter'] * torch.randn_like(prediction)
        with torch.no_grad():
            labels = torch.cdist(query, target).min(-1).values
        estimated = self.distance(prediction.transpose(1, 2).contiguous(), query.transpose(1, 2).contiguous()).squeeze(1)
        return F.smooth_l1_loss(estimated, labels, beta=.01)

    def refine(self, prediction):
        context = prediction.detach().transpose(1, 2).contiguous()
        with torch.no_grad():
            global_features, local_features = self.distance.extract_feature(context)
            neighbors = gather(prediction, knn(prediction, prediction, min(8, prediction.shape[1])))
            spacing = (neighbors - prediction[:, :, None]).norm(dim=-1)[..., 1:].mean(-1, keepdim=True).clamp_min(1e-5)
        current = prediction.detach()
        steps = getattr(self, 'inference_steps', 3 if self.config['variant'] == 'm6b' else self.config['refine_steps'])
        for _ in range(steps):
            with torch.enable_grad():
                query = current.detach().requires_grad_(True)
                distance = self.distance.regress_distance(context, query.transpose(1, 2).contiguous(), global_features, local_features)
                gradient = torch.autograd.grad(distance.sum(), query, create_graph=False)[0]
            delta = -self.config['refine_step_size'] * gradient
            maximum = spacing * self.config['refine_max_spacing']
            delta = delta * (maximum / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
            current = (current + delta).detach()
        return current, {'refine_steps': steps, 'output_displacement_mean': (current - prediction).norm(dim=-1).mean(1),
                         'output_displacement_max': (current - prediction).norm(dim=-1).amax(1)}


def build_candidate(config):
    return GradientRefiner(config) if config['variant'] in {'m6a', 'm6b'} else None


def candidate_loss(prediction, target, config):
    raise RuntimeError('M6 trains only its learned distance model')
