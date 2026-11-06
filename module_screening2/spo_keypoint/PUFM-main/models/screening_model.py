"""One physical PUFM copy and one observation-only candidate mechanism."""
from types import SimpleNamespace
import torch
from torch import nn
from .diffusion import PUFM_w_attn
from .utils import midpoint_interpolate
from .adaptive_ops import gather, knn
from .candidate import build_candidate


class ScreeningPUFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.flow = PUFM_w_attn(SimpleNamespace())
        self.candidate = build_candidate(config)
        self.times = [i / 5 for i in range(6)]
        self.loss_diagnostics = {}

    def prepare(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 64 or not torch.isfinite(points).all():
            raise ValueError('Expected finite observation-only B,N,3, N>=64')
        observed, diagnostics = points, {}
        kind = getattr(self.candidate, 'kind', None)
        if kind == 'input':
            observed, diagnostics = self.candidate.refine(points)
        initial = midpoint_interpolate(SimpleNamespace(up_rate=4), observed.transpose(1, 2).contiguous()).transpose(1, 2)
        state = {'observed': observed, 'initial': initial, 'features': None,
                 'diagnostics': diagnostics}
        if kind in {'initialization', 'anchors'}:
            state['initial'], cache = self.candidate.initialize(observed, initial)
            if kind == 'anchors':
                state['anchor_cache'] = cache
            state['diagnostics'].update({k: v for k, v in cache.items()
                                         if k not in {'anchors', 'all_coarse', 'rank_logits'}})
        if kind == 'features':
            state['features'], state['feature_cache'] = self.candidate.encode(observed)
            state['diagnostics']['feature_residual_mean'] = state['features'].norm(dim=-1).mean().detach()
        return state

    def velocity(self, query, t, state):
        residual = None
        if state['features'] is not None:
            ids = knn(state['observed'], query, 3)
            distance = (query[:, :, None] - gather(state['observed'], ids)).norm(dim=-1).clamp_min(1e-6)
            weight = distance.reciprocal()
            weight = weight / weight.sum(-1, keepdim=True)
            residual = (gather(state['features'], ids) * weight[..., None]).sum(2).transpose(1, 2)
        return self.flow(query.transpose(1, 2).contiguous(), state['initial'].transpose(1, 2).contiguous(),
                         t, feature_residual=residual).transpose(1, 2)

    def integrate(self, state):
        query = state['initial']
        for left, right in zip(self.times, self.times[1:]):
            query = query + (right - left) * self.velocity(query, query.new_full((len(query),), left), state)
        return query

    def forward(self, points, *, return_diagnostics=False):
        state = self.prepare(points)
        prediction = self.integrate(state)
        if prediction.shape != (len(points), 4 * points.shape[1], 3) or not torch.isfinite(prediction).all():
            raise FloatingPointError('Invalid fixed-budget output')
        return (prediction, state['diagnostics']) if return_diagnostics else prediction

    def training_loss(self, batch):
        from adaptive_resampling.v3_transport import assignment_indices
        points, target = batch['input_xyz'], batch['gt_xyz']
        state = self.prepare(points)
        initial = state['initial'] + .01 * torch.randn_like(state['initial'])
        matched = gather(target, assignment_indices(initial, target))
        t = 1 - torch.cos(torch.rand(len(initial), device=initial.device) * torch.pi / 2)
        query = (1 - t[:, None, None]) * initial + t[:, None, None] * matched
        velocity = self.velocity(query, t, state)
        fm = (velocity - (matched - initial)).square().mean()
        auxiliary = fm.new_zeros(())
        kind = getattr(self.candidate, 'kind', None)
        if kind in {'input', 'features', 'anchors'}:
            auxiliary = self.candidate.auxiliary(batch, state)
        elif kind == 'loss':
            # One-call straight-line endpoint estimate; no extra endpoint CD.
            endpoint = query + (1 - t[:, None, None]) * velocity
            auxiliary = self.candidate.endpoint_loss(endpoint)
        self.loss_diagnostics = {'fm_loss': float(fm.detach()), 'auxiliary_loss': float(auxiliary.detach())}
        return fm + self.config['auxiliary_weight'] * auxiliary


def load_baseline(model, saved):
    if saved.get('schema', '').startswith('pufm-module'):
        raise ValueError('Initialize from original PUFM, never another screening model')
    cfg = saved.get('configuration', {})
    if cfg.get('use_rgr', False) or cfg.get('condition', False) or cfg.get('allocation', 'midpoint') != 'midpoint':
        raise ValueError('The common checkpoint must be the original midpoint PUFM')
    if cfg.get('backbone', 'pufm_w_attn') != 'pufm_w_attn' or saved.get('partial_debug_run'):
        raise ValueError('Unexpected backbone/debug initial checkpoint')
    state = saved.get('model', saved.get('state_dict', saved))
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if any(k.startswith('flow.') for k in state):
        state = {k[5:]: v for k, v in state.items() if k.startswith('flow.')}
    model.flow.load_state_dict(state, strict=True)
