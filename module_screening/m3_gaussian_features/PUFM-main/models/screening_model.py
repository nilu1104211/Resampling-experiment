"""One PUFM plus one local candidate. Inference accepts observed points only."""
from types import SimpleNamespace
import torch
from torch import nn
from .diffusion import PUFM, PUFM_w_attn
from .utils import midpoint_interpolate
from .adaptive_ops import gather, knn
from .candidate import build_candidate, candidate_loss


def validate_times(times, steps=5):
    times = [float(t) for t in times]
    if len(times) != steps + 1 or times[0] != 0.0 or times[-1] != 1.0:
        raise ValueError('Time grid must have six entries from exactly 0 to 1')
    if any(not torch.isfinite(torch.tensor(t)) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Time grid must be finite and strictly increasing')
    return times


class ScreeningPUFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if config['flow_steps'] != 5:
            raise ValueError('This screening protocol fixes the PUFM solver at five steps')
        cls = PUFM_w_attn if config['backbone'] == 'pufm_w_attn' else PUFM
        self.flow = cls(SimpleNamespace())
        self.candidate = build_candidate(config)
        self.times = validate_times([i / 5 for i in range(6)])
        self.freeze_flow = config['variant'].startswith(('m5', 'm6', 'm7')) or config['variant'] == 'b0'
        if self.freeze_flow:
            self.flow.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_flow:
            self.flow.eval()  # BatchNorm buffers are part of the frozen baseline too.
        return self

    def prepare(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 32 or not torch.isfinite(points).all():
            raise ValueError('Expected finite observed points [B,N,3], N>=32')
        observed = points
        diagnostics = {}
        if self.config['variant'].startswith('m5'):
            observed, diagnostics = self.candidate.denoise(points)
        initial = midpoint_interpolate(SimpleNamespace(up_rate=4), observed.transpose(1, 2).contiguous()).transpose(1, 2)
        features = None
        if self.candidate is not None and getattr(self.candidate, 'kind', None) == 'features':
            features = self.candidate(observed)
        return {'observed': observed, 'initial': initial, 'features': features, 'diagnostics': diagnostics}

    def velocity(self, query, t, state):
        residual = None
        if state['features'] is not None:
            ids = knn(state['observed'], query, 3)
            relative = query[:, :, None] - gather(state['observed'], ids)
            weight = relative.norm(dim=-1).clamp_min(1e-6).reciprocal()
            weight = weight / weight.sum(-1, keepdim=True)
            residual = (gather(state['features'], ids) * weight[..., None]).sum(2).transpose(1, 2)
        return self.flow(query.transpose(1, 2).contiguous(), state['initial'].transpose(1, 2).contiguous(), t,
                         feature_residual=residual).transpose(1, 2)

    def integrate(self, state):
        query = state['initial']
        for left, right in zip(self.times, self.times[1:]):
            t = query.new_full((len(query),), left)
            query = query + (right - left) * self.velocity(query, t, state)
        return query

    def forward(self, points, *, return_diagnostics=False):
        state = self.prepare(points)
        prediction = self.integrate(state)
        diagnostics = state['diagnostics']
        if self.config['variant'].startswith('m6'):
            prediction, diagnostics = self.candidate.refine(prediction)
        if prediction.shape != (len(points), points.shape[1] * 4, 3) or not torch.isfinite(prediction).all():
            raise FloatingPointError('Invalid fixed-budget prediction')
        return (prediction, diagnostics) if return_diagnostics else prediction

    def training_loss(self, batch):
        points, target = batch['input_xyz'], batch['gt_xyz']
        variant = self.config['variant']
        if variant.startswith('m5'):
            return self.candidate.training_loss(points, batch['supervision']['clean_input_xyz'])
        if variant.startswith('m6'):
            with torch.no_grad():
                prediction = self.integrate(self.prepare(points))
            return self.candidate.training_loss(prediction.detach(), target)
        state = self.prepare(points)
        if variant.startswith(('m1', 'm2')):
            initial = state['initial']
            if variant == 'm1b':
                initial = initial + self.config['perturb_sigma'] * torch.randn_like(initial)
            prediction = initial + self.velocity(initial, initial.new_zeros(len(initial)), state)
            return candidate_loss(prediction, target, self.config)
        # Exactly the existing V3 baseline FM target and sampling distribution.
        from adaptive_resampling.v3_transport import assignment_indices
        initial = state['initial'] + .01 * torch.randn_like(state['initial'])
        matched = gather(target, assignment_indices(initial, target))
        t = 1 - torch.cos(torch.rand(len(initial), device=initial.device) * torch.pi / 2)
        query = (1 - t[:, None, None]) * initial + t[:, None, None] * matched
        return (self.velocity(query, t, state) - (matched - initial)).square().mean()


def load_baseline(model, saved):
    """Strict flow loading; reject another experiment's or an adaptive checkpoint."""
    if saved.get('schema') == 'pufm-module-checkpoint-v1':
        raise ValueError('Initialize from original PUFM, not another screening checkpoint')
    cfg = saved.get('configuration', {})
    if cfg and (cfg.get('use_rgr', False) or cfg.get('condition', False) or cfg.get('allocation', 'midpoint') != 'midpoint'):
        raise ValueError('The common initial checkpoint must be the original midpoint PUFM baseline')
    if cfg.get('backbone', model.config['backbone']) != model.config['backbone']:
        raise ValueError('Baseline backbone mismatch')
    if saved.get('partial_debug_run'):
        raise ValueError('A debug checkpoint cannot initialize a formal screening run')
    state = saved.get('model', saved.get('state_dict', saved))
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    if any(k.startswith('flow.') for k in state):
        state = {k[5:]: v for k, v in state.items() if k.startswith('flow.')}
    model.flow.load_state_dict(state, strict=True)
