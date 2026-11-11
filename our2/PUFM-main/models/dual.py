"""PUFM with two independent observation-only modules; original five-step flow."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .diffusion import PUFM_w_attn
from .utils import midpoint_interpolate
from .adaptive_ops import gather, knn, distance_vectors
from .geometry import geometry
from .ragr import RAGR
from .gdba import GDBA


class DualPUFM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.flow = PUFM_w_attn(SimpleNamespace())
        self.m1 = RAGR(config['width'], config['max_move']) if config['enable_m1'] else None
        self.m2 = GDBA(config['width'], config['minimum'], config['maximum'], config['temperature']) if config['enable_m2'] else None
        self.stage = config['stage']
        self.loss_diagnostics = {}
        self.set_stage(self.stage)

    def set_stage(self, stage):
        self.stage = stage
        for p in self.parameters(): p.requires_grad_(False)
        if stage in ('baseline', 'm1_refine', 'm2_hard', 'joint_finetune'):
            if stage == 'baseline' or self.config['train_flow']:
                for p in self.flow.parameters(): p.requires_grad_(True)
        if self.m1 is not None and stage.startswith('m1') or self.m1 is not None and stage == 'joint_finetune':
            for p in self.m1.parameters(): p.requires_grad_(True)
            if stage in ('m1_rel', 'm1_hole'):
                self.m1.gamma.requires_grad_(False)
                for p in self.m1.direction.parameters(): p.requires_grad_(False)
        if self.m2 is not None and stage in ('m2_soft', 'm2_hard', 'joint_condition', 'joint_finetune'):
            for p in self.m2.parameters(): p.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BN statistics must stay frozen, including the first joint stage.
        if not any(p.requires_grad for p in self.flow.parameters()): self.flow.eval()
        if self.m1 is not None and not any(p.requires_grad for p in self.m1.parameters()): self.m1.eval()
        return self

    def prepare(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 64 or not torch.isfinite(points).all():
            raise ValueError('Expected finite observation-only B,N,3, N>=64')
        initial = midpoint_interpolate(SimpleNamespace(up_rate=4), points.transpose(1,2).contiguous()).transpose(1,2)
        state = dict(initial=initial, original=initial, diagnostics={})
        if self.m1 is None and self.m2 is None: return state
        geo = geometry(points, self.config['region_divisor'])
        state['geo'] = geo
        m1 = self.m1(geo) if self.m1 is not None else None
        state['m1'] = m1
        if self.m2 is not None:
            logits = self.m2(geo, m1, self.config['joint_condition'], self.config['budget_policy'])
            state['logits'] = logits
            if self.stage != 'm2_soft':
                initial, budget = self.m2.allocate(initial, geo, logits, self.config['allocation_strength'])
                state['budget'] = budget
                fraction = budget['hard'].float() / initial.shape[1]
                state['diagnostics'].update(budget=budget['hard'], budget_min=budget['hard'].min(),
                    budget_max=budget['hard'].max(), budget_entropy=-(fraction*fraction.clamp_min(1e-8).log()).sum(-1).mean(),
                    soft_hard_gap=(budget['soft']-budget['hard']).abs().mean())
        if m1 is not None:
            if self.stage not in ('m1_rel','m1_hole'):
                initial, movement = self.m1.refine(initial, geo, m1, self.config['use_reliability'], self.config['use_hole'])
                state['movement'] = movement
                state['diagnostics']['movement_mean'] = movement.norm(dim=-1).mean().detach()
            state['diagnostics'].update(reliability_mean=m1['r'].mean().detach(), hole_mean=m1['h'].mean().detach())
        state['initial'] = initial
        return state

    def velocity(self, query, t, state):
        return self.flow(query.transpose(1,2).contiguous(), state['initial'].transpose(1,2).contiguous(), t).transpose(1,2)

    def integrate(self, state):
        query = state['initial']
        for i in range(5):
            query = query + self.velocity(query, query.new_full((len(query),), i/5), state) / 5
        return query

    def forward(self, points, *, return_diagnostics=False):
        state = self.prepare(points)
        prediction = self.integrate(state)
        if prediction.shape != (len(points), 4*points.shape[1], 3) or not torch.isfinite(prediction).all():
            raise FloatingPointError('Invalid fixed-budget output')
        return (prediction,state['diagnostics']) if return_diagnostics else prediction

    def training_loss(self, batch):
        from adaptive_resampling.v3_transport import assignment_indices
        state = self.prepare(batch['input_xyz'])
        target = batch['gt_xyz']
        losses = {}
        warmup = self.stage in ('m1_rel','m1_hole','m2_soft','joint_condition')
        if not warmup:
            initial = state['initial'] + .01*torch.randn_like(state['initial'])
            matched = gather(target, assignment_indices(initial, target))
            t = 1-torch.cos(torch.rand(len(initial),device=initial.device)*torch.pi/2)
            query = (1-t[:,None,None])*initial+t[:,None,None]*matched
            velocity = self.velocity(query,t,state)
            losses['fm'] = (velocity-(matched-initial)).square().mean()
            endpoint = query+(1-t[:,None,None])*velocity
        if self.m1 is not None and any(p.requires_grad for p in self.m1.parameters()):
            labels = batch['supervision']
            m1, geo = state['m1'],state['geo']
            if self.config['use_reliability']:
                reliability = labels['reliability_target_input']
                weight = 1+4*(1-reliability)
                losses['rel'] = .01*(F.binary_cross_entropy_with_logits(m1['rlogit'],reliability,reduction='none')*weight).mean()
            if self.stage != 'm1_rel' and self.config['use_hole']:
                # A candidate is positive only near real deleted surface, never just empty space.
                with torch.no_grad():
                    nearest = knn(target,geo['candidates'],1).squeeze(-1)
                    near = (gather(target,nearest)-geo['candidates']).norm(dim=-1)
                    hole = gather(labels['hole_mask_gt'][...,None],nearest).squeeze(-1)
                    htarget = hole*torch.exp(-near.square()/geo['region_scale'].squeeze(-1).square().clamp_min(1e-8))
                losses['hole'] = .01*(F.binary_cross_entropy_with_logits(m1['hlogit'],htarget,reduction='none')*(1+4*htarget)).mean()
            if not warmup and self.config['use_hole']:
                a,b = distance_vectors(endpoint,target)
                mask = labels['hole_mask_gt'].float()
                local = (b*mask).sum()/mask.sum().clamp_min(1)
                losses['recovery'] = self.config['recovery_weight']*(local+a.mean())
            if 'movement' in state:
                losses['move'] = .01*state['movement'].square().mean()
        if self.m2 is not None and any(p.requires_grad for p in self.m2.parameters()):
            with torch.no_grad():
                geo = state['geo']
                ids = knn(geo['anchors'],target,1).squeeze(-1)
                nearest = knn(state['original'],target,1).squeeze(-1)
                error = (gather(state['original'],nearest)-target).norm(dim=-1)
                scale = gather(geo['region_scale'],ids).squeeze(-1)
                demand = 1+(error/scale.clamp_min(1e-6)).clamp_max(3)
                membership = F.one_hot(ids,geo['anchors'].shape[1]).to(target.dtype)
                demand = (membership*demand[...,None]).sum(1)+1e-4
                demand = demand/demand.sum(-1,keepdim=True)
            # Explicit proxy gradient; hard budget itself is not differentiated.
            losses['budget'] = .01*F.kl_div(F.log_softmax(state['logits']/self.config['temperature'],-1),demand,reduction='batchmean')
        if not losses: raise ValueError('Stage has no active training objective')
        self.loss_diagnostics = {k:float(v.detach()) for k,v in losses.items()}
        return sum(losses.values())


def load_baseline(model,saved):
    cfg = saved.get('configuration',{})
    if saved.get('partial_debug_run') or saved.get('schema','').startswith(('pufm-module','our2')):
        raise ValueError('Original non-debug PUFM checkpoint required')
    if cfg.get('use_rgr',False) or cfg.get('condition',False) or cfg.get('allocation','midpoint')!='midpoint':
        raise ValueError('Adaptive checkpoint cannot initialize PUFM baseline')
    state = saved.get('model',saved.get('state_dict',saved))
    if all(k.startswith('module.') for k in state): state = {k[7:]:v for k,v in state.items()}
    if any(k.startswith('flow.') for k in state): state = {k[5:]:v for k,v in state.items() if k.startswith('flow.')}
    model.flow.load_state_dict(state,strict=True)
