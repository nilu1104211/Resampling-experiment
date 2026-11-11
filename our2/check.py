"""Bounded local verification with real V3 labels and pretrained PUFM weights."""
import argparse
import json
from pathlib import Path
import sys
import time

LOCAL=Path(__file__).resolve().parent
sys.path.insert(0,str(LOCAL.parent))
sys.path.insert(0,str(LOCAL/'PUFM-main'))
from our2.config import DEFAULT
from our2.run import configure_runtime,seed_all,file_hash,write_json,read_checkpoint,save_checkpoint


def main():
    import torch
    from models.dual import DualPUFM,load_baseline
    from models.gdba import bounded_soft_budget,integer_budget
    from models.utils import midpoint_interpolate
    from types import SimpleNamespace
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--fps-backend',choices=['cuda','torch'],default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--whole-shape',action='store_true')
    a=p.parse_args()
    if a.output.resolve().is_relative_to(LOCAL): p.error('Outputs must stay outside our2')
    if a.output.exists(): raise FileExistsError(a.output)
    configure_runtime(a);seed_all(21)
    if file_hash(a.baseline_checkpoint)!=DEFAULT['baseline_sha256']: raise ValueError('Baseline SHA mismatch')
    saved=read_checkpoint(a.baseline_checkpoint)
    ds=AdaptiveResamplingDataset(a.root,'validation',condition='C2_hole_10',include_supervision=True)
    sample=ds[0];ds.close()
    batch={k:torch.as_tensor(sample[k],device=a.device)[None] for k in ('input_xyz','gt_xyz')}
    batch['supervision']={k:torch.as_tensor(v,device=a.device).float()[None] for k,v in sample['supervision'].items()}
    report=dict(baseline_sha256=DEFAULT['baseline_sha256'],sample_id=sample['sample_id'],
                supervision_fields=sorted(batch['supervision']),torch=torch.__version__,
                gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None,checks={})
    started=time.perf_counter()
    # Extreme logits, infeasible capacities and fractional budgets.
    for n,k in [(1024,8),(8192,64),(1003,9)]:
        logits=(torch.randn(2,k,device=a.device)*100).requires_grad_()
        lower=torch.full_like(logits,int(.5*n/k));upper=torch.full_like(logits,int(2*n/k))
        soft=bounded_soft_budget(logits,n,lower,upper)
        hard=integer_budget(soft,n,lower,upper)
        assert (hard.sum(-1)==n).all() and (hard>=lower).all() and (hard<=upper).all()
        torch.testing.assert_close(soft.sum(-1),soft.new_full((2,),n),atol=.01,rtol=0)
    report['checks']['bounded_integer_budget']=True
    baseline=DualPUFM(DEFAULT).to(a.device).eval();load_baseline(baseline,saved)
    x=batch['input_xyz']
    with torch.no_grad():
        initial=midpoint_interpolate(SimpleNamespace(up_rate=4),x.transpose(1,2).contiguous())
        q=initial.clone()
        for i in range(5): q=q+baseline.flow(q,initial,q.new_full((1,),i/5))/5
        y=baseline(x)
        torch.testing.assert_close(y,q.transpose(1,2),atol=0,rtol=0)
    report['checks']['baseline_exact']=True
    stages=['m1_rel','m1_hole','m1_refine','m2_soft','m2_hard','joint_condition','joint_finetune']
    for stage in stages:
        joint=stage.startswith('joint')
        config={**DEFAULT,'stage':stage,'enable_m1':stage.startswith('m1') or joint,'enable_m2':stage.startswith('m2') or joint}
        model=DualPUFM(config).to(a.device).eval();load_baseline(model,saved)
        with torch.no_grad():
            state=model.prepare(x)
            if config['enable_m1'] and not config['enable_m2']:
                torch.testing.assert_close(state['initial'],initial.transpose(1,2),atol=0,rtol=0)
            if 'budget' in state:
                hard=state['budget']['hard']
                assert hard.sum()==1024
                assert len(torch.unique(state['initial'][0],dim=0))==1024
        model.train()
        loss=model.training_loss(batch);loss.backward()
        grads={}
        for name,module in [('flow',model.flow),('m1',model.m1),('m2',model.m2)]:
            if module is None: continue
            active=[p for p in module.parameters() if p.requires_grad]
            g=[p.grad for p in active if p.grad is not None]
            assert all(torch.isfinite(v).all() for v in g)
            grads[name]=sum(float(v.abs().sum()) for v in g)
            if active: assert grads[name]>0,(stage,name)
        if model.m2 is not None:
            assert model.m2.independent[-1].weight.grad.abs().sum()>0
        if stage in ('m1_refine','joint_finetune'):
            assert model.m1.gamma.grad.abs()>0
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        optimizer.step();model.eval()
        with torch.no_grad():
            pred=model(x); repeated=model(x)
            torch.testing.assert_close(pred,repeated,atol=0,rtol=0)
        report['checks'][stage]=dict(loss=float(loss.detach()),gradients=grads,shape=list(pred.shape))
        if stage=='joint_finetune':
            a.output.mkdir(parents=True)
            weight=a.output/'verification_joint.pt'
            save_checkpoint(weight,dict(schema='our2-verification-only',partial_debug_run=True,config=config,model=model.state_dict(),baseline_sha256=DEFAULT['baseline_sha256']))
            loaded=DualPUFM(config).to(a.device).eval()
            loaded.load_state_dict(read_checkpoint(weight)['model'],strict=True)
            with torch.no_grad(): torch.testing.assert_close(pred,loaded(x),atol=0,rtol=0)
            report['checks']['checkpoint_reload_exact']=True
        if a.whole_shape and stage in ('m1_refine','m2_hard','joint_finetune'):
            # Interface scaling check only; synthetic cloud is not a quality benchmark.
            whole=torch.randn(1,2048,3,device=a.device)*.2
            with torch.no_grad(): result=model(whole)
            assert result.shape==(1,8192,3)
            report['checks'][stage]['whole_shape']=list(result.shape)
        del model,loss,pred,optimizer
    report.update(status='passed',seconds=time.perf_counter()-started,quality_claim=False)
    write_json(a.output/'verification.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
