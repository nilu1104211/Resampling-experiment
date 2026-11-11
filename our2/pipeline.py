"""Sequential stage execution; only best/last per stage, safe resumable subprocesses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

LOCAL=Path(__file__).resolve().parent
sys.path.insert(0,str(LOCAL.parent))
from our2.config import DEFAULT
from our2.run import file_hash,write_json,code_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--fps-backend',default='cuda',choices=['cuda','torch'])
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--seed',type=int,default=21)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit-batches',type=int,default=0)
    p.add_argument('--limit-samples',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-debug',action='store_true')
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or min(a.limit_batches,a.limit_samples)<0: p.error('Invalid budget')
    if a.output.resolve().is_relative_to(LOCAL): p.error('Keep outputs outside our2')
    if file_hash(a.baseline_checkpoint)!=DEFAULT['baseline_sha256']: raise ValueError('Baseline SHA mismatch')
    # Equal sample/update budgets: M1=3E, M2=2E, Joint includes both parents +2E=7E.
    stages=[('control_m1','baseline',3),('control_m2','baseline',2),('control_joint','baseline',7),
            ('m1_rel','m1_rel',1),('m1_hole','m1_hole',1),('m1_refine','m1_refine',1),
            ('m2_soft','m2_soft',1),('m2_hard','m2_hard',1),
            ('joint_condition','joint_condition',1),('joint_finetune','joint_finetune',1)]
    parent={'m1_hole':'m1_rel','m1_refine':'m1_hole','m2_hard':'m2_soft','joint_finetune':'joint_condition'}
    offset={'m1_rel':0,'m1_hole':1,'m1_refine':2,'m2_soft':0,'m2_hard':1,'joint_condition':5,'joint_finetune':6}
    identity={'args':{k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'},
              'sources':code_identity(),'baseline':file_hash(a.baseline_checkpoint)}
    queue=a.output/'queue.json'
    if a.resume:
        state=json.loads(queue.read_text())
        if state['identity']!=identity: raise ValueError('Queue configuration/source changed')
    else:
        if a.output.exists(): raise FileExistsError(a.output)
        a.output.mkdir(parents=True)
        state={'identity':identity,'status':'running','completed':[]}
        write_json(queue,state)
    # Never steal stale locks automatically; PID and host are recorded for inspection.
    lock=a.output/'queue.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f: json.dump({'pid':os.getpid(),'host':__import__('socket').gethostname()},f)
    try:
        for name,stage,multiple in stages:
            if name in state['completed']: continue
            config={**DEFAULT,**json.loads((LOCAL/f'configs/{stage}.json').read_text()),
                    'epochs':a.epochs*multiple,'seed':a.seed,'batch_size':a.batch_size,
                    'data_epoch_offset':30+offset.get(stage,0)*a.epochs}
            path=a.output/'configs'/f'{name}.json';write_json(path,config)
            run=a.output/name
            command=[sys.executable,'-B',str(LOCAL/'run.py'),'train','--config',str(path),'--root',str(a.root),
                     '--baseline-checkpoint',str(a.baseline_checkpoint),'--output',str(run),
                     '--device',a.device,'--fps-backend',a.fps_backend,'--limit-batches',str(a.limit_batches),
                     '--limit-samples',str(a.limit_samples)]
            if a.allow_debug: command+=['--allow-debug']
            if stage in parent: command+=['--init',str(a.output/parent[stage]/'best.pt')]
            if stage=='joint_condition':
                command+=['--m1-checkpoint',str(a.output/'m1_refine/best.pt'),'--m2-checkpoint',str(a.output/'m2_hard/best.pt')]
            if a.resume and (run/'last.pt').exists(): command+=['--resume',str(run/'last.pt')]
            process=subprocess.Popen(command,cwd=LOCAL.parent)
            try:
                if process.wait()!=0: raise RuntimeError(f'Stage failed: {name}')
            finally:
                if process.poll() is None:
                    process.terminate();process.wait()
            import torch
            saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            if not saved['training_complete']: raise RuntimeError('Stage is incomplete')
            state['completed'].append(name);write_json(queue,state)
        state['status']='completed';write_json(queue,state)
    except BaseException:
        state['status']='interrupted';write_json(queue,state)
        raise
    finally: lock.unlink()


if __name__=='__main__': main()
