"""Read-only audit of local V3 supervision and old continuation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

LOCAL=Path(__file__).resolve().parent
sys.path.insert(0,str(LOCAL.parent))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    import h5py
    import torch
    from adaptive_resampling.v3_dataset import AdaptiveResamplingDataset,SUPERVISION_KEYS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--server-runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(LOCAL): p.error('Keep audit output outside source')
    baseline=a.server_runs/'patch32/pufm_baseline_b8_w0_a8_fused/epoch_0030.pt'
    saved=torch.load(baseline,map_location='cpu',weights_only=False)
    result={'baseline_sha256':sha(baseline),'baseline_epoch':saved['epoch'],
            'baseline_training_profile':saved['training_profile'],
            'baseline_optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in saved['optimizer']['param_groups']],
            'data':{},'continuation':{}}
    manifest=json.loads((a.data_root/'metadata/base_manifest.json').read_text())
    for split in ('train','validation'):
        entry=manifest['groups'][split][0]
        path=a.data_root/entry['path'] if split=='train' else a.data_root/'processed'/split/Path(entry['path']).name
        with h5py.File(path) as f: result['data'][split]={'shard_keys':sorted(f.keys()),'complete':bool(f.attrs['complete'])}
        ds=AdaptiveResamplingDataset(a.data_root,split,include_supervision=True)
        sample=ds[0];ds.close()
        missing=set(SUPERVISION_KEYS)-set(sample['supervision'])
        if missing: raise ValueError(f'Missing supervision: {missing}')
        result['data'][split].update(sample_id=sample['sample_id'],input_shape=list(sample['input_xyz'].shape),
                                    gt_shape=list(sample['gt_xyz'].shape),supervision_keys=sorted(sample['supervision']))
    for name,relative in [('old10','module_screening/sequential10_v1'),('new15','module_screening2/sequential15_v1')]:
        base=a.server_runs/'patch32'/relative/'runs'
        path=base/'b_extra'
        if not path.exists(): continue
        config=json.loads((path/'config.json').read_text())
        history=[json.loads(l) for l in (path/'train.jsonl').read_text().splitlines()]
        b0path=base/'b0/summary.json'
        b0=json.loads(b0path.read_text())['macro'] if b0path.exists() else None
        result['continuation'][name]={'config':config,'best':json.loads((path/'best.json').read_text()),
           'b0_validation':b0,'history':[{k:v for k,v in row.items() if k in ('epoch','loss','seen_patches','optimizer_steps','validation')} for row in history]}
    result['interpretation']=[
        'Exact corruption-source labels exist; reliability is displacement/spacing soft confidence, not a binary noise flag.',
        'Both prior trainers create fresh Adam and restart corruption epoch at zero; neither restores baseline moments for a fresh continuation.',
        'Prior best checkpoints were selected at epochs 5/10(/15); epoch zero was not a selection candidate.',
        'Continuation degradation is not yet causally attributed. Patch validation is not a guarantee of whole-object quality.',
        'our2 lowers flow LR, restores Adam moments, advances corruption epochs and validates epoch zero/every epoch; these require controlled server verification.'
    ]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','output':str(a.output),'supervision_available':True}))


if __name__=='__main__': main()
