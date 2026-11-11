"""One explicit configuration contract for training, checkpoints and inference."""
DEFAULT = dict(schema='our2-v1', enable_m1=False, enable_m2=False, stage='baseline',
    train_flow=True, width=64, region_divisor=32, max_move=.5, minimum=.5, maximum=2.,
    temperature=1., joint_condition=True, budget_policy='learned', allocation_strength=1.,
    use_reliability=True,use_hole=True,recovery_weight=.01,
    epochs=5,batch_size=8,workers=0,assignment_workers=8,validation_every=1,
    lr=1e-6,module_lr=1e-4,seed=21,train_patches_per_object=32,
    data_epoch_offset=30,restore_baseline_optimizer=True,gradient_clip=1.,
    baseline_sha256='c602029630079252fcd249078877ebaae1be083c99e829a82c8e15aafd01887e')


def validate_config(c):
    if set(c)!=set(DEFAULT): raise ValueError(f'Unknown/missing config fields: {set(c)^set(DEFAULT)}')
    if c['schema']!='our2-v1': raise ValueError('Unknown configuration version')
    stages={'baseline':(False,False),'m1_rel':(True,False),'m1_hole':(True,False),
            'm1_refine':(True,False),'m2_soft':(False,True),'m2_hard':(False,True),
            'joint_condition':(True,True),'joint_finetune':(True,True)}
    if c['stage'] not in stages or (c['enable_m1'],c['enable_m2'])!=stages[c['stage']]:
        raise ValueError('Stage/module switches disagree')
    for k in ['epochs','batch_size','assignment_workers','validation_every','width','region_divisor']:
        if not isinstance(c[k],int) or c[k]<1: raise ValueError(k)
    if c['workers']<0 or c['data_epoch_offset']<0: raise ValueError('Invalid epoch/workers')
    if not 0<=c['minimum']<=1<=c['maximum'] or c['temperature']<=0: raise ValueError('Budget bounds')
    if not 0<c['allocation_strength']<=1 or c['max_move']<0: raise ValueError('Invalid geometry bound')
    if min(c['lr'],c['module_lr'],c['gradient_clip'])<=0: raise ValueError('Invalid optimizer setting')
    if c['budget_policy'] not in ('learned','uniform','density','curvature'): raise ValueError('Budget policy')
