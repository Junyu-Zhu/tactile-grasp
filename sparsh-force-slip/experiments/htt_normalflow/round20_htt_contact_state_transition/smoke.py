#!/usr/bin/env python3
"""Real-cache, separate-directory R20 checks before formal dispatch."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

import train as r20


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=torch.load(a.data,map_location='cpu',weights_only=False)
    data['round14_data_sha256']=r20.sha(a.data)
    r20.validate_data(data)
    fit=data['roles']['fit'];n=r20.normalizer(fit)
    x=r20.norm_x(fit['x'][:256],n).to(a.device)
    delta=fit['y'][:256]-fit['y_current'][:256,None,:]
    assert torch.allclose(delta,fit['y'][:256]-fit['y_current'][:256,None,:])
    support=json.loads(Path(data['provenance']['support']).read_text())
    by_eid={e['episode_id']:e for e in support['entries']}
    source_checks=[]
    for role in ('fit','selection','calibration','validation'):
        r=data['roles'][role]
        for i in (0,len(r['t'])//2,len(r['t'])-1):
            eid=r['episode_id'][i];t=int(r['t'][i]);arr=torch.from_numpy(__import__('numpy').load(by_eid[eid]['force_native_n_path']))
            actual=r['y'][i]-r['y_current'][i]
            expected=arr[[t+1,t+5,t+10]]-arr[t]
            assert torch.allclose(actual,expected,atol=1e-6)
            source_checks.append((role,eid,t))
    anchor=fit['x'][:256,-1,192:195]
    assert torch.allclose(anchor[:,None,:]+torch.zeros_like(delta),anchor[:,None,:].expand_as(delta))
    results={}
    for group in r20.GROUPS:
        model=r20.init_model(group,20260914).to(a.device)
        count=sum(q.numel() for q in model.parameters())
        y,traj=model(x,return_trajectory=True)
        assert y.shape==(256,3,3) and torch.count_nonzero(y)==0
        if group.startswith('T-'):
            assert traj.shape==(256,10,3)
            assert torch.allclose(y,traj[:,[0,4,9]])
            if group=='T-visual':
                xp=x.clone();xp[...,192:195]+=3
                assert torch.equal(model(xp),y)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        losses=[];grad_after_second={}
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            pred=model(x)
            loss=torch.nn.functional.smooth_l1_loss(pred,delta.to(a.device)/n['delta_scale'].to(a.device),beta=1)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
            if step==1:
                grad_after_second={name:float(q.grad.abs().sum()) for name,q in model.named_parameters() if q.grad is not None}
            optimizer.step();losses.append(float(loss))
        updated={k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])}
        assert 'readout.weight' in updated
        assert 'gru.weight_ih_l0' in updated
        assert 'gru.weight_hh_l0' in updated and grad_after_second['gru.weight_hh_l0']>0
        if group.startswith('T-'):
            assert 'transition.weight' in updated and grad_after_second['transition.weight']>0
        if group=='T-visual':
            yp=model(x)
            xp=x.clone();xp[...,192:195]+=3
            assert torch.equal(model(xp),yp)
            xv=x.clone();xv[...,0]+=3
            assert not torch.equal(model(xv),yp)
        results[group]={'parameters':count,'losses':losses,'updated':sorted(updated),
                        'grad_after_second':grad_after_second}
    assert (max(v['parameters'] for v in results.values())/min(v['parameters'] for v in results.values())-1)<.05
    # Exact epoch recovery on the accepted real cache; smoke outputs are isolated.
    start=time.time()
    uninterrupted=r20.train(data,a.output/'recovery_uninterrupted','T-force',1,20260914,a.device,3,10)
    command=[sys.executable,str(Path(r20.__file__)),'--data',str(a.data),'--output',str(a.output/'recovery_interrupted'),
             '--group','T-force','--fold','1','--seed','20260914','--device',a.device,
             '--max-epochs','3','--patience','10']
    env=os.environ.copy();env['R20_SMOKE_KILL_AFTER_LATEST_EPOCH']='0'
    crashed=subprocess.run(command,env=env,capture_output=True,text=True)
    assert crashed.returncode==91,crashed.stderr
    assert (a.output/'recovery_interrupted/latest.pth').exists()
    assert not (a.output/'recovery_interrupted/best.pth').exists()
    resumed=subprocess.run(command,capture_output=True,text=True)
    assert resumed.returncode==0,resumed.stderr
    ck1=torch.load(a.output/'recovery_uninterrupted/latest.pth',map_location='cpu',weights_only=False)
    ck2=torch.load(a.output/'recovery_interrupted/latest.pth',map_location='cpu',weights_only=False)
    assert r20.state_hash(ck1['model'])==r20.state_hash(ck2['model'])
    assert r20.state_hash(ck1['best_state'])==r20.state_hash(ck2['best_state'])
    for k in ('history','best_epoch','best_metric','wait','epoch'):
        assert ck1[k]==ck2[k],k
    assert torch.equal(ck1['sampler_rng'],ck2['sampler_rng'])
    assert torch.equal(ck1['torch_rng'],ck2['torch_rng'])
    assert ck1['python_rng']==ck2['python_rng']
    assert ck1['numpy_rng'][0]==ck2['numpy_rng'][0] and __import__('numpy').array_equal(ck1['numpy_rng'][1],ck2['numpy_rng'][1])
    assert set(ck1['optimizer'])==set(ck2['optimizer'])
    for key in ck1['optimizer']['state']:
        for field in ck1['optimizer']['state'][key]:
            x1=ck1['optimizer']['state'][key][field];x2=ck2['optimizer']['state'][key][field]
            assert torch.equal(x1,x2) if torch.is_tensor(x1) else x1==x2
    alias=torch.load(a.output/'recovery_interrupted/best.pth',map_location='cpu',weights_only=False)
    assert r20.state_hash(alias['model'])==r20.state_hash(ck2['best_state'])
    out={'status':'pass','groups':results,'recovery_state_sha256':r20.state_hash(ck1['model']),
         'recovery_history':ck1['history'],'three_epoch_pair_seconds':time.time()-start,
         'cache_sha256':data['round14_data_sha256'],'source_sha256':r20.sha(r20.__file__),
         'protocol_sha256':r20.sha(Path(r20.__file__).with_name('PROTOCOL.md')),
         'independent_source_checks':source_checks,'subprocess_crash_exit':crashed.returncode,
         'smoke_weights_excluded_from_formal':True}
    r20.atomic_json(out,a.output/'SMOKE.json')
    print(json.dumps({'status':'pass','parameter_counts':{k:v['parameters'] for k,v in results.items()},
                      'three_epoch_pair_seconds':out['three_epoch_pair_seconds']}))

if __name__=='__main__':main()
