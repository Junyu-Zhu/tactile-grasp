#!/usr/bin/env python3
"""Unified R20 true-change and future-force evaluation on fixed roles."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

from train import GROUPS, SEEDS, ChangeModel, atomic_json, norm_x, normalizer, sha, state_hash, true_change, validate_data

HORIZONS=(1,5,10)


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(pred, target):
    err=pred-target
    return float(np.abs(err).mean()),float(np.sqrt(np.mean(err*err)))


def linear_fit(data,n):
    fit=data['roles']['fit']
    x=norm_x(fit['x'],n).numpy().reshape(len(fit['x']),-1)
    y=true_change(fit).numpy().reshape(len(x),-1)
    model=Ridge(alpha=1e-6,fit_intercept=True,solver='lsqr',tol=1e-4,max_iter=1000)
    model.fit(x,y)
    return model


def load_model(path,group,seed,cache_sha):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    ident=ck['identity']
    expected={'group':group,'fold':1,'seed':seed,'cache_sha256':cache_sha,
              'source_sha256':sha(Path(__file__).with_name('train.py')),
              'protocol_sha256':sha(Path(__file__).with_name('PROTOCOL.md'))}
    if any(ident.get(k)!=v for k,v in expected.items()):raise ValueError('best identity mismatch')
    model=ChangeModel(group);model.load_state_dict(ck['model']);model.eval()
    return model,ck


def model_predict(model,x,n,group):
    outputs=[];trajectories=[];latents=[]
    with torch.inference_mode():
        for xb in x.split(1024):
            pred,traj=model(xb,return_trajectory=True)
            outputs.append((pred*n['delta_scale']).numpy())
            if traj is not None:
                trajectories.append((traj*n['delta_scale']).numpy())
                input_x=xb[...,:192] if group=='T-visual' else xb
                z=model.gru(model.input(input_x))[0][:,-1]
                states=[]
                for step in range(10):
                    z=z+torch.tanh(model.transition(z));states.append(z)
                latents.append(torch.stack(states,dim=1).numpy())
                assert torch.allclose(traj,torch.stack([model.readout(s) for s in states],dim=1))
    pred=np.concatenate(outputs)
    if trajectories:return pred,np.concatenate(trajectories),np.concatenate(latents)
    return pred,None,None


def full_targets(role,support):
    entries={e['episode_id']:e for e in support['entries']}
    arrays={};labels_by_episode={}
    target=[];label=[]
    for eid,t in zip(role['episode_id'],role['t'].tolist()):
        if eid not in arrays:
            arrays[eid]=np.load(entries[eid]['force_native_n_path'])
            labels_by_episode[eid]=np.load(entries[eid]['label_path'])
        arr=arrays[eid]
        if t+10>=len(arr):raise ValueError('future missing')
        target.append(arr[np.arange(t+1,t+11)]-arr[t])
        label.append(int(labels_by_episode[eid][t]))
    return np.asarray(target,dtype=np.float32),np.asarray(label)


def rank_diagnostic(latent):
    if latent is None:return None
    z=latent[:,[0,4,9]].reshape(-1,latent.shape[-1]).astype(np.float64)
    centered=z-z.mean(0)
    eig=np.linalg.eigvalsh(centered.T@centered/max(1,len(z)-1)).clip(min=0)
    eff=float((eig.sum()**2)/max(1e-15,(eig*eig).sum()))
    return {'latent_mean_variance':float(z.var(0).mean()),'effective_rank':eff,
            'rank_fraction':eff/len(eig)}


def evaluate_seed(seed,prepare,root,out):
    cache=next(c for c in prepare['caches'] if c['seed']==seed)
    if sha(cache['path'])!=cache['sha256']:raise ValueError('cache changed')
    data=torch.load(cache['path'],map_location='cpu',weights_only=False);validate_data(data)
    n=normalizer(data['roles']['fit']);linear=linear_fit(data,n)
    support=json.loads(Path(cache['support']).read_text())
    rows=[];diagnostics=[];trial_rows=[]
    for role_name in ('calibration','validation'):
        role=data['roles'][role_name]
        x=norm_x(role['x'],n)
        gt=role['y'].numpy();current=role['y_current'].numpy();true_delta=gt-current[:,None,:]
        anchor=role['x'][:,-1,192:195].numpy()
        linear_delta=linear.predict(x.numpy().reshape(len(x),-1)).reshape(-1,3,3).astype(np.float32)
        all_ten,labels=full_targets(role,support)
        assert np.allclose(all_ten[:,[0,4,9]],true_delta,atol=1e-5)
        strata=np.where(np.max(np.abs(true_delta[:,2]),axis=1)<=.25,'stable',
                        np.where(np.max(np.abs(true_delta[:,2]),axis=1)>=1.,'changing','transition'))
        for group in ('hold','linear','ideal_GT_hold')+GROUPS:
            if group=='hold':pred_delta=np.zeros_like(true_delta);trajectory=None;latent=None
            elif group=='linear':pred_delta=linear_delta;trajectory=None;latent=None
            elif group=='ideal_GT_hold':pred_delta=np.zeros_like(true_delta);trajectory=None;latent=None
            else:
                path=root/'formal'/f'{group}_p1_s{seed}'/'best.pth'
                model,ck=load_model(path,group,seed,cache['sha256'])
                pred_delta,trajectory,latent=model_predict(model,x,n,group)
            anchor_used=current if group=='ideal_GT_hold' else anchor
            future=anchor_used[:,None,:]+pred_delta
            anchor_error=anchor_used-current
            change_error=pred_delta-true_delta
            assert np.allclose(future-gt,anchor_error[:,None,:]+change_error,atol=2e-5)
            npz=out/'predictions'/f'{group}_p1_s{seed}_{role_name}.npz'
            npz.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(npz,episode_id=np.array(role['episode_id']),t=role['t'].numpy(),
                                leakage_group=np.array(role['leakage_group']),label=labels,stratum=strata,
                                current_gt=current,anchor=anchor_used,deployable_anchor=anchor,gt=gt,true_delta=true_delta,
                                pred_delta=pred_delta,pred_future=future,
                                trajectory=trajectory if trajectory is not None else np.array([]),
                                ten_step_gt_delta=all_ten)
            d=rank_diagnostic(latent) or {}
            d.update({'seed':seed,'role':role_name,'group':group,'prediction_npz':str(npz),
                      'prediction_sha256':sha(npz),'output_delta_variance':float(pred_delta.var(axis=0).mean()),
                      'true_delta_variance':float(true_delta.var(axis=0).mean()),
                      'copy_ratio_abs_delta_le_0p01':float((np.abs(pred_delta)<=.01).mean()),
                      'future_outside_20_ratio':float((np.abs(future)>20).mean())})
            diagnostics.append(d)
            if trajectory is not None:
                for step in range(10):
                    mae,rmse=metrics(trajectory[:,step],all_ten[:,step])
                    rows.append({'seed':seed,'role':role_name,'group':group,'kind':'free_rollout',
                                 'horizon':step+1,'axis':'all','stratum':'all','mae':mae,'rmse':rmse,'n':len(role['t'])})
            for hidx,h in enumerate(HORIZONS):
                for axis in range(3):
                    for segment in ('all','stable','transition','changing'):
                        mask=np.ones(len(strata),bool) if segment=='all' else strata==segment
                        if not mask.any():continue
                        for kind,a0,b0 in (('future',future[:,hidx,axis],gt[:,hidx,axis]),
                                           ('true_change',pred_delta[:,hidx,axis],true_delta[:,hidx,axis])):
                            mae,rmse=metrics(a0[mask],b0[mask])
                            rows.append({'seed':seed,'role':role_name,'group':group,'kind':kind,
                                         'horizon':h,'axis':axis,'stratum':segment,'mae':mae,'rmse':rmse,'n':int(mask.sum())})
                        if segment=='all':
                            for eid in sorted(set(role['episode_id'])):
                                ix=np.array([x==eid for x in role['episode_id']])
                                a=anchor_error[ix,axis];c=change_error[ix,hidx,axis]
                                trial_rows.append({'seed':seed,'role':role_name,'group':group,'episode_id':eid,
                                                   'leakage_group':role['leakage_group'][int(np.flatnonzero(ix)[0])],
                                                   'horizon':h,'axis':axis,'n':int(ix.sum()),
                                                   'future_mae':float(np.abs(a+c).mean()),
                                                   'future_rmse':float(np.sqrt(np.mean((a+c)**2))),
                                                   'change_mae':float(np.abs(c).mean()),
                                                   'change_rmse':float(np.sqrt(np.mean(c*c))),
                                                   'anchor_mse':float(np.mean(a*a)),
                                                   'change_mse':float(np.mean(c*c)),
                                                   'cross_2mean':float(2*np.mean(a*c))})
    return rows,trial_rows,diagnostics


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare=json.loads(a.prepare.read_text())
    all_rows=[];all_trial=[];all_diag=[]
    for seed in SEEDS:
        rows,trials,diag=evaluate_seed(seed,prepare,a.root,a.output)
        all_rows+=rows;all_trial+=trials;all_diag+=diag
        write_csv(a.output/'metrics'/f'RUN_METRICS_s{seed}.csv',rows)
        write_csv(a.output/'metrics'/f'TRIAL_METRICS_s{seed}.csv',trials)
    write_csv(a.output/'metrics'/'RUN_METRICS.csv',all_rows)
    write_csv(a.output/'metrics'/'TRIAL_METRICS.csv',all_trial)
    atomic_json({'schema':'round20_evaluation_v1','diagnostics':all_diag,
                 'seed_count':len(SEEDS),'model_groups':list(GROUPS),'roles':['calibration','validation'],
                 'test_consumed':False},a.output/'EVALUATION.json')
    print(json.dumps({'status':'complete','run_rows':len(all_rows),'trial_rows':len(all_trial),
                      'diagnostics':len(all_diag)}))

if __name__=='__main__':main()
