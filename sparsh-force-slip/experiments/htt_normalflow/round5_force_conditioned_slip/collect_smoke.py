#!/usr/bin/env python3
"""Collect real preparation evidence, rejecting missing or failed requirements."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
OUT=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round5_force_conditioned_slip')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
def main():
    records=[]
    paths=[OUT/'current/recovery_v2/force_continuous/training_summary.json']
    paths += [OUT/f'current/final_smoke/slip_{v}/training_summary.json' for v in ('V','F-old','F-adapt')]
    paths += [OUT/f'future/final_smoke/source/{v}/summary.json' for v in ('z_p_slip','z_p_slip_force','z_p_slip_force_pred_delta')]
    paths += [OUT/f'future/final_smoke/htt/{v}/summary.json' for v in ('risk','base','full_state')]
    for path in paths:
        data=json.loads(path.read_text());assert data['status']=='complete' and data['smoke'] is True,path
        if 'finite_gradients_all_epochs' in data:assert data['finite_gradients_all_epochs']
        if 'frozen_base_unchanged' in data:assert data['frozen_base_unchanged'] and data['all_trainable_tensors_changed']
        records.append({'path':str(path),'sha256':sha(path),'status':'pass','scope':'smoke_only'})
    recovery=[]
    pairs=[('force','current/recovery_v2/force_continuous','current/recovery_v2/force_actual_interrupted'),('slip','current/final_smoke/slip_F-adapt','current/final_smoke/slip_F-adapt_interrupted'),('source_future','future/final_smoke/source/z_p_slip_force_pred_delta','future/final_smoke/source_interrupted'),('htt_future','future/final_smoke/htt/full_state','future/final_smoke/htt_interrupted')]
    for name,left,right in pairs:
        pa,pb=OUT/left/'latest.pth',OUT/right/'latest.pth'
        a=torch.load(pa,map_location='cpu',weights_only=False);b=torch.load(pb,map_location='cpu',weights_only=False)
        fields=[k for k in ('model_state','optimizer_state','history','epoch','rng_state','best_model_state','best_score','best_epoch','normalization','condition_normalization','config','run_config') if k in a]
        checks={k:exact(a[k],b[k]) for k in fields};assert all(checks.values()),(name,checks)
        recovery.append({'name':name,'status':'pass','checks':checks,'continuous':{'path':str(pa),'sha256':sha(pa)},'resumed':{'path':str(pb),'sha256':sha(pb)},'method':'separate process interrupted after committed epoch1; unchanged total smoke budget; actual resume'})
    payload={'status':'pass','formal_training_started':False,'runs':records,'actual_recovery':recovery,'limitations':['smoke does not establish scientific improvement','large cached encoder is outside the optimizer by construction; see data encoder-state proof','future upstream is explicitly smoke and formal loaders reject it']}
    (OUT/'SMOKE_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'status':'pass','smoke_runs':len(records),'recovery_paths':len(recovery)}))
if __name__=='__main__':main()
