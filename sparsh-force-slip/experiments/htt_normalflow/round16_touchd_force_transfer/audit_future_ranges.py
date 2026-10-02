#!/usr/bin/env python3
"""Audit future prediction boundary, clip, range, variance, and copy behavior."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from touchd_common import atomic_json,sha256
AXES=('fx','fy','fz');HORIZONS=(1,5,10)
def write(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[];trial_rows=[];hashes={}
 for role in ('fit','selection','calibration','validation'):
  path=a.evaluation/f'predictions_{role}.pt';d=torch.load(path,map_location='cpu',weights_only=False);hashes[role]=sha256(path);y=d['y'].numpy();current=d['predictions']['predicted_current_persistence'][:,0].numpy();episodes=np.asarray(d['episode_id'],dtype=str);groups=np.asarray(d['leakage_group'],dtype=str)
  for method,tensor in d['predictions'].items():
   pred=tensor.numpy()
   for hi,h in enumerate(HORIZONS):
    for ai,axis in enumerate(AXES):
     q=pred[:,hi,ai];target=y[:,hi,ai];anchor=current[:,ai]
     rows.append({'role':role,'method':method,'horizon':h,'axis':axis,'n':len(q),'target_boundary_fraction':float(np.mean(np.abs(target)==20)),'prediction_exact_boundary_fraction':float(np.mean(np.abs(q)==20)),'prediction_outside_20_fraction':float(np.mean(np.abs(q)>20)),'prediction_min':float(q.min()),'prediction_max':float(q.max()),'prediction_variance':float(q.var()),'target_variance':float(target.var()),'copy_current_fraction_1e6':float(np.mean(np.abs(q-anchor)<=1e-6)),'copy_current_mae':float(np.mean(np.abs(q-anchor)))})
     for episode in sorted(set(episodes)):
      mask=episodes==episode;group_values=set(groups[mask])
      if len(group_values)!=1:raise RuntimeError('episode group drift')
      qq,tt,aa=q[mask],target[mask],anchor[mask]
      trial_rows.append({'role':role,'episode_id':episode,'leakage_group':next(iter(group_values)),'method':method,'horizon':h,'axis':axis,'n':int(mask.sum()),'target_boundary_fraction':float(np.mean(np.abs(tt)==20)),'prediction_exact_boundary_fraction':float(np.mean(np.abs(qq)==20)),'prediction_outside_20_fraction':float(np.mean(np.abs(qq)>20)),'prediction_min':float(qq.min()),'prediction_max':float(qq.max()),'prediction_variance':float(qq.var()),'target_variance':float(tt.var()),'copy_current_fraction_1e6':float(np.mean(np.abs(qq-aa)<=1e-6)),'copy_current_mae':float(np.mean(np.abs(qq-aa)))})
 write(a.output/'RANGE_DIAGNOSTICS.csv',rows);write(a.output/'TRIAL_RANGE_DIAGNOSTICS.csv',trial_rows);result={'schema':'round16_future_range_audit_v1','status':'complete','input_hashes':hashes,'rows':len(rows),'trial_rows':len(trial_rows),'diagnostics_sha256':sha256(a.output/'RANGE_DIAGNOSTICS.csv'),'trial_diagnostics_sha256':sha256(a.output/'TRIAL_RANGE_DIAGNOSTICS.csv'),'test_consumed':False};atomic_json(a.output/'SUMMARY.json',result);print(json.dumps(result))
if __name__=='__main__':main()
