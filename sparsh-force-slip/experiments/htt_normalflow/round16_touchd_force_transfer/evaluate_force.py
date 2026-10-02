#!/usr/bin/env python3
"""Evaluate one R16 HTT force route with axis, change, stage, and trial diagnostics."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np,torch
from touchd_common import atomic_json,sha256

AXES=('fx','fy','fz');STAGES={0:'stable',1:'incipient',2:'gross'}
def stats(y,p):
 e=p-y
 return {'n':int(e.size),'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.square(e).mean())),'bias':float(e.mean()),
         'amplitude_mae':float(np.abs(np.abs(p)-np.abs(y)).mean()),'target_abs_mean':float(np.abs(y).mean()),'prediction_abs_mean':float(np.abs(p).mean()),
         'target_boundary_fraction':float((np.abs(y)==20).mean()),'prediction_outside_20_fraction':float((np.abs(p)>20).mean()),'prediction_min':float(p.min()),'prediction_max':float(p.max())}
def writecsv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--support',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 support=json.loads(a.support.read_text());pred=json.loads(a.predictions.read_text());prepared=torch.load(a.prepared,map_location='cpu',weights_only=False)
 if pred['status']!='complete' or bool(pred['smoke'])!=a.smoke or pred['fold']!=support['fold'] or prepared['fold']!=support['fold']:raise RuntimeError('identity')
 sm={e['episode_id']:e for e in support['entries']};pm={e['episode_id']:e for e in pred['entries']};em={e['episode_id']:e for e in prepared['episodes']}
 if set(sm)!=set(pm) or set(sm)!=set(em):raise RuntimeError('episode coverage')
 blocks=[];trials=[]
 for eid in sorted(sm):
  s,pr,ep=sm[eid],pm[eid],em[eid];y=np.load(s['force_native_n_path']).astype(np.float32);q=np.load(pr['prediction_path']).astype(np.float32);stage=ep['stage'].numpy();role=s['role'];n=len(y);mask=np.arange(n)>=5
  if q.shape!=y.shape or len(stage)!=n or not np.isfinite(q).all():raise RuntimeError('shape/finite')
  blocks.append({'episode':eid,'group':s['leakage_group'],'role':role,'y':y,'p':q,'stage':stage,'mask':mask})
  for axis,name in enumerate(AXES):trials.append({'episode_id':eid,'leakage_group':s['leakage_group'],'role':role,'axis':name,**stats(y[mask,axis],q[mask,axis])})
 metrics=[]
 for role in ('fit','selection','calibration','validation'):
  chosen=[b for b in blocks if b['role']==role]
  for stage_name,stage_value in [('all',None),*[(v,k) for k,v in STAGES.items()]]:
   for axis,name in enumerate(AXES):
    yy=[];pp=[];dy=[];dp=[]
    for b in chosen:
     m=b['mask'] if stage_value is None else b['mask']&(b['stage']==stage_value);yy.append(b['y'][m,axis]);pp.append(b['p'][m,axis])
     cm=m.copy();cm[:10]=False;indices=np.flatnonzero(cm);dy.append(b['y'][indices,axis]-b['y'][indices-5,axis]);dp.append(b['p'][indices,axis]-b['p'][indices-5,axis])
    if not yy or not sum(map(len,yy)):continue
    row={'role':role,'stage':stage_name,'axis':name,**stats(np.concatenate(yy),np.concatenate(pp))};ddy,ddp=np.concatenate(dy),np.concatenate(dp)
    row.update(change_n=len(ddy),change_mae=float(np.abs(ddp-ddy).mean()) if len(ddy) else None,change_rmse=float(np.sqrt(np.square(ddp-ddy).mean())) if len(ddy) else None);metrics.append(row)
  yy=np.concatenate([b['y'][b['mask']] for b in chosen]);pp=np.concatenate([b['p'][b['mask']] for b in chosen]);metrics.append({'role':role,'stage':'all','axis':'vector_norm','n':len(yy),'mae':float(np.abs(np.linalg.norm(pp,axis=1)-np.linalg.norm(yy,axis=1)).mean()),'rmse':float(np.sqrt(np.square(np.linalg.norm(pp,axis=1)-np.linalg.norm(yy,axis=1)).mean()))})
 writecsv(a.output/'metrics.csv',metrics);writecsv(a.output/'trial_axis_metrics.csv',trials)
 result={'schema':'round16_force_evaluation_v1','status':'complete','route':pred['route'],'fold':pred['fold'],'seed':pred['seed'],'smoke':a.smoke,'test_consumed':False,'domain_local_only':True,'support_sha256':sha256(a.support),'predictions_sha256':sha256(a.predictions),'prepared_sha256':sha256(a.prepared),'metrics_sha256':sha256(a.output/'metrics.csv'),'trial_axis_metrics_sha256':sha256(a.output/'trial_axis_metrics.csv')}
 atomic_json(a.output/'SUMMARY.json',result);print(json.dumps(result))
if __name__=='__main__':main()
