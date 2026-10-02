#!/usr/bin/env python3
"""Fixed non-overlapping endpoint-window diagnostics; never used for selection."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from touchd_common import atomic_json,sha256
AXES=('fx','fy','fz');HORIZONS=(1,5,10);WINDOW=32;MIN_FINAL=8
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(y,p,previous=None):
 e=p-y;out={'n':len(y),'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(np.square(e).mean())),'bias':float(e.mean()),'prediction_variance':float(p.var()),'target_variance':float(y.var()),'prediction_exact_boundary_fraction':float((np.abs(p)==20).mean()),'prediction_outside_20_fraction':float((np.abs(p)>20).mean()),'target_boundary_fraction':float((np.abs(y)==20).mean()),'prediction_min':float(p.min()),'prediction_max':float(p.max())}
 if previous is not None:out.update(copy_reference_fraction_1e6=float((np.abs(p-previous)<=1e-6).mean()),copy_reference_mae=float(np.abs(p-previous).mean()))
 return out
def windows(indices):
 for start in range(0,len(indices),WINDOW):
  x=indices[start:start+WINDOW]
  if len(x)>=MIN_FINAL:yield x
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);force_trial=[];force_window=[];future_window=[];inputs={};r10=a.root.parent/'round10_htt_force_supervision_adaptation'
 for route in ('H','T_H'):
  for fold in range(1,5):
   support_path=r10/f'force_support/fold_p{fold}.json';support=json.loads(support_path.read_text());sm={e['episode_id']:e for e in support['entries']};inputs[str(support_path)]=sha256(support_path)
   for seed in (20260914,20260915,20260916):
    key=f'{route}_p{fold}_s{seed}';manifest_path=a.root/f'formal/predictions/{key}/prediction_manifest.json';manifest=json.loads(manifest_path.read_text());inputs[str(manifest_path)]=sha256(manifest_path)
    for e in manifest['entries']:
     s=sm[e['episode_id']];y=np.load(s['force_native_n_path']);q=np.load(e['prediction_path']);valid=np.arange(5,len(y))
     for ai,axis in enumerate(AXES):
      force_trial.append({'route':route,'fold':fold,'seed':seed,'role':s['role'],'episode_id':e['episode_id'],'leakage_group':s['leakage_group'],'axis':axis,'copy_reference':'previous force prediction at t-1',**metrics(y[valid,ai],q[valid,ai],q[valid-1,ai])})
      for wi,ix in enumerate(windows(valid)):force_window.append({'route':route,'fold':fold,'seed':seed,'role':s['role'],'episode_id':e['episode_id'],'leakage_group':s['leakage_group'],'window_index':wi,'t_start':int(ix[0]),'t_end':int(ix[-1]),'axis':axis,'copy_reference':'previous force prediction at t-1',**metrics(y[ix,ai],q[ix,ai],q[ix-1,ai])})
    future_dir=a.root/f'evaluation/future/{key}'
    for role in ('fit','selection','calibration','validation'):
     path=future_dir/f'predictions_{role}.pt';d=torch.load(path,map_location='cpu',weights_only=False);inputs[str(path)]=sha256(path);episodes=np.asarray(d['episode_id'],str);groups=np.asarray(d['leakage_group'],str);times=d['t'].numpy();target=d['y'].numpy();current=d['predictions']['predicted_current_persistence'][:,0].numpy()
     for episode in sorted(set(episodes)):
      base=np.flatnonzero(episodes==episode);base=base[np.argsort(times[base])];group=groups[base[0]];assert len(set(groups[base]))==1
      for wi,ix in enumerate(windows(base)):
       for method,tensor in d['predictions'].items():
        pred=tensor.numpy()
        for hi,h in enumerate(HORIZONS):
         for ai,axis in enumerate(AXES):future_window.append({'route':route,'fold':fold,'seed':seed,'role':role,'episode_id':episode,'leakage_group':group,'window_index':wi,'t_start':int(times[ix[0]]),'t_end':int(times[ix[-1]]),'method':method,'horizon':h,'axis':axis,'copy_reference':'route-own predicted current force',**metrics(target[ix,hi,ai],pred[ix,hi,ai],current[ix,ai])})
 write(a.output/'FORCE_TRIAL_AXIS_DIAGNOSTICS.csv',force_trial);write(a.output/'FORCE_WINDOW_AXIS_DIAGNOSTICS.csv',force_window);write(a.output/'FUTURE_WINDOW_AXIS_DIAGNOSTICS.csv',future_window)
 result={'schema':'round16_fixed_window_diagnostics_v1','status':'complete','provenance':'fixed descriptive diagnostic window added after results in response to independent review; not preregistered','window_definition':{'ordered_endpoint_windows':'non-overlapping consecutive blocks within each episode','size':WINDOW,'final_window_minimum':MIN_FINAL,'force_valid_t_start':5,'future_valid_t_start':'in inherited future prediction payload (t>=13)','cross_episode_windows':False},'selection_or_tuning_use':False,'training_or_checkpoint_selection_use':False,'changes_registered_primary_metrics_or_ci':False,'test_consumed':False,'rows':{'force_trial_axis':len(force_trial),'force_window_axis':len(force_window),'future_window_axis':len(future_window)},'input_hashes':inputs,'output_hashes':{n:sha256(a.output/n) for n in ('FORCE_TRIAL_AXIS_DIAGNOSTICS.csv','FORCE_WINDOW_AXIS_DIAGNOSTICS.csv','FUTURE_WINDOW_AXIS_DIAGNOSTICS.csv')}};atomic_json(a.output/'AUDIT.json',result);print(json.dumps({'status':'complete','rows':result['rows']}))
if __name__=='__main__':main()
