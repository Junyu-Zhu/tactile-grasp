#!/usr/bin/env python3
"""Preregistered same-trial descriptive association, no inferential fold pooling."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
import evaluate as e

def num(x):return float(x) if x not in ('',None) else float('nan')
def main():
 p=argparse.ArgumentParser();p.add_argument('--force-trials',type=Path,required=True);p.add_argument('--current-evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 es=json.loads((a.current_evaluation/'summary.json').read_text());assert es['status']=='complete' and es['new_runs']==12 and not es.get('synthetic',False)
 for name,h in es['output_hashes'].items():assert e.sha(a.current_evaluation/name)==h
 rows=e.readcsv(a.current_evaluation/'trials.csv');cur={}
 for r in rows:
  if r['role']=='validation' and r['point']=='FPR0.05' and r['rule']=='raw':
   key=(r['fold'],int(r['seed']),r['episode'],r['group']);assert key not in cur;cur[key]=r
 cal={(r['fold'],int(r['seed'])):float(r['static_fpr']) for r in e.readcsv(a.current_evaluation/'metrics.csv') if r['role']=='calibration' and r['point']=='FPR0.05' and r['rule']=='raw' and r['group']=='F_history_new'}
 force={}
 force_files=sorted(a.force_trials.rglob('per_trial_axis.csv')) if a.force_trials.is_dir() else [a.force_trials]
 if not force_files:raise ValueError('Missing force outputs')
 for f in force_files:
  audit=json.loads(f.with_name('AUDIT.json').read_text())
  if audit.get('status')!='pass' or audit.get('test_consumed') is not False or audit['output_hashes']['per_trial_axis.csv']!=e.sha(f):raise ValueError('Force result audit mismatch')
 force_rows=[r for f in force_files for r in e.readcsv(f)]
 for r in force_rows:
  if r['role']!='validation' or r.get('population')!='all' or r.get('task')!='slip_force':continue
  key=(r['fold'],int(r['seed']),r['episode_id'],r['variant'],r['axis']);assert key not in force;force[key]=r
 for key in sorted({k[:4] for k in force}):
  rr=[r for k,r in force.items() if k[:4]==key]
  if {r['axis'] for r in rr}!={'shear_x','shear_y','normal'}:raise ValueError('Expected exactly three force axes')
  force[(*key,'mean_axes')]={'mae':float(np.mean([num(r['mae']) for r in rr]))}
 paired=[];excluded=[]
 for fold,seed,episode,group in sorted(cur):
  if group!='V_temporal':continue
  k=(fold,seed,episode);v=cur[(*k,'V_temporal')];old=cur[(*k,'F_history_old')];new=cur[(*k,'F_history_new')]
  def rates(r):return e.divide(num(r['fp']),num(r['fp'])+num(r['tn'])),e.divide(num(r['fn']),num(r['fn'])+num(r['tp']))
  vr,vm=rates(v);orr,om=rates(old);nr,nm=rates(new)
  for axis in sorted({key[-1] for key in force if key[:3]==k}):
   if (*k,'old',axis) not in force or (*k,'new',axis) not in force:raise ValueError('Missing same-trial paired force')
   of=force[(*k,'old',axis)];nf=force[(*k,'new',axis)]
   paired.append(dict(fold=fold,seed=seed,episode=episode,axis=axis,old_MAE=num(of['mae']),new_MAE=num(nf['mae']),delta_MAE=num(nf['mae'])-num(of['mae']),V_static_fpr=vr,old_static_fpr=orr,new_static_fpr=nr,delta_static_fpr=nr-orr,old_minus_V_fpr=orr-vr,new_minus_V_fpr=nr-vr,delta_gross_miss=nm-om,old_gross_miss=om,new_gross_miss=nm,calibration_FPR=cal[(fold,seed)],threshold_transfer_gap=abs(nr-cal[(fold,seed)]),delta_false_starts=num(new['false_starts'])-num(old['false_starts'])))
  if not any(key[:3]==k for key in force):raise ValueError('Trial has no force pairing')
  for label,value in [('no_static',vr),('no_gross',vm)]:
   if not np.isfinite(value):excluded.append(dict(fold=fold,seed=seed,episode=episode,reason=label))
 averaged=[]
 for key in sorted({(r['fold'],r['episode'],r['axis']) for r in paired}):
  rr=[r for r in paired if (r['fold'],r['episode'],r['axis'])==key];assert {r['seed'] for r in rr}==set(e.SEEDS)
  averaged.append(dict(fold=key[0],episode=key[1],axis=key[2],**{name:float(np.mean([r[name] for r in rr])) for name in rr[0] if name not in ('fold','seed','episode','axis')}))
 correlations=[]
 for fold,axis in sorted({(r['fold'],r['axis']) for r in averaged}):
  rr=[r for r in averaged if (r['fold'],r['axis'])==(fold,axis)]
  for x,y in [('delta_MAE','delta_static_fpr'),('delta_MAE','delta_gross_miss'),('old_MAE','V_static_fpr'),('new_MAE','V_static_fpr'),('new_MAE','new_static_fpr')]:
   vals=np.array([[r[x],r[y]] for r in rr]);vals=vals[np.isfinite(vals).all(1)];valid=len(vals)>=3 and len(set(vals[:,0]))>1 and len(set(vals[:,1]))>1
   correlations.append(dict(fold=fold,axis=axis,x=x,y=y,n_trials=len(vals),spearman_r=float(spearmanr(vals[:,0],vals[:,1]).statistic) if valid else None,status='descriptive' if valid else 'undefined',unit='complete_trial_seedmean'))
 cases=[]
 for kind in ('force_improved_slip_not','lower_fpr_recall_cost','persistent_static_error','threshold_transfer'):
  rr=[r for r in averaged if r['axis']=='mean_axes' and np.isfinite(r['new_static_fpr'])]
  if kind=='force_improved_slip_not':rr=[r for r in rr if r['delta_MAE']<0 and r['delta_static_fpr']>=0];order=lambda r:(r['delta_MAE'],-r['delta_static_fpr'],r['fold'],r['episode'])
  elif kind=='lower_fpr_recall_cost':rr=[r for r in rr if r['delta_static_fpr']<0 and r['delta_gross_miss']>0];order=lambda r:(r['delta_static_fpr'],-r['delta_gross_miss'],r['fold'],r['episode'])
  elif kind=='threshold_transfer':order=lambda r:(-r['threshold_transfer_gap'],r['fold'],r['episode'])
  else:order=lambda r:(-r['new_static_fpr'],r['fold'],r['episode'])
  for rank,r in enumerate(sorted(rr,key=order)):cases.append(dict(kind=kind,rank=rank,selected=rank==0,**r))
 e.csvout(a.output/'representative_cases.csv',cases)
 e.csvout(a.output/'per_seed_trial.csv',paired);e.csvout(a.output/'seedmean_trial.csv',averaged);e.csvout(a.output/'correlations.csv',correlations);e.csvout(a.output/'exclusions.csv',excluded)
 e.js(a.output/'AUDIT.json',dict(status='pass',point='FPR0.05',rule='raw',paired_rows=len(paired),seedmean_rows=len(averaged),correlations=len(correlations),sources={**{str(f):e.sha(f) for f in force_files},str(a.current_evaluation/'summary.json'):e.sha(a.current_evaluation/'summary.json'),str(Path(__file__)):e.sha(__file__),str(Path(__file__).with_name('ASSOCIATION_PROTOCOL.md')):e.sha(Path(__file__).with_name('ASSOCIATION_PROTOCOL.md'))},outputs={f.name:e.sha(f) for f in a.output.glob('*.csv')},causal_claim=False))
if __name__=='__main__':main()
