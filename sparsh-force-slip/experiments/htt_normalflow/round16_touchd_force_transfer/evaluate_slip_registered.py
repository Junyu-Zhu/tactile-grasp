#!/usr/bin/env python3
"""Evaluate the locked R13 alarm family for the R16 V/H/T_H predictions."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
R13=HERE.parent/'round13_trial_level_alarm_calibration/evaluation'
sys.path.insert(0,str(R13))
import r13_evaluate as r13

ROLES=('train','calibration','validation')

def historical_points(cal):
 cache=r13.build_fit_cache(cal,1);candidates=cache.thresholds
 evaluated=[]
 for threshold in candidates:
  _,agg=r13.evaluate_policy(cal,float(threshold),1)
  evaluated.append((float(threshold),agg))
 points={'fixed0.5':.5,'maxBA':max(evaluated,key=lambda x:(x[1]['balanced_accuracy'],-x[1]['frame_static_FPR'],x[0]))[0]}
 for alpha in r13.ALPHAS:
  fit=r13.fit_cached(cache,'trial_macro_static_FPR',alpha,constraint_override='frame_static_FPR')
  if fit['status']=='fit':points[f'FPR{alpha:.2f}']=fit['threshold']
 return points

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 manifest=json.loads(a.manifest.read_text());runs=manifest['runs'];a.output.mkdir(parents=True,exist_ok=True)
 expected={(g,f'htt_leave_p{i}',s) for g in ('V','H','T_H') for i in range(1,5) for s in (20260914,20260915,20260916)}
 if {(r['group'],r['fold'],int(r['seed'])) for r in runs}!=expected or len(runs)!=36:raise RuntimeError('run grid')
 thresholds=[];metrics=[];trials=[];inputs={}
 for run in runs:
  meta={k:run[k] for k in ('group','fold','seed')};datasets={role:r13.load_episodes(run,role) for role in ROLES}
  for kind in ('predictions','endpoints'):
   for role in ROLES:inputs[run[kind][role]['path']]=run[kind][role]['sha256']
  cal=datasets['calibration'];historical=historical_points(cal)
  for point,threshold in historical.items():
   thresholds.append({**meta,'family':'historical','point':point,'k':1,'threshold':threshold,'selection':'fixed' if point=='fixed0.5' else 'calibration_only'})
   for rule,k in (('raw',1),('confirm2',2)):
    for role in ROLES:
     stats,agg=r13.evaluate_policy(datasets[role],threshold,k);rank=r13.rank_metrics(datasets[role])
     metrics.append({**meta,'role':role,'family':'historical','point':point,'rule':rule,'k':k,'threshold':threshold,**agg,**rank})
     trials.extend({**meta,'role':role,'family':'historical','point':point,'rule':rule,'k':k,'threshold':threshold,**x} for x in stats)
  caches={k:r13.build_fit_cache(cal,k) for k in r13.KS}
  for family in r13.FAMILIES:
   for alpha in r13.ALPHAS:
    for k in r13.KS:
     fit=r13.fit_cached(caches[k],family,alpha);thresholds.append({**meta,'family':family,'alpha':alpha,'k':k,**fit})
     if fit['status']!='fit':continue
     for role in ROLES:
      stats,agg=r13.evaluate_policy(datasets[role],fit['threshold'],k);rank=r13.rank_metrics(datasets[role])
      metrics.append({**meta,'role':role,'family':family,'alpha':alpha,'k':k,'threshold':fit['threshold'],**agg,**rank})
      trials.extend({**meta,'role':role,'family':family,'alpha':alpha,'k':k,'threshold':fit['threshold'],**x} for x in stats)
  for alpha in r13.ALPHAS:
   threshold=historical[f'FPR{alpha:.2f}']
   for role in ROLES:
    stats,agg=r13.evaluate_policy(datasets[role],threshold,4)
    metrics.append({**meta,'role':role,'family':'confirm4_reference','alpha':alpha,'k':4,'threshold':threshold,**agg})
    trials.extend({**meta,'role':role,'family':'confirm4_reference','alpha':alpha,'k':4,'threshold':threshold,**x} for x in stats)
 r13.write_csv(a.output/'thresholds.csv',thresholds);r13.write_csv(a.output/'metrics.csv',metrics);r13.write_csv(a.output/'trials.csv',trials)
 result={'schema':'round16_slip_registered_evaluation_v1','status':'complete','runs':36,'groups':['V','H','T_H'],'test_consumed':False,'validation_used_for_threshold_fitting':False,'threshold_rows':len(thresholds),'metric_rows':len(metrics),'trial_rows':len(trials),'input_hashes':inputs,'protocol':json.loads((HERE/'EVALUATION_PROTOCOL.json').read_text())['r13_exact_inheritance'],'hashes':{name:r13.sha256(a.output/name) for name in ('thresholds.csv','metrics.csv','trials.csv')}}
 r13.write_json(a.output/'SUMMARY.json',result);print(json.dumps({k:result[k] for k in ('status','runs','threshold_rows','metric_rows','trial_rows')},indent=2))
if __name__=='__main__':main()
