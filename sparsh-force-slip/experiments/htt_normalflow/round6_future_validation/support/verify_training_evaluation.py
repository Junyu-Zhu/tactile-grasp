#!/usr/bin/env python3
"""Independently recompute Round-6 final frame and trial aggregates."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 q=p.with_name(p.name+f'.tmp.{os.getpid()}');q.write_text(json.dumps(d,indent=2));os.replace(q,p)
def close(a,b,tol=2e-12):return abs(float(a)-float(b))<=tol
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.evaluation
 summary=json.loads((root/'summary.json').read_text())
 if summary.get('status')!='complete' or summary.get('neural_runs')!=9:raise ValueError('evaluation incomplete')
 if not all(sha(path)==digest for path,digest in summary['source_hashes'].items()):raise ValueError('evaluation source drift')
 if not all(sha(root/name)==digest for name,digest in summary['output_hashes'].items()):raise ValueError('evaluation output drift')
 data=torch.load(a.data,map_location='cpu',weights_only=False);outer=data['roles']['outer'];y=np.asarray(outer['y'],dtype=int);pred=np.load(root/'predictions.npz')
 rows=list(csv.DictReader((root/'metrics.csv').open()));by={(r['method'],r['point']):r for r in rows}
 if len(rows)!=len(pred.files)*5 or set(pred.files)!=set(summary['methods']):raise ValueError('method/metric cardinality mismatch')
 for method in pred.files:
  score=np.asarray(pred[method],float)
  if score.shape!=(len(y),) or not np.isfinite(score).all():raise ValueError(f'invalid prediction {method}')
  for point,threshold in summary['thresholds'][method].items():
   r=by[(method,point)];alarm=score>=threshold;tp=int(np.sum(alarm&(y==1)));fp=int(np.sum(alarm&(y==0)));fn=int(np.sum(~alarm&(y==1)));tn=int(np.sum(~alarm&(y==0)))
   expected={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'recall':tp/(tp+fn),'FPR':fp/(fp+tn),'BA':.5*(tp/(tp+fn)+tn/(tn+fp)),'AP':average_precision_score(y,score),'Brier':np.mean((score-y)**2)}
   if any(not close(r[k],v) for k,v in expected.items()):raise ValueError(f'raw metric mismatch {method}/{point}')
 trials=list(csv.DictReader((root/'trial_events.csv').open()));trial_by=defaultdict(list)
 for row in trials:trial_by[(row['method'],row['point'])].append(row)
 for key,r in by.items():
  cohort=trial_by[key]
  if sum(int(x['false_alarm_frames']) for x in cohort)!=int(float(r['fp'])):raise ValueError(f'trial FP mismatch {key}')
  detected=sum(x['event_detected']=='True' for x in cohort)
  if detected!=int(float(r['tp'])):raise ValueError(f'trial event/TP mismatch {key}')
 ci=list(csv.DictReader((root/'confidence_intervals.csv').open()))
 if not ci or any(int(x['valid_draws'])!=200 or int(x['requested_draws'])!=200 or not all(math.isfinite(float(x[k])) for k in ('lower','upper')) for x in ci):raise ValueError('cluster CI incomplete')
 paired=list(csv.DictReader((root/'paired_vs_A.csv').open()))
 for row in paired:
  candidate=row['candidate'];point=row['point'];metric=row['metric'];values=[]
  for seed in (20260914,20260915,20260916):
   base=by[(f'future_A_visual_{seed}',point)];cand=by[(f'future_{candidate}_{seed}',point)];values.append(float(cand[metric])-float(base[metric]))
  if not close(np.mean(values),row['mean_difference']):raise ValueError(f'paired mean mismatch {candidate}/{point}/{metric}')
  if int(row['valid_draws'])!=200 or row['shared_draw_all_three_seeds']!='True':raise ValueError('paired bootstrap incomplete')
 result={'format':'round6_training_evaluation_independent_audit_v1','status':'pass','summary_sha256':sha(root/'summary.json'),'prepared_data_sha256':sha(a.data),'checks':{'all_declared_source_hashes':True,'all_declared_output_hashes':True,'all_predictions_finite_complete':True,'all_frame_metrics_recomputed':True,'all_trial_confusions_reconciled':True,'all_cluster_ci_have_200_draws':True,'all_paired_observed_differences_recomputed':True},'counts':{'methods':len(pred.files),'metric_rows':len(rows),'trial_rows':len(trials),'ci_rows':len(ci),'paired_rows':len(paired),'outer_frames':len(y),'outer_positive_frames':int(y.sum())}}
 save(a.output,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
