#!/usr/bin/env python3
"""Role-local metrics for frozen R16 routes on the old dedicated HTT force task."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from touchd_common import atomic_json,sha256
AXES=('shear_x','shear_y','normal')
def write(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--r10',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[];inputs={};excluded=[]
 for manifest_path in sorted(a.root.glob('*_p*_s*/prediction_manifest.json')):
  m=json.loads(manifest_path.read_text());assert m['status']=='complete' and not m['test_consumed'] and not m['training_or_tuning'];inputs[str(manifest_path)]=sha256(manifest_path)
  for e in m['entries']:
   assert e['role']!='test';prediction=np.load(e['prediction_path']);target=np.load(e['target_path']);assert prediction.shape==target.shape==(e['frames'],3)
   ix=np.arange(13,len(target))
   if not len(ix):excluded.append({'route':m['route'],'fold':m['fold'],'seed':m['seed'],'role':e['role'],'episode_id':e['episode_id'],'frames':e['frames'],'reason':'no_endpoint_t_ge13'});continue
   for ai,axis in enumerate(AXES):
    error=prediction[ix,ai]-target[ix,ai];delta=(prediction[ix,ai]-prediction[ix-5,ai])-(target[ix,ai]-target[ix-5,ai])
    rows.append({'route':m['route'],'fold':m['fold'],'seed':m['seed'],'role':e['role'],'episode_id':e['episode_id'],'leakage_group':e['leakage_group'],'axis':axis,'n':len(ix),'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt(np.square(error).mean())),'bias':float(error.mean()),'change_mae':float(np.abs(delta).mean()),'prediction_variance':float(prediction[ix,ai].var()),'target_variance':float(target[ix,ai].var()),'prediction_outside_20_fraction':float((np.abs(prediction[ix,ai])>20).mean()),'prediction_exact_boundary_fraction':float((np.abs(prediction[ix,ai])==20).mean()),'target_boundary_fraction':float((np.abs(target[ix,ai])==20).mean())})
 assert len({(r['route'],r['fold'],int(r['seed'])) for r in rows})==24 and all(r['role']!='test' for r in rows)
 write(a.output/'PER_TRIAL_AXIS.csv',rows)
 summary=[]
 for key in sorted({(r['route'],r['role'],r['axis']) for r in rows}):
  chosen=[r for r in rows if (r['route'],r['role'],r['axis'])==key]
  for metric in ('mae','rmse','bias','change_mae','prediction_variance','target_variance','prediction_outside_20_fraction','target_boundary_fraction'):
   summary.append({'route':key[0],'role':key[1],'axis':key[2],'metric':metric,'trial_macro_mean':statistics.fmean(float(r[metric]) for r in chosen),'trials':len(chosen)})
 write(a.output/'SUMMARY_METRICS.csv',summary)
 historical=a.r10/'force_aggregate';historical_refs={name:{'path':str(historical/name),'sha256':sha256(historical/name)} for name in ('AUDIT.json','per_run_summary.csv','paired_group_ci.csv')}
 with (historical/'per_run_summary.csv').open(newline='') as f:hr=list(csv.DictReader(f))
 historical_validation={}
 for variant in ('old','new'):
  historical_validation[variant]={}
  for axis in AXES:
   chosen=[r for r in hr if r['task']=='old_force_regression' and r['role']=='validation' and r['variant']==variant and r['population']=='all' and r['axis']==axis and r['aggregation']=='complete_trial_macro']
   assert len(chosen)==12 and sum(int(r['trials']) for r in chosen)==141
   historical_validation[variant][axis]=sum(float(r['mae'])*int(r['trials']) for r in chosen)/141
 result={'schema':'round16_old_htt_force_regression_evaluation_v2','status':'pass','runs':24,'routes':['H','T_H'],'roles':['calibration','train','validation'],'test_consumed':False,'training_or_tuning':False,'frame_start':13,'target':'accepted R10 dedicated HTT force task, native clipped N','axis_mapping':{'model_output_0':'shear_x','model_output_1':'shear_y','model_output_2':'normal','source':'accepted R10/R5 force target contract'},'r16_per_trial_rows':len(rows),'summary_aggregation':'descriptive arithmetic mean over per-run episode-axis rows','summary_trials_field_semantics':'row count includes the same physical episodes repeated across three seeds and overlapping folds; it is not a count of independent trials','validation_rows_per_route_axis':141,'excluded_no_endpoint_t_ge13':excluded,'historical_reference_only':historical_refs,'historical_validation_episode_weighted_mae_n':historical_validation,'historical_aggregation':'trials-weighted mean of the 12 validation/all/complete_trial_macro per-run rows for each variant/axis; 141 per-run episode rows per variant/axis, with physical episodes repeated across seeds/overlapping folds','historical_not_in_r16_transfer_ci':True,'old_task_is_not_sparsh_original_domain':True,'input_hashes':inputs,'outputs':{'PER_TRIAL_AXIS.csv':sha256(a.output/'PER_TRIAL_AXIS.csv'),'SUMMARY_METRICS.csv':sha256(a.output/'SUMMARY_METRICS.csv')}}
 atomic_json(a.output/'AUDIT.json',result);print(json.dumps({'status':'pass','runs':24,'rows':len(rows)}))
if __name__=='__main__':main()
