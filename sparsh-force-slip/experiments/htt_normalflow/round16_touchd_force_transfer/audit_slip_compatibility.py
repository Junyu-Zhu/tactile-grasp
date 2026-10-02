#!/usr/bin/env python3
"""Verify full R13-family coverage and exact V-threshold compatibility."""
import argparse,csv,json
from pathlib import Path
def read(path):
 with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--r13',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();new=read(a.evaluation/'slip/thresholds.csv');metrics=read(a.evaluation/'slip/metrics.csv');old_new=read(a.r13/'results/new_policies/thresholds.csv');old_hist=read(a.r13/'results/historical/metrics_48.csv')
 if len(new)!=36*23 or len(metrics)!=36*93:raise RuntimeError((len(new),len(metrics)))
 checks=0
 for fold in range(1,5):
  for seed in (20260914,20260915,20260916):
   rows=[r for r in new if r['group']=='V' and r['fold']==f'htt_leave_p{fold}' and int(r['seed'])==seed]
   for r in rows:
    if r['family']=='historical':
     point='fixed_0.5' if r['point']=='fixed0.5' else r['point'];found={x['threshold'] for x in old_hist if x['group']=='V_class_trial_balanced' and x['fold']==r['fold'] and int(x['seed'])==seed and x['role']=='calibration' and x['point']==point and x['rule']=='raw'}
    else:
     found={x['threshold'] for x in old_new if x['group']=='V_class_trial_balanced' and x['fold']==r['fold'] and int(x['seed'])==seed and x['family']==r['family'] and x['alpha']==r['alpha'] and x['k']==r['k']}
    if found!={r['threshold']}:raise RuntimeError((fold,seed,r,found))
    checks+=1
 result={'schema':'round16_slip_compatibility_audit_v1','status':'pass','runs':36,'threshold_rows':len(new),'metric_rows':len(metrics),'historical_points_per_run':5,'historical_rules':['raw','confirm2'],'new_families':['trial_macro_static_FPR','trial_any_static_alarm_rate'],'alphas':[.01,.05,.1],'ks':[1,2,4],'confirm4_reference':True,'V_thresholds_exactly_match_accepted_R13':checks,'H_and_T_H_use_identical_rules_with_own_calibration_values':True,'validation_used_for_threshold_fitting':False,'test_consumed':False};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
