#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
def read(p):return list(csv.DictReader(Path(p).open()))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--supplement',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=json.loads((a.supplement/'SUPPLEMENT_AUDIT.json').read_text());checks={}
 checks['source_evaluation_bound']=s['source_evaluation_summary_sha256']==sha(a.evaluation/'SUMMARY.json')
 checks['all_hashes_match']=all(sha(a.supplement/k)==v for k,v in s['hashes'].items())
 film=read(a.supplement/'diagnostics/FILM_FEATURE_TIME_ROLE.csv');checks['film_expected_rows']=len(film)==12*4*9*192;checks['film_unique_cells']=len({(x['fold'],x['seed'],x['role'],x['time_index'],x['feature']) for x in film})==len(film)
 sf=read(a.supplement/'metrics/SAME_FPR_ENVELOPE.csv');sr=read(a.supplement/'metrics/SAME_RECALL_ENVELOPE.csv');checks['same_fpr_grid']=len(sf)==36*101 and {round(float(x['target_static_FPR']),6) for x in sf}=={round(i/1000,6) for i in range(101)};checks['same_recall_grid']=len(sr)==36*1001 and {round(float(x['target_gross_recall']),6) for x in sr}=={round(i/1000,6) for i in range(1001)}
 checks['same_fpr_monotone']=all(np.all(np.diff([float(x['envelope_interpolated_gross_recall']) for x in sf if x['group']==g and x['fold']==str(f) and x['seed']==str(seed)])>=-1e-12) for g in ('V0','C0','M0') for f in range(1,5) for seed in (20260914,20260915,20260916))
 checks['same_recall_monotone']=all(np.all(np.diff([float(x['envelope_interpolated_static_FPR']) for x in sr if x['group']==g and x['fold']==str(f) and x['seed']==str(seed)])>=-1e-12) for g in ('V0','C0','M0') for f in range(1,5) for seed in (20260914,20260915,20260916))
 cases=read(a.supplement/'cases/CASE_ALIGNED_EVIDENCE.csv');checks['case_rows_and_figures']=len(cases)==3390 and len(list((a.supplement/'cases').glob('*.svg')))==5;checks['history_current_force_aligned']=all(np.allclose(np.asarray(json.loads(x['pred_force_history_xyz_json']))[-1],[float(x['pred_force_x']),float(x['pred_force_y']),float(x['pred_force_z'])]) for x in cases)
 recurrence=True
 for key in sorted({(x['case_index'],x['group']) for x in cases}):
  rr=[x for x in cases if (x['case_index'],x['group'])==key];rr.sort(key=lambda x:int(x['t']));prev=None;run=0;state=0
  for x in rr:
   t=int(x['t']);score=float(x['score']);th=float(x['original_calibration_FPR5_threshold'])
   if prev is None or t!=prev+1:run=0;state=0
   if score>=th:run+=1;state=int(run>=2)
   else:run=0;state=0
   recurrence &= int(x['raw_alarm'])==int(score>=th) and int(x['confirm2_alarm'])==state;prev=t
 checks['case_alarm_recurrence']=bool(recurrence)
 censor=True
 for key in sorted({(x['case_index'],x['group']) for x in cases}):
  rr=sorted([x for x in cases if (x['case_index'],x['group'])==key],key=lambda x:int(x['t']))
  for j,x in enumerate(rr):
   expected_reset=(j==0 or int(x['t'])!=int(rr[j-1]['t'])+1);expected_end=(j==len(rr)-1 or int(rr[j+1]['t'])!=int(x['t'])+1)
   censor &= int(x['native_gap_or_left_reset'])==int(expected_reset) and int(x['observed_segment_right_edge'])==int(expected_end)
   censor &= (not int(x['gross_event_left_censored']) or int(x['stage'])==2) and (not int(x['gross_event_right_boundary_censored']) or int(x['stage'])==2)
   censor &= (not int(x['raw_alarm_run_right_censored']) or int(x['raw_alarm'])==1) and (not int(x['confirm2_alarm_run_right_censored']) or int(x['confirm2_alarm'])==1)
 checks['case_gap_and_censor_flags']=bool(censor);checks['thresholds_unchanged']=s['original_thresholds_unchanged'];checks['test_not_consumed']=not s['test_consumed'];status='pass' if all(checks.values()) else 'fail';o={'schema':'round18_supplement_audit_v1','status':status,'checks':checks,'counts':s['counts'],'test_consumed':False};a.output.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2));assert status=='pass'
if __name__=='__main__':main()
