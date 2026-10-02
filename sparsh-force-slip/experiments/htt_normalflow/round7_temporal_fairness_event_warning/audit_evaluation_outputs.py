#!/usr/bin/env python3
"""Independent arithmetic, coverage and provenance acceptance of event outputs."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;e=r/'evaluation/formal_analysis';s=json.loads((e/'summary.json').read_text());checks={};fail=[]
 def ck(k,v):
  checks[k]=bool(v)
  if not v:fail.append(k)
 ck('complete12',s['status']=='complete' and s['evaluated_runs']==12)
 for kind in ('source_hashes','output_hashes'):
  ck(kind,bool(s[kind]) and all(Path(n).is_file() and sha(Path(n))==h for n,h in s[kind].items()))
 rows=list(csv.DictReader((e/'metrics.csv').open()));neural=[x for x in rows if x['method_type']=='neural']
 keys=[(x['group'],x['seed'],x['horizon'],x['population'],x['rule'],x['operating_point']) for x in neural]
 ck('neural_unique_3168',len(keys)==3168 and len(set(keys))==3168)
 checked=0;undefined=0;never=0
 for i,x in enumerate(rows):
  if x['threshold_status']!='available':
   ck(f'unavailable_{i}',x['threshold'] in ('','None') and x['n']=='');undefined+=1;continue
  tn,fp,fn,tp=(int(x[k]) for k in ['tn','fp','fn','tp']);n=int(x['n']);pos=int(x['positive']);fpr=fp/(tn+fp);rec=tp/(tp+fn)
  f1=.5*(2*tp/max(1,2*tp+fp+fn)+2*tn/max(1,2*tn+fp+fn))
  v=(min(tn,fp,fn,tp)>=0 and tn+fp+fn+tp==n and fn+tp==pos and abs(float(x['prevalence'])-pos/n)<1e-12 and abs(float(x['frame_fpr'])-fpr)<1e-12 and abs(float(x['frame_recall'])-rec)<1e-12 and abs(float(x['balanced_accuracy'])-.5*(rec+1-fpr))<1e-12 and abs(float(x['macro_f1'])-f1)<1e-12 and 0<=float(x['average_precision'])<=1 and 0<=float(x['brier'])<=1)
  if not v:fail.append('metric_arithmetic_'+str(i))
  checked+=1
  if x['never_alarm']=='True':
   never+=1
   if fp+tp:fail.append('never_alarm_'+str(i))
 ck('metric_arithmetic',not any(x.startswith('metric_arithmetic_') for x in fail))
 ck('never_alarm_explicit_consistent',not any(x.startswith('never_alarm_') for x in fail))
 selection=list(csv.DictReader((e/'rule_selection.csv').open()));selected=[x for x in selection if x['group'] in ['A_visual','B_force','C_force_delta','D_visual_delta'] and x['selected']=='True']
 ck('one_rule_per_run_h_population',len(selected)==72 and len({(x['group'],x['seed'],x['horizon'],x['population']) for x in selected})==72)
 ck('trial_shards24',len(list((e/'trials').glob('H*.csv')))==24)
 ck('platt36',len(json.loads((e/'platt_models.json').read_text())['models'])==36)
 ck('baseline18',len(json.loads((e/'baseline_models.json').read_text())['models'])==18)
 pairs=list(csv.DictReader((e/'paired_bootstrap_ci.csv').open()));ck('trial_FA10_CI60',sum(x['operating_point']=='trial_FA_0.10' for x in pairs)==60)
 ck('failure_cases_nonempty',len(list(csv.DictReader((e/'failure_cases.csv').open())))>0)
 ck('sensitivity_nonempty',len(list(csv.DictReader((e/'sensitivity_metrics.csv').open())))>0)
 ck('r6_reference_nonempty',len(list(csv.DictReader((e/'r6_full_timeline_reference_metrics.csv').open())))>0)
 out={'status':'pass' if not fail else 'fail','checks':checks,'failures':fail,'metric_rows':len(rows),'arithmetic_rows':checked,'unavailable_workpoints':undefined,'never_alarm_rows':never,'source_summary_sha256':sha(e/'summary.json'),'audit_code_sha256':sha(Path(__file__)),'limitations':'Arithmetic/coverage audit supplements independent scientific code review; it does not turn reused development data into a blind test.'}
 (r/'EVALUATION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['status','failures','metric_rows','arithmetic_rows','unavailable_workpoints','never_alarm_rows']}));return bool(fail)
if __name__=='__main__':raise SystemExit(main())
