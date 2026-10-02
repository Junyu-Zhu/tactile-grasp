import csv,json,hashlib,math
from pathlib import Path
import numpy as np
C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round12_class_preserving_trial_balance');O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round12_class_preserving_trial_balance');cache={}
def sha(p):
 p=Path(p)
 if str(p) not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  cache[str(p)]=h.hexdigest()
 return cache[str(p)]
def read(p):return list(csv.DictReader(Path(p).open()))
def verify(p,h):assert sha(p)==h,p
sens=json.loads((O/'current_sensitivity/AUDIT.json').read_text());assert sens['status']=='pass' and sens['runs']==24 and sens['metrics']==1152
for p,h in sens['sources'].items():verify(p,h)
for name,h in sens['output_hashes'].items():verify(O/'current_sensitivity'/name,h)
preds={}
for rec in sens['prediction_artifacts']:verify(rec['path'],rec['sha256']);preds[Path(rec['path']).stem]=read(rec['path'])
assert len(preds)==72 and len(sens['original_prediction_parity'])==24 and all(r['max_abs']<=1e-6 for r in sens['original_prediction_parity']);metrics=read(O/'current_sensitivity/metrics.csv');assert len(metrics)==1152
for r in metrics:
 key=f"{r['group']}_{r['fold']}_{r['seed']}_{r['intervention']}";rr=preds[key];v=np.zeros(4,int);active=False;count=0;prev=None;k=1 if r['rule']=='raw' else 2;th=float(r['threshold'])
 for x in rr:
  ident=(x['episode_id'],int(x['t']))
  if prev is None or ident[0]!=prev[0] or ident[1]!=prev[1]+1:active=False;count=0
  if active:
   if float(x['p_slip'])<th:active=False;count=0
  else:
   count=count+1 if float(x['p_slip'])>=th else 0
   if count>=k:active=True
  stage=int(x['stage'])
  if stage==0:v[1 if active else 0]+=1
  elif stage==2:v[3 if active else 2]+=1
  prev=ident
 assert list(v)==[int(r[z]) for z in ('tn','fp','fn','tp')]
 base=next(x for x in metrics if all(x[q]==r[q] for q in ('group','fold','seed','point','rule')) and x['intervention']=='unperturbed');assert r['threshold']==base['threshold']
 if r['group'].startswith('V_'):assert rr==preds[f"{r['group']}_{r['fold']}_{r['seed']}_unperturbed"]
case=json.loads((O/'fixed_case/AUDIT.json').read_text());assert case['status']=='pass' and case['models']==18 and case['metrics']==36
for p,h in case['source_hashes'].items():verify(p,h)
for name,h in case['outputs'].items():verify(O/'fixed_case'/name,h)
cm=read(O/'fixed_case/fixed_case_metrics.csv');cp=read(O/'fixed_case/fixed_case_probabilities.csv');assert len(cm)==36
for r in cm:
 static=int(r['tn'])+int(r['fp']);gross=int(r['tp'])+int(r['fn']);assert r['static_applicable']==str(bool(static)) and r['gross_applicable']==str(bool(gross))
 if not static:assert r['static_fpr']==''
 else:assert abs(float(r['static_fpr'])-int(r['fp'])/static)<1e-12
 if not gross:assert r['gross_miss']==''
 else:assert abs(float(r['gross_miss'])-int(r['fn'])/gross)<1e-12
 if r['rule']=='raw':
  rr=[x for x in cp if all(x[k]==r[k] for k in ('group','fold','seed'))];y=np.array([int(x['stage']) for x in rr]);p=np.array([float(x['p_slip']) for x in rr]);th=float(r['threshold']);assert int(((y==0)&(p>=th)).sum())==int(r['fp']);assert int(((y==2)&(p<th)).sum())==int(r['fn']);assert all(float(x['calibration_threshold'])==th for x in rr)
b=json.loads((O/'benchmark/F_class_trial_balanced.json').read_text());assert b['status']=='complete' and b['parity']['pass'] and all(b['frozen'].values()) and b['force_executed'];verify(C/'deployment/benchmark.py',b['benchmark_source_sha256']);verify(C/'deployment/protocol.json',b['protocol_sha256'])
for p,h in b['source_hashes'].items():verify(p,h)
assert b['total_deployment_parameters']==sum(b['parameter_counts'].values());samples={}
for mode,reps in b['measurements'].items():
 assert len(reps)==3;samples[mode]=[]
 for r in reps:
  x=np.array(r['samples_ms']);assert len(x)==30 and np.isfinite(x).all() and (x>0).all();assert abs(np.median(x)-r['median_ms'])<1e-10 and abs(np.quantile(x,.1)-r['p10_ms'])<1e-10 and abs(np.quantile(x,.9)-r['p90_ms'])<1e-10;samples[mode].extend(x.tolist())
out={'status':'pass','reviewer':'r11_eval not sensitivity/fixed-case/deployment author','unique_hashes_verified':len(cache),'checks':{'sensitivity_confusions_independently_recomputed':1152,'prediction_csv_hashes':72,'checkpoint_parity':24,'original_thresholds_unchanged':True,'visual_force_invariance':True,'fixed_case_models':18,'fixed_case_rates_and_missing_classes_checked':36,'R11coverage_sources_and_sync_proof_hashes_verified':True,'fresh_cache_parity':b['parity'],'cost_samples':{k:len(v) for k,v in samples.items()}},'cost_median_ms':{k:float(np.median(v)) for k,v in samples.items()},'parameter_counts':b['parameter_counts'],'bound_artifacts':{str(p):sha(p) for p in [O/'current_sensitivity/AUDIT.json',O/'fixed_case/AUDIT.json',O/'benchmark/F_class_trial_balanced.json']},'limitations':['Fixed case still descriptive not population inference','R11 neighbor diagnostic reused only one historical fold/seed, not 3seed coverage','Shared GPU resident-image microbenchmark excludes capture/disk/control'],'figure_review':'pending visible inspection'};(C/'reviews/INDEPENDENT_ANALYSIS_ACTUAL_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('bound_artifacts','checks')},indent=2))
