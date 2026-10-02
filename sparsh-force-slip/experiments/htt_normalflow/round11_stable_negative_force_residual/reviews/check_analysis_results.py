import json,csv,hashlib,sys
from pathlib import Path
import numpy as np
C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round11_stable_negative_force_residual');O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round11_stable_negative_force_residual');cache={}
def sha(p):
 p=Path(p)
 if str(p) not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  cache[str(p)]=h.hexdigest()
 return cache[str(p)]
def csvread(p):return list(csv.DictReader(Path(p).open()))
def match(p,h):assert sha(p)==h,(p,h)
a=json.loads((O/'analysis/AUDIT.json').read_text());assert a['status']=='pass' and a['runs']==36 and len(a['parity'])==72
for p,h in a['sources'].items():match(p,h)
for f,h in a['outputs'].items():match(O/'analysis'/f,h)
for r in a['prediction_artifacts']:match(r['path'],r['sha256'])
assert all(p['max_abs']<=1e-6 for p in a['parity']);metrics=csvread(O/'analysis/metrics.csv');assert len(metrics)==1320
preds={}
for r in a['prediction_artifacts']:
 if not r['path'].endswith('_components.csv'):preds[Path(r['path']).stem]=csvread(r['path'])
for r in metrics:
 key=f"{r['group']}_{r['fold']}_{r['seed']}_{r['intervention']}";rr=preds[key];th=float(r['threshold']);k=1 if r['rule']=='raw' else 2;active=False;count=0;prev=None;v=np.zeros(4,int)
 for x in rr:
  ident=(x['episode_id'],int(x['t']))
  if prev is None or ident[0]!=prev[0] or ident[1]!=prev[1]+1:active=False;count=0
  score=float(x['p_slip']);stage=int(x['stage'])
  if active:
   if score<th:active=False;count=0
  else:
   count=count+1 if score>=th else 0
   if count>=k:active=True
  if stage==0:v[1 if active else 0]+=1
  elif stage==2:v[3 if active else 2]+=1
  prev=ident
 assert list(v)==[int(r[z]) for z in ('tn','fp','fn','tp')],key
 if r['intervention'] not in ('base_only_independent_calibration','unperturbed'):
  orig=next(x for x in metrics if all(x[k]==r[k] for k in ('group','fold','seed','point','rule')) and x['intervention']=='unperturbed');assert r['threshold']==orig['threshold']
for key,rr in preds.items():
 if key.startswith('V_balanced') and not key.endswith('unperturbed'):
  prefix=key.split('_force_')[0];assert rr==preds[prefix+'_unperturbed']
components={}
for r in a['prediction_artifacts']:
 if r['path'].endswith('_components.csv'):
  rr=csvread(r['path']);components[Path(r['path']).stem]=rr
  for x in rr:
   b=float(x['base_probability']);f=float(x['full_probability']);z=float(x['residual_logit']);assert abs(z)<=2+1e-6
   # float32 sigmoid saturates: only test invertible probabilities away from endpoints.
   if 1e-5<b<1-1e-5 and 1e-5<f<1-1e-5:assert abs((np.log(f/(1-f))-np.log(b/(1-b)))-z)<.01
for r in csvread(O/'analysis/correction_harm.csv'):
 rr=[x for x in components[f"{r['group']}_{r['fold']}_{r['seed']}_components"] if x['episode_id']==r['episode']];th=float(r['threshold']);vals=dict(corrected=0,harmed=0,static_corrected=0,static_harmed=0,gross_corrected=0,gross_harmed=0)
 for x in rr:
  s=int(x['stage']);b=float(x['base_probability'])>=th;f=float(x['full_probability'])>=th
  if s==1:continue
  y=s==2;good=(b!=y and f==y);bad=(b==y and f!=y);vals['corrected']+=good;vals['harmed']+=bad;prefix='gross' if y else 'static';vals[prefix+'_corrected']+=good;vals[prefix+'_harmed']+=bad
 assert all(int(r[k])==v for k,v in vals.items())
x=json.loads((O/'cross_model_corrections/AUDIT.json').read_text());assert x['status']=='pass' and x['run_records']==24
for p,h in x['input_hashes'].items():match(p,h)
for f,h in x['outputs'].items():match(O/'cross_model_corrections'/f,h)
ct=csvread(O/'cross_model_corrections/per_trial.csv');cr=csvread(O/'cross_model_corrections/per_run.csv')
for r in ct:
 assert int(r['candidate_static_errors'])-int(r['base_static_errors'])==int(r['static_harmed'])-int(r['static_corrected']);assert int(r['candidate_gross_errors'])-int(r['base_gross_errors'])==int(r['gross_harmed'])-int(r['gross_corrected'])
 for groupkey,thkey in [('candidate','candidate_threshold'),('base','base_threshold')]:
  orig=next(z for z in metrics if z['group']==r[groupkey] and z['fold']==r['fold'] and z['seed']==r['seed'] and z['intervention']=='unperturbed' and z['point']=='FPR0.05' and z['rule']=='raw');assert float(r[thkey])==float(orig['threshold'])
for r in cr:
 rr=[z for z in ct if all(z[k]==r[k] for k in ('candidate','base','fold','seed'))];assert len(rr)==int(r['trials'])
 for key in ('static_frames','gross_frames','static_corrected','static_harmed','gross_corrected','gross_harmed','common_static_error','common_gross_error'):assert int(r[key])==sum(int(z[key]) for z in rr)
b=json.loads((O/'benchmark/F_residual_balanced.json').read_text());assert b['status']=='complete' and b['group']=='F_residual_balanced' and b['parity']['pass'];assert all(b['frozen'].values()) and b['force_executed'];match(C/'deployment/benchmark.py',b['benchmark_source_sha256']);match(C/'deployment/protocol.json',b['protocol_sha256']);assert sum(b['parameter_counts'].values())==b['total_deployment_parameters'];samples={}
for mode,reps in b['measurements'].items():
 assert len(reps)==3;samples[mode]=[]
 for r in reps:
  t=np.array(r['samples_ms']);assert len(t)==30 and np.isfinite(t).all() and (t>0).all();assert abs(np.median(t)-r['median_ms'])<1e-9;assert abs(np.quantile(t,.1)-r['p10_ms'])<1e-9;assert abs(np.quantile(t,.9)-r['p90_ms'])<1e-9;samples[mode].extend(t.tolist())
result={'status':'pass','reviewer':'r11_eval; not analysis/deployment/crossmodel author','unique_file_hashes_verified':len(cache),'checks':{'restored_parity_pairs':72,'intervention_confusions_independently_recomputed':1320,'predicted_artifact_hashes':len(a['prediction_artifacts']),'cross_model_runs':24,'cross_model_trial_rows':len(ct),'all_intervention_thresholds_unchanged':True,'V_bitwise_invariant':True,'residual_correction_harm_recomputed':True,'cross_model_thresholds_match_original_models':True,'cost_samples_per_mode':{k:len(v) for k,v in samples.items()}},'benchmark':{'risk_parity_max_abs':b['parity']['risk_max_abs'],'parameters':b['total_deployment_parameters'],'median_ms':{k:float(np.median(v)) for k,v in samples.items()}},'bound_artifact_hashes':{str(p):sha(p) for p in [O/'analysis/AUDIT.json',O/'cross_model_corrections/AUDIT.json',O/'benchmark/F_residual_balanced.json']},'limitations':['No inference of physical causality from interventions','C base jointly trained, not A control','Cross-model operating points use different own-calibrated thresholds','Shared GPU microbenchmark excludes disk/capture/controller'],'figure_review':'pending root figure generation'}
(C/'reviews/INDEPENDENT_ANALYSIS_ACTUAL_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
