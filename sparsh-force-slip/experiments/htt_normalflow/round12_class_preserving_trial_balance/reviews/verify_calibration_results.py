import json,csv,hashlib
from pathlib import Path
import numpy as np
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round12_class_preserving_trial_balance');U=O/'calibration_uncertainty';E=O/'current_evaluation';M=O/'formal_delivery/EVALUATION_MANIFEST.json'
def rows(p):return list(csv.DictReader(Path(p).open()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((U/'AUDIT.json').read_text());assert a['status']=='pass' and not a['synthetic'] and a['runs']==72 and not a['validation_consumed'] and not a['deployment_threshold_changed'] and a['manifest_sha256']==sha(M)
for p,h in a['input_hashes'].items():assert sha(p)==h and 'validation' not in p
for n,h in a['outputs'].items():assert sha(U/n)==h
m=json.loads(M.read_text());draws=json.loads((U/'GROUP_DRAWS.json').read_text());rr=rows(U/'bootstrap_thresholds.csv');summ=rows(U/'threshold_uncertainty.csv');lookup={(r['group'],r['fold'],int(r['seed']),int(r['draw']),float(r['fpr_constraint'])):r for r in rr};assert len(lookup)==len(rr)==43200;checks=0
for run in m['runs']:
 raw=rows(run['predictions']['calibration']['path']);ep=rows(run['endpoints']['calibration']['path']);allgroups=[r['leakage_group'] for r in ep];names=sorted(set(allgroups));primary=[i for i,r in enumerate(raw) if int(r['stage'])!=1];labels=np.array([int(raw[i]['stage']) for i in primary]);scores=np.array([float(raw[i]['p_slip']) for i in primary]);gs=[allgroups[i] for i in primary];rng=np.random.default_rng(20260916+int(run['fold'][-1])-1)
 for di,w in enumerate(draws[run['fold']]):
  expected=dict(zip(names,np.bincount(rng.integers(0,len(names),len(names)),minlength=len(names)).tolist()));assert expected==w
  multiplicity=np.array([w[g] for g in gs]);ix=np.repeat(np.arange(len(primary)),multiplicity);y=labels[ix];p=scores[ix];neg=int((y==0).sum());pos=int((y==2).sum());status='invalid_no_static' if not neg else 'invalid_no_gross' if not pos else 'valid'
  if status=='valid':
   order=np.argsort(-p,kind='stable');p=p[order];y=y[order];ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1];fpr=np.r_[0,np.cumsum(y==0)[ends]]/neg;rec=np.r_[0,np.cumsum(y==2)[ends]]/pos;threshold=np.r_[np.nextafter(1.,np.inf),p[ends]]
  for target in (.01,.05,.1):
   got=lookup[(run['group'],run['fold'],run['seed'],di,target)];assert got['status']==status
   if status=='valid':
    candidates=[i for i,f in enumerate(fpr) if f<=target+1e-15];j=max(candidates,key=lambda i:(rec[i],-fpr[i],threshold[i]));assert float(got['threshold'])==threshold[j] and float(got['bootstrap_calibration_fpr'])==fpr[j] and float(got['bootstrap_calibration_recall'])==rec[j];assert (got['never_alarm']=='True')==(threshold[j]>1)
   else:assert got['threshold']==''
   checks+=1
for r in summ:
 vals=[v for k,v in lookup.items() if k[:3]==(r['group'],r['fold'],int(r['seed'])) and k[-1]==float(r['fpr_constraint'])];valid=[v for v in vals if v['status']=='valid'];assert len(vals)==200 and int(r['valid'])==len(valid);assert float(r['valid_fraction'])==len(valid)/200
 for reason in ('invalid_no_static','invalid_no_gross'):assert int(r[reason])==sum(v['status']==reason for v in vals)
 if valid:
  t=[float(v['threshold']) for v in valid]
  for q in (.025,.5,.975):assert np.isclose(float(r[f'threshold_q{int(q*1000):03d}']),np.quantile(t,q))
  assert np.isclose(float(r['never_alarm_fraction']),sum(v['never_alarm']=='True' for v in valid)/len(valid))
old=O.parent/'round11_stable_negative_force_residual/current_evaluation/metrics.csv';before={(r['group'],r['fold'],r['seed'],r['role'],r['point'],r['rule']):r for r in rows(old)};current=rows(E/'metrics.csv');comparisons=0
for r in current:
 if r['group'] in ('V_original','F_history_original','V_balanced','F_history_balanced'):
  x=before[tuple(r[k] for k in ('group','fold','seed','role','point','rule'))]
  for key in ('threshold','tn','fp','fn','tp','AP','pAUC','static_fpr','gross_recall','false_starts_per_trial','event_recall','mean_delay'):assert r[key]==x[key],key
  comparisons+=1
out=dict(status='pass',reviewer='r10_eval independent of calibration uncertainty author',literal_group_replication_thresholds_checked=checks,summary_records_recomputed=len(summ),draw_generation_exact=True,calibration_only_inputs_verified=True,historical_metric_rows_exact_R11=comparisons,main_historical_thresholds_unchanged=True,all_seeds_retained=True,bootstrap_refit_diagnostic_not_deployment=True,uncertainty_audit_sha256=sha(U/'AUDIT.json'),verifier_source_sha256=sha(__file__))
(O/'reviews').mkdir(exist_ok=True);(O/'reviews/INDEPENDENT_CALIBRATION_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
