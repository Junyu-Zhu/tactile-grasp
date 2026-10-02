from pathlib import Path
import json,csv,hashlib,itertools
import numpy as np
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round10_htt_force_supervision_adaptation');C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round10_htt_force_supervision_adaptation')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
a=O/'current_sensitivity/AUDIT.json';s=json.loads(a.read_text());assert s['status']=='pass' and s['runs']==36 and s['metrics']==1728 and s['perturbed_predictions']==72 and s['all_model_states_unchanged'] and s['V_force_invariance'] and s['no_recalibration']
for p,h in s['sources'].items():assert sha(p)==h
for p,h in s['output_hashes'].items():assert sha(a.parent/p)==h
assert len(s['prediction_artifacts'])==108 and len({x['path'] for x in s['prediction_artifacts']})==108
for p in s['prediction_artifacts']:assert sha(p['path'])==p['sha256']
groups=('V_temporal','F_history_old','F_history_new');folds=tuple('htt_leave_p'+str(i) for i in range(1,5));seeds=(20260914,20260915,20260916);grid=set(itertools.product(groups,folds,seeds))
assert {(r['group'],r['fold'],r['seed']) for r in s['original_prediction_parity']}==grid
assert all(0<=r['max_abs']<=1e-6 for r in s['original_prediction_parity'])
manifest=json.loads((O/'formal_delivery/EVALUATION_MANIFEST.json').read_text());cache={};paritymax=0
for r in manifest['runs']:
 key=(r['group'],r['fold'],r['seed']);original=read(r['predictions']['validation']['path']);assert sha(r['predictions']['validation']['path'])==r['predictions']['validation']['sha256']
 for intervention in ('unperturbed','force_fit_mean_zero','force_causal_lag1'):
  pp=a.parent/'predictions'/f'{key[0]}_{key[1]}_{key[2]}_{intervention}.csv';rows=read(pp);cache[(*key,intervention)]=rows
  assert [(x['episode_id'],x['t'],x['stage'],x['role']) for x in rows]==[(x['episode_id'],x['t'],x['stage'],x['role']) for x in original]
  p=np.array([float(x['p_slip']) for x in rows]);orig=np.array([float(x['p_slip']) for x in original]);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
  if intervention=='unperturbed':paritymax=max(paritymax,float(np.abs(p-orig).max()));assert np.abs(p-orig).max()<=1e-6
  if key[0]=='V_temporal':assert np.array_equal(p,orig)
metrics=read(a.parent/'metrics.csv');assert len(metrics)==1728
current=read(O/'current_evaluation/metrics.csv');thresholds={(r['group'],r['fold'],int(r['seed']),r['point']):float(r['threshold']) for r in current if r['role']=='calibration' and r['rule']=='raw'}
checked=0
for m in metrics:
 key=(m['group'],m['fold'],int(m['seed']));th=float(m['threshold']);assert th==thresholds[(*key,m['point'])];rr=cache[(*key,m['intervention'])];positive=np.array([float(x['p_slip'])>=th for x in rr]);stage=np.array([int(x['stage']) for x in rr]);alarm=positive.copy()
 if m['rule']=='confirm2':
  prev=np.r_[False,positive[:-1]];contiguous=np.array([i>0 and rr[i-1]['episode_id']==x['episode_id'] and int(rr[i-1]['t'])+1==int(x['t']) for i,x in enumerate(rr)]);alarm &=prev&contiguous
 else:assert m['rule']=='raw'
 for name,count in [('tp',((stage==2)&alarm).sum()),('fn',((stage==2)&~alarm).sum()),('fp',((stage==0)&alarm).sum()),('tn',((stage==0)&~alarm).sum())]:assert int(m[name])==count
 checked+=1
bp=O/'benchmark/F_history_new.json';b=json.loads(bp.read_text());assert b['status']=='complete' and b['group']=='F_history' and b['fold']=='htt_leave_p1' and b['seed']==20260914 and b['force_executed'] and all(b['frozen'].values()) and b['role']=='validation' and b['raw_image_range']==[0,13]
assert b['parity']['pass'] and all(r['pass'] for r in b['parity']['blocks'].values()) and b['parity']['risk_max_abs']<1e-5
for path,h in b['source_hashes'].items():assert sha(path)==h
assert sha(C/'deployment/benchmark.py')==b['benchmark_source_sha256'] and sha(C/'deployment/protocol.json')==b['protocol_sha256']
ta=O/'formal_delivery/TRAINING_AUDIT.json';assert sha(ta)==b['accepted_audit_sha256'];au=json.loads(ta.read_text());assert au['expected_count']==12 and len(au['accepted_runs'])==12
accepted=next(r for r in au['accepted_runs'] if r['fold']==b['fold'] and r['seed']==b['seed']);assert sha(accepted['checkpoint']['path'])==b['checkpoint_sha256']==accepted['checkpoint']['sha256']
for name in ('source_checkpoint','visual_checkpoint','force_checkpoint'):
 rec=b['upstream'][name];assert sha(rec['path'])==rec['sha256']
assert sum(b['parameter_counts'].values())==b['total_deployment_parameters']==101318664 and b['parameter_counts']['head']==143299
stats={}
for mode in ('cold_history','streaming'):
 reps=b['measurements'][mode];assert len(reps)==3;values=[]
 for rep in reps:
  vals=np.asarray(rep['samples_ms']);assert len(vals)==30 and np.isfinite(vals).all() and (vals>0).all();assert np.median(vals)==rep['median_ms'] and np.quantile(vals,.1)==rep['p10_ms'] and np.quantile(vals,.9)==rep['p90_ms'];values.extend(vals);assert rep['peak_allocated_bytes']>0
 stats[mode]={'samples':len(values),'median_ms':float(np.median(values)),'p10_ms':float(np.quantile(values,.1)),'p90_ms':float(np.quantile(values,.9)),'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in reps)}
out={'status':'pass','reviewer':'r10_force; independent of sensitivity and benchmark authors','scope':'actual outputs independently hashed and numeric checks; read-only CPU review, no GPU execution','checks':{'all_source_and_108_prediction_hashes':True,'36_exact_grid_and_original_parity':True,'V_all12_allinterventions_bitwise_original':True,'all1728_confusion_matrices_independently_recomputed':True,'all_thresholds_equal_original_calibration_workpoints':True,'model_freeze_runtime_evidence':s['all_model_states_unchanged'],'benchmark_12acceptedgrid_and_model_binding':True,'benchmark_fresh_image_block_and_risk_parity':True,'benchmark_all4_modules_frozen':True,'benchmark90_samples_each_mode_quantiles_recomputed':True,'parameter_sum_and_nonzero_memory':True},'input_audits':{str(a):sha(a),str(bp):sha(bp),str(ta):sha(ta)},'source_review_script_sha256':sha(__file__),'metric_rows_checked':checked,'max_original_probability_difference':paritymax,'benchmark_parity':b['parity'],'benchmark_stats':stats,'parameter_counts':b['parameter_counts'],'full_parameters':b['total_deployment_parameters'],'measurement_environment':{'gpu_name':b['gpu_name'],'CUDA_VISIBLE_DEVICES':b['CUDA_VISIBLE_DEVICES'],'torch_version':b['torch_version']},'limits':['Cache-based perturbation inference is not end-to-end latency','Shared-GPU timing does not establish architecture speed superiority','Deployment excludes image acquisition, disk I/O, checkpoint loading and robot control','Freeze assertion is verified recorded runtime state-hash evidence, not an independent GPU rerun']}
d=C/'reviews';d.mkdir(exist_ok=True);(d/'INDEPENDENT_SENSITIVITY_COST_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'pass','metrics':checked,'stats':stats}))
