import csv,json,hashlib,sys,collections,statistics
from pathlib import Path
import numpy as np,torch
from PIL import Image
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round16_touchd_force_transfer');E=O/'evaluation';C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round16_touchd_force_transfer');R={}
def read(p):return list(csv.DictReader(open(p)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# F1 raw old-task exports, roles and metrics across every run.
audit=json.loads((E/'old_force_regression/AUDIT.json').read_text());rows=read(E/'old_force_regression/PER_TRIAL_AXIS.csv');ix={(r['route'],r['fold'],int(r['seed']),r['episode_id'],r['axis']):r for r in rows};n=0;err=0;manifests={}
for path in sorted((E/'old_force_regression/exports').glob('*/prediction_manifest.json')):
 m=json.loads(path.read_text());assert not m['training_or_tuning'] and not m['test_consumed'];contract=json.loads(Path(m['contract']).read_text());assert sha(m['contract'])==m['contract_sha256'];eligible={e['episode_id']:e for e in contract['entries'] if e['task']=='force' and e['roles_by_fold'][m['fold']] in ('train','validation','calibration')};assert set(eligible)=={e['episode_id'] for e in m['entries']};manifests[m['route'],m['fold'],m['seed']]=m
 for e in m['entries']:
  assert e['role']==eligible[e['episode_id']]['roles_by_fold'][m['fold']];assert sha(e['prediction_path'])==e['prediction_sha256'];assert sha(e['target_path'])==e['target_sha256'];assert eligible[e['episode_id']]['force_target_definition']=='clip((6d_force - ref_force)[:, :3], -20 N, 20 N)'
  y=np.load(e['target_path']);p=np.load(e['prediction_path']);assert y.shape==p.shape==(e['frames'],3) and np.isfinite(p).all()
  if len(y)<=13:continue
  for ai,axis in enumerate(('shear_x','shear_y','normal')):
   q=ix[m['route'],m['fold'],m['seed'],e['episode_id'],axis];de=p[13:,ai]-y[13:,ai];vals={'mae':abs(de).mean(),'rmse':np.sqrt((de**2).mean()),'bias':de.mean()}
   for k,v in vals.items():err=max(err,abs(float(v)-float(q[k])))
   assert int(q['n'])==len(y)-13;n+=1
assert len(manifests)==24 and n==len(rows) and err==0
assert len(audit['excluded_no_endpoint_t_ge13'])==36
for r in read(E/'old_force_regression/SUMMARY_METRICS.csv'):
 selected=[x for x in rows if all(x[k]==r[k] for k in ('route','role','axis'))];assert len(selected)==int(r['trials']);assert abs(statistics.fmean(float(x[r['metric']]) for x in selected)-float(r['trial_macro_mean']))<1e-12
 if r['role']=='validation':assert len(selected)==141
R['F1']={'raw_metric_rows':n,'max_error':err,'run_coverage':24,'short_run_episode_combinations_excluded':36,'repeated_validation_rows_per_route_axis':141}
# F2 direct NPZ images and per-curve source samples.
images=read(E/'case_evidence/TACTILE_RAW_INDEX.csv')
for r in images:
 src=np.load(r['source_npz'])[r['array_key']][int(r['source_index'])];assert np.array_equal(src,np.asarray(Image.open(r['path'])));assert sha(r['path'])==r['sha256']
plots=read(E/'case_evidence/CASE_PLOT_INDEX.csv');assert len(plots)==48
selected=read(E/'failure_cases/SELECTED_CASES.csv');wanted={(r['task'],r['group']+'_p'+r['fold']+'_s'+r['seed']):r['episode_id'] for r in selected if r['task'] in ('force','future')}
for p in plots:assert wanted[p['task'],p['key']]==p['episode_id'] and sha(p['path'])==p['sha256']
source=read(E/'case_evidence/CASE_CURVE_SOURCE.csv');assert len(source)==66384;cache={};checks=0
for j,r in enumerate(source):
 if j%500:continue
 key=r['route']+'_p'+r['fold']+'_s'+r['seed'];ai=('fx','fy','fz').index(r['axis']);t=int(r['t'])
 if r['task']=='force':
  if (key,r['episode_id']) not in cache:
   sup=json.loads((O.parent/f'round10_htt_force_supervision_adaptation/force_support/fold_p{r["fold"]}.json').read_text());s=next(e for e in sup['entries'] if e['episode_id']==r['episode_id']);pm=json.loads((O/f'formal/predictions/{key}/prediction_manifest.json').read_text());pr=next(e for e in pm['entries'] if e['episode_id']==r['episode_id']);cache[key,r['episode_id']]=(np.load(s['force_native_n_path']),np.load(pr['prediction_path']))
  y,p=cache[key,r['episode_id']];yy,pp=y[t,ai],p[t,ai]
 else:
  if key not in cache:cache[key]=torch.load(E/f'future/{key}/predictions_validation.pt',map_location='cpu',weights_only=False)
  d=cache[key];mask=(np.asarray(d['episode_id'])==r['episode_id'])&(d['t'].numpy()==t);assert mask.sum()==1;h=(1,5,10).index(int(r['horizon']));yy=d['y'].numpy()[mask,h,ai][0];pp=d['predictions']['neural'].numpy()[mask,h,ai][0]
 assert float(yy)==float(r['target']) and float(pp)==float(r['prediction']);checks+=1
R['F2']={'unmodified_raw_images_exact':len(images),'registered_plots':len(plots),'source_rows':len(source),'sampled_curve_rows_exact':checks}
# F3 independent selected window statistics. All entries checked for lengths and per-episode bounds.
wdir=E/'window_diagnostics';a=json.loads((wdir/'AUDIT.json').read_text());assert 'not preregistered' in a['provenance'];assert not a['selection_or_tuning_use'] and not a['changes_registered_primary_metrics_or_ci'];checked={};maxerr=0;cache={}
for task,name in (('force','FORCE_WINDOW_AXIS_DIAGNOSTICS.csv'),('future','FUTURE_WINDOW_AXIS_DIAGNOSTICS.csv')):
 count=sampled=0
 for r in csv.DictReader(open(wdir/name)):
  count+=1;assert 8<=int(r['n'])<=32;assert int(r['t_end'])>=int(r['t_start']);assert r['role']!='test'
  if count%7001 and count!=1:continue
  key=r['route']+'_p'+r['fold']+'_s'+r['seed'];ai=('fx','fy','fz').index(r['axis']);start,end=int(r['t_start']),int(r['t_end'])
  if task=='force':
   sup=json.loads((O.parent/f'round10_htt_force_supervision_adaptation/force_support/fold_p{r["fold"]}.json').read_text());s=next(e for e in sup['entries'] if e['episode_id']==r['episode_id']);pm=json.loads((O/f'formal/predictions/{key}/prediction_manifest.json').read_text());pr=next(e for e in pm['entries'] if e['episode_id']==r['episode_id']);y=np.load(s['force_native_n_path']);p=np.load(pr['prediction_path']);ix=np.arange(start,end+1);y,p,anchor=y[ix,ai],p[ix,ai],p[ix-1,ai]
  else:
   ck=(key,r['role'])
   if ck not in cache:cache[ck]=torch.load(E/f'future/{key}/predictions_{r["role"]}.pt',map_location='cpu',weights_only=False)
   d=cache[ck];mask=(np.asarray(d['episode_id'])==r['episode_id'])&(d['t'].numpy()>=start)&(d['t'].numpy()<=end);hi=(1,5,10).index(int(r['horizon']));y=d['y'].numpy()[mask,hi,ai];p=d['predictions'][r['method']].numpy()[mask,hi,ai];anchor=d['predictions']['predicted_current_persistence'].numpy()[mask,0,ai]
  assert len(y)==int(r['n']);vals={'mae':abs(p-y).mean(),'prediction_variance':p.var(),'target_variance':y.var(),'copy_reference_mae':abs(p-anchor).mean(),'prediction_outside_20_fraction':(abs(p)>20).mean()}
  for k,v in vals.items():maxerr=max(maxerr,abs(float(v)-float(r[k])))
  sampled+=1
 checked[task]={'rows':count,'sampled_raw_windows':sampled}
assert maxerr==0;R['F3']={'coverage':checked,'max_sample_metric_error':maxerr,'posthoc_descriptive_only':True}
# F4 curve points independently re-evaluated against raw probability CSVs, no workpoint fitting.
manifest=json.loads((E/'EVALUATION_MANIFEST.json').read_text());curves=read(E/'low_fpr_curves/LOW_FPR_CURVES.csv');points=0
for run in manifest['runs']:
 rr=[r for r in curves if all(str(r[k])==str(run[k]) for k in ('group','fold','seed'))];assert rr and all(0<=float(r['fpr'])<=.1 for r in rr)
 raw=read(run['predictions']['validation']['path']);scores=np.array([float(r['p_slip']) for r in raw if int(r['stage']) in (0,2)]);y=np.array([int(r['stage'])==2 for r in raw if int(r['stage']) in (0,2)])
 for r in (rr[0],rr[len(rr)//2],rr[-1]):
  alarm=scores>=float(r['threshold']);fp=int((alarm&~y).sum());tp=int((alarm&y).sum());assert fp==int(r['fp']) and tp==int(r['tp']);assert fp/(~y).sum()==float(r['fpr']) and tp/y.sum()==float(r['tpr']);points+=1
R['F4']={'runs':len(manifest['runs']),'curve_rows':len(curves),'raw_points_recomputed':points}
# F5 inherited note fixed without old anchor claims; immutable result hashes from initial review.
notes=[]
for p in (E/'future').glob('*/SUMMARY.json'):
 d=json.loads(p.read_text());assert 'common R10' not in d['note'];assert 'route' in d['note'];notes.append(str(p))
R['F5']={'route_own_note_summaries':len(notes)}
R['immutable_hashes']={str(p.relative_to(O)):sha(p) for p in [E/'slip/thresholds.csv',E/'bootstrap/PAIRED_CI.csv',E/'failure_cases/SELECTED_CASES.csv']}
print(json.dumps(R,indent=2))
