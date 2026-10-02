import json,csv,hashlib,sys,collections
from pathlib import Path
import numpy as np,torch
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round16_touchd_force_transfer'); E=O/'evaluation'; S=(20260914,20260915,20260916)
def read(p):return list(csv.DictReader(open(p)))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
report={'verification':'independent readonly CPU recomputation','checks':{},'max_errors':{}}
ci=read(E/'bootstrap/PAIRED_CI.csv'); assert len(ci)==8544
assert len({tuple(r[k] for k in ('task','fold','candidate','base','policy','metric')) for r in ci})==8544
D=json.loads((E/'bootstrap/GROUP_DRAWS.json').read_text()); slip=read(E/'slip/trials.csv'); W={}; groups={}
for f in range(1,5):
 g=sorted({r['leakage_group'] for r in slip if r['group']=='V' and r['fold']==f'htt_leave_p{f}' and int(r['seed'])==S[0] and r['role']=='validation'}); groups[f]=g
 rng=np.random.default_rng(2026091600+f); expected=[rng.choice(g,len(g),replace=True).tolist() for _ in range(2000)]; assert expected==D[str(f)]
 W[f]=np.array([[draw.count(x) for x in g] for draw in expected],float)
report['checks']['draws_regenerated_all_four_folds']=True
report['checks']['ci_rows_unique']=len(ci)
# Independently compare episode raw predictions against all force validation CSVs.
force_arrays={}; maxforce=0
for f in range(1,5):
 sup=json.loads((O.parent/f'round10_htt_force_supervision_adaptation/force_support/fold_p{f}.json').read_text()); es={r['episode_id']:r for r in sup['entries']}
 roles={role:{r['leakage_group'] for r in es.values() if r['role']==role} for role in ('fit','selection','calibration','validation')}
 assert all(not roles[a]&roles[b] for a in roles for b in roles if a!=b)
 assert roles['validation']==set(groups[f])
 for s in S:
  for route in ('H','T_H'):
   key=f'{route}_p{f}_s{s}'; pm=json.loads((O/f'formal/predictions/{key}/prediction_manifest.json').read_text()); pe={r['episode_id']:r for r in pm['entries']}
   rr=read(E/f'force/{key}/trial_axis_metrics.csv'); indexed={(r['episode_id'],r['axis']):r for r in rr if r['role']=='validation'}; accum=collections.defaultdict(list)
   for eid,e in es.items():
    if e['role']!='validation':continue
    y=np.load(e['force_native_n_path'])[5:]; p=np.load(pe[eid]['prediction_path'])[5:]; assert y.shape==p.shape and np.isfinite(p).all()
    err=p-y
    for ai,axis in enumerate(('fx','fy','fz')):
     vals={'mae':np.abs(err[:,ai]).mean(),'rmse':np.sqrt((err[:,ai]**2).mean()),'bias':err[:,ai].mean(),'amplitude_mae':np.abs(np.abs(p[:,ai])-np.abs(y[:,ai])).mean()}
     for met,val in vals.items():maxforce=max(maxforce,abs(float(val)-float(indexed[eid,axis][met])))
    accum[e['leakage_group']].append((y,p))
   force_arrays[route,f,s]={g:(np.concatenate([a for a,b in v]),np.concatenate([b for a,b in v])) for g,v in accum.items()}
assert maxforce<1e-6; report['max_errors']['force_trial_metrics']=maxforce
force_ci_error=0; force_checked=0
for row in ci:
 if row['task']!='force':continue
 f=int(row['fold']);ai=('fx','fy','fz').index(row['policy']);met=row['metric']; values=[]
 for s in S:
  pair=[]
  for route in ('T_H','H'):
   sums=[];counts=[]
   for g in groups[f]:
    y,p=force_arrays[route,f,s][g]; y=y[:,ai].astype(float);p=p[:,ai].astype(float);e=p-y
    v={'mae':np.abs(e),'rmse':e**2,'bias':e,'amplitude_mae':np.abs(np.abs(p)-np.abs(y))}[met];sums.append(v.sum());counts.append(len(v))
   result=W[f]@np.array(sums)/(W[f]@np.array(counts)); pair.append(np.sqrt(result) if met=='rmse' else result)
  values.append(pair[0]-pair[1])
 q=np.quantile(np.mean(values,axis=0),[.025,.5,.975],method='linear'); force_ci_error=max(force_ci_error,max(abs(q[i]-float(row[k])) for i,k in enumerate(('q025','q50','q975'))));force_checked+=1
assert force_ci_error<2e-6;report['checks']['force_ci_recomputed_from_raw']=force_checked;report['max_errors']['force_ci']=force_ci_error
# Future all route/fold/seed endpoint, route-own current anchor, error identity and raw metrics.
future={};maxfuture=0;maxidentity=0
for f in range(1,5):
 for s in S:
  pair={}
  for route in ('H','T_H'):
   key=f'{route}_p{f}_s{s}';d=torch.load(E/f'future/{key}/predictions_validation.pt',map_location='cpu',weights_only=False)
   raw=torch.load(O/f'formal/future_prepared/{key}/prepared.pt',map_location='cpu',weights_only=False)['roles']['validation']
   assert torch.equal(d['y'],raw['y']) and torch.equal(d['y_current'],raw['y_current']) and torch.equal(d['t'],raw['t'])
   assert torch.equal(d['predictions']['predicted_current_persistence'][:,0],raw['x'][:,-1,192:195])
   assert int(d['t'].min())>=13 and set(d['leakage_group'])==set(groups[f]);pair[route]=d
   rr=[r for r in read(E/f'future_diagnostics/{key}/trial_diagnostics.csv') if r['role']=='validation' and r['stratum']=='all']
   ep=np.asarray(d['episode_id']); y=d['y'].numpy();current=d['predictions']['predicted_current_persistence'][:,0].numpy();yc=d['y_current'].numpy()
   for r in rr:
    ai=('fx','fy','fz').index(r['axis']);hi=(1,5,10).index(int(r['horizon']));mask=ep==r['episode_id'];p=d['predictions'][r['method']].numpy();fe=p[mask,hi,ai]-y[mask,hi,ai];ce=current[mask,ai]-yc[mask,ai];de=(p[mask,hi,ai]-current[mask,ai])-(y[mask,hi,ai]-yc[mask,ai])
    vals={'future_mae':np.abs(fe).mean(),'future_mse':(fe**2).mean(),'change_mae':np.abs(de).mean(),'change_mse':(de**2).mean(),'cross_2_change_current':(2*de*ce).mean()}
    for met,val in vals.items():maxfuture=max(maxfuture,abs(float(val)-float(r[met])))
    maxidentity=max(maxidentity,float(abs(fe-de-ce).max()))
   future[route,f,s]=d
  a,b=pair['H'],pair['T_H'];assert a['episode_id']==b['episode_id'] and a['leakage_group']==b['leakage_group'] and torch.equal(a['t'],b['t']) and torch.equal(a['y'],b['y'])
report['checks']['future_route_endpoint_anchor_and_metrics_runs']=24;report['max_errors']['future_raw_metrics']=maxfuture;report['max_errors']['future_identity']=maxidentity
assert maxfuture<1e-6 and maxidentity<2e-5
# Core future CIs: all 4 folds x 3 horizons x 3 axes, direct route effect and own-persistence gain difference.
cache={};future_ci_error=0;nci=0
for row in ci:
 if row['task']!='future' or not row['policy'].endswith('|all') or row['metric']!='future_mae' or row['candidate'] not in ('T_H:neural','T_H_gain_over_persistence'):continue
 if row['candidate']=='T_H:neural' and row['base']!='H:neural':continue
 f=int(row['fold']);h,axis,_=row['policy'].split('|');hi=(1,5,10).index(int(h[1:]));ai=('fx','fy','fz').index(axis);vals=[]
 for s in S:
  results={}
  for route in ('H','T_H'):
   d=future[route,f,s];gg=np.asarray(d['leakage_group']);y=d['y'][:,hi,ai].numpy().astype(float)
   for method in ('neural','predicted_current_persistence'):
    er=np.abs(d['predictions'][method][:,hi,ai].numpy().astype(float)-y);sums=[er[gg==g].sum() for g in groups[f]];counts=[(gg==g).sum() for g in groups[f]];results[route,method]=W[f]@sums/(W[f]@counts)
  delta=results['T_H','neural']-results['H','neural']
  if row['candidate']=='T_H_gain_over_persistence':delta=delta-results['T_H','predicted_current_persistence']+results['H','predicted_current_persistence']
  vals.append(delta)
 q=np.quantile(np.mean(vals,axis=0),[.025,.5,.975],method='linear');future_ci_error=max(future_ci_error,max(abs(q[i]-float(row[k])) for i,k in enumerate(('q025','q50','q975'))));nci+=1
assert future_ci_error<2e-6;report['checks']['future_core_ci_recomputed_from_raw']=nci;report['max_errors']['future_ci']=future_ci_error
print(json.dumps(report,indent=2))
