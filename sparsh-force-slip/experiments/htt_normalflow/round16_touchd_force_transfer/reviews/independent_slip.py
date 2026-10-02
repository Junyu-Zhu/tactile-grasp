import sys,csv,json,collections,hashlib
from pathlib import Path
import numpy as np
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round16_touchd_force_transfer');C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round16_touchd_force_transfer');sys.path.insert(0,str(C.parent/'round13_trial_level_alarm_calibration/evaluation'));import r13_evaluate as r
read=lambda p:list(csv.DictReader(open(p)))
new=read(O/'evaluation/slip/thresholds.csv');manifest=json.loads((O/'evaluation/EVALUATION_MANIFEST.json').read_text());report={'checks':{}}
# Independent candidate enumeration and constraints on two new-route calibration sets.
checks=0
for run in manifest['runs']:
 if run['group'] not in ('H','T_H') or run['fold']!='htt_leave_p2' or int(run['seed'])!=20260914:continue
 eps=r.load_episodes(run,'calibration');own=[x for x in new if x['group']==run['group'] and x['fold']==run['fold'] and int(x['seed'])==int(run['seed'])]
 tables={}
 for k in (1,2,4):
  blocks=[]
  for ep in eps:
   q=[];window=[];prev=None
   for t,p in zip(ep.t,ep.score):
    if prev is None or t!=prev+1:window=[]
    window.append(p);q.append(min(window[-k:]) if len(window)>=k else -np.inf);prev=t
   blocks.append((np.array(q),ep.stage))
  candidates=np.unique(np.concatenate([q[np.isfinite(q)&np.isin(st,[0,2])] for q,st in blocks]+[np.array([np.nextafter(1.,np.inf)])]))
  fp=np.zeros(len(candidates));tp=fp.copy();static=0;gross=0;macro=[];anys=[]
  for q,st in blocks:
   a=q[st==0];b=q[st==2];sf=(a[None,:]>=candidates[:,None]).sum(1);gt=(b[None,:]>=candidates[:,None]).sum(1);fp+=sf;tp+=gt;static+=len(a);gross+=len(b)
   if len(a):macro.append(sf/len(a));anys.append(sf>0)
  tables[k]=(candidates,fp/static,tp/gross,np.mean(macro,axis=0),np.mean(anys,axis=0))
 for row in own:
  if row['family']=='historical':
   c,fpr,rec,macro,anys=tables[1];pt=row['point']
   if pt=='fixed0.5':value=.5
   elif pt=='maxBA':idx=max(range(len(c)),key=lambda i:((rec[i]+1-fpr[i])/2,-fpr[i],c[i]));value=c[idx]
   else:
    alpha=float(pt[3:]);allowed=np.flatnonzero(fpr<=alpha+1e-15);idx=max(allowed,key=lambda i:(rec[i],-fpr[i],c[i]));value=c[idx]
  else:
   c,fpr,rec,macro,anys=tables[int(row['k'])];constraint=macro if row['family']=='trial_macro_static_FPR' else anys;allowed=np.flatnonzero(constraint<=float(row['alpha'])+1e-15);idx=max(allowed,key=lambda i:(rec[i],-constraint[i],c[i]));value=c[idx]
  assert value==float(row['threshold']),(run['group'],row,value);checks+=1
report['checks']['new_route_thresholds_independently_refit_cal_only']=checks
oldroot=O.parent/'round13_trial_level_alarm_calibration';old=read(oldroot/'results/new_policies/thresholds.csv');hist=read(oldroot/'results/historical/metrics_48.csv');count=0
for x in new:
 if x['group']!='V':continue
 if x['family']=='historical':
  point='fixed_0.5' if x['point']=='fixed0.5' else x['point'];found=[y for y in hist if y['group']=='V_class_trial_balanced' and y['fold']==x['fold'] and y['seed']==x['seed'] and y['role']=='calibration' and y['point']==point and y['rule']=='raw']
 else:found=[y for y in old if y['group']=='V_class_trial_balanced' and all(y[k]==x[k] for k in ('fold','seed','family','alpha','k'))]
 assert len(found)==1 and float(found[0]['threshold'])==float(x['threshold']);count+=1
report['checks']['V_accepted_historical_thresholds_exact']=count
# Independently bootstrap frame FPR using complete-trial counts, seed average inside draws.
trials=read(O/'evaluation/slip/trials.csv');draws=json.loads((O/'evaluation/bootstrap/GROUP_DRAWS.json').read_text());cis=read(O/'evaluation/bootstrap/PAIRED_CI.csv');err=0;n=0
for fold in range(1,5):
 rows=[x for x in trials if x['role']=='validation' and x['fold']==f'htt_leave_p{fold}' and x['family']=='trial_macro_static_FPR' and x['alpha']=='0.05' and x['k']=='1'];g=sorted({x['leakage_group'] for x in rows});w=np.array([[d.count(a) for a in g] for d in draws[str(fold)]],float);diff=[]
 for seed in (20260914,20260915,20260916):
  v=[]
  for route in ('T_H','H'):
   rr=[x for x in rows if x['group']==route and int(x['seed'])==seed];fp=[sum(int(x['static_fp']) for x in rr if x['leakage_group']==a) for a in g];den=[sum(int(x['static_frames']) for x in rr if x['leakage_group']==a) for a in g];v.append((w@fp)/(w@den))
  diff.append(v[0]-v[1])
 reps=np.mean(diff,axis=0);q=np.quantile(reps[np.isfinite(reps)],[.025,.5,.975],method='linear');ref=next(x for x in cis if x['task']=='slip' and int(x['fold'])==fold and x['candidate']=='T_H' and x['base']=='H' and x['policy']=='trial_macro_static_FPR|||0.05|1' and x['metric']=='frame_static_FPR');assert int(ref['valid'])==int(np.isfinite(reps).sum());err=max(err,max(abs(q[i]-float(ref[k])) for i,k in enumerate(('q025','q50','q975'))));n+=1
assert err<1e-15;report['checks']['slip_core_ci_recomputed']=n;report['ci_max_abs_error']=err
print(json.dumps(report,indent=2))
