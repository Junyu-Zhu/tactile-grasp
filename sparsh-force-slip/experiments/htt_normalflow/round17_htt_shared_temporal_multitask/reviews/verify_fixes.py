import csv,json,math,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from sklearn.metrics import roc_curve
O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round17_htt_shared_temporal_multitask');E=O/'evaluation';R=O/'results/final_report'
# Resolve output root from actual report location supplied by author, without writing it.
if not (R/'FUTURE_RUN_RESULTS.csv').exists():R=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round17_htt_shared_temporal_multitask/results/final_report')
def read(p):return list(csv.DictReader(open(p)))
out={'checks':defaultdict(int),'errors':[],'max_errors':defaultdict(float),'report_root':str(R)}
def compare(name,a,b,tol=1e-6):
 a=float(a);b=float(b);err=abs(a-b);out['checks'][name]+=1;out['max_errors'][name]=max(out['max_errors'][name],err)
 if not(np.isnan(a) and np.isnan(b)) and (not np.isfinite(err) or err>tol):out['errors'].append({'name':name,'a':a,'b':b})
fm=read(E/'future/metrics.csv');detail=read(E/'future/trial_axis_horizon.csv');curves=read(E/'slip/descriptive_matched_curves.csv');case=read(R/'FIXED_FUTURE_CASE_FORCE_CURVES.csv')
for r in read(R/'FUTURE_RUN_RESULTS.csv')+read(R/'BASELINE_RESULTS.csv'):
 pred=r.get('predictor',r.get('group'));rr=[x for x in fm if x['fold']==r['fold'] and x['seed']==r['seed'] and x['predictor']==pred and x['group']==(pred if pred in ('F','J') else 'F') and x['stratum']=='all' and x['role']=='validation'];expected=np.sqrt(np.average([float(x['future_error_sq']) for x in rr],weights=[int(x['n']) for x in rr]));compare('pooled_run_rmse',r['future_rmse'],expected)
for fold in range(1,5):
 for seed in (20260914,20260915,20260916):
  data=torch.load(O/f'prepared/p{fold}_s{seed}/prepared.pt',map_location='cpu',weights_only=False);d=data['roles']['validation'];y=d['y'].numpy();yc=d['y_current'].numpy();pc=d['x'][:,-1,192:].numpy();eps=np.array(d['episode_id']);mag=abs(y[:,2]-yc).max(1);q=data['joint_weight']['fit_change_strata'];st=np.full(len(y),'transition',object);stable=mag<=q['stable_le_n'];changing=(mag>=q['changing_ge_n'])&~stable;st[stable]='stable';st[changing]='changing';preds={g:np.load(E/f'predictions/{g}/p{fold}_s{seed}/validation.npz')['future'] for g in ('F','J')};preds['predicted_current_persistence']=np.repeat(pc[:,None,:],3,1);preds['ground_truth_current_persistence_ideal']=np.repeat(yc[:,None,:],3,1)
  rr=[r for r in detail if int(r['fold'])==fold and int(r['seed'])==seed]
  for r in rr:
   pred=r['predictor'];mask=eps==r['episode'];mask &= np.ones(len(y),bool) if r['stratum']=='all' else st==r['stratum'];compare('trial_cell_n',r['n'],sum(mask),0)
   if pred not in preds:continue
   hi=(1,5,10).index(int(r['horizon']));ai=('x','y','z').index(r['axis']);p=preds[pred][mask,hi,ai];yy=y[mask,hi,ai];e=p-yy;dep=(p-pc[mask,ai])-(yy-yc[mask,ai]);anc=pc[mask,ai]-yc[mask,ai]
   for key,v in {'future_mae':abs(e).mean(),'future_rmse':np.sqrt((e*e).mean()),'deployed_change_mae':abs(dep).mean(),'anchor_sq':(anc*anc).mean(),'cross_2_anchor_residual':(2*anc*dep).mean(),'prediction_abs_ge20_fraction':(abs(p)>=20).mean()}.items():
    if key in r:compare('trial_'+key,r[key],v,tol=2e-5)
  for g in ('S','J'):
   z=np.load(E/f'predictions/{g}/p{fold}_s{seed}/validation.npz');mask=z['stage']!=1;fpr,tpr,_=roc_curve(z['stage'][mask]==2,z['score'][mask],drop_intermediate=False)
   for r in [r for r in curves if r['group']==g and int(r['fold'])==fold and int(r['seed'])==seed]:
    if r['kind']=='same_fpr':compare('curve_same_fpr',r['gross_recall'],np.interp(float(r['target_fpr']),fpr,tpr))
    else:compare('curve_same_recall',r['static_fpr'],fpr[np.flatnonzero(tpr>=float(r['target_recall']))[0]])
  for r in [r for r in case if int(r['fold'])==fold and int(r['seed'])==seed]:
   ids=np.flatnonzero((eps==r['episode'])&(d['t'].numpy()==int(r['t'])));i=int(ids[0]);h=(1,5,10).index(int(r['horizon']));a=('x','y','z').index(r['axis'])
   for k,v in {'gt_current_n':yc[i,a],'predicted_current_n':pc[i,a],'gt_future_n':y[i,h,a],'F_future_n':preds['F'][i,h,a],'J_future_n':preds['J'][i,h,a]}.items():compare('case_raw_'+k,r[k],v,tol=2e-6)
# Independent aggregation of new trial cells must recover existing axis/horizon tables including fit-only linear baseline.
for r in fm:
 if r['role']!='validation' or (r['predictor'] not in ('F','J') and r['group']!='F'):continue
 rr=[x for x in detail if all(x[k]==r[k] for k in ('predictor','fold','seed','horizon','axis','stratum'))]
 for key in ('future_mae','deployed_change_mae','anchor_sq','cross_2_anchor_residual','future_error_sq'):
  compare('recover_cell_'+key,r[key],np.average([float(x[key]) for x in rr],weights=[int(x['n']) for x in rr]),tol=2e-4)
out['status']='pass' if not out['errors'] else 'fail';out['files']={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [E/'future/trial_axis_horizon.csv',E/'slip/descriptive_matched_curves.csv',R/'FIXED_FUTURE_CASE_FORCE_CURVES.csv',R/'FUTURE_RUN_RESULTS.csv',R/'BASELINE_RESULTS.csv']};print(json.dumps(out,indent=2))
