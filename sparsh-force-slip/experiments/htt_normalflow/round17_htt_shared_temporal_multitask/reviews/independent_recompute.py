import sys,json,csv,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from sklearn.metrics import roc_curve,average_precision_score
C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round17_htt_shared_temporal_multitask');O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round17_htt_shared_temporal_multitask');sys.path.insert(0,str(C));import multitask_train as mt
# Independent formulas and replay, no train/evaluator main called, no writes remotely.
torch.set_num_threads(4)
def read(p):return list(csv.DictReader(open(p)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pauc(st,s,w=None):
 m=(st==0)|(st==2);y=st[m]==2
 if len(np.unique(y))<2:return np.nan
 a,b,_=roc_curve(y,s[m],sample_weight=None if w is None else w[m],drop_intermediate=False);i=np.searchsorted(a,.1,side='right');return np.trapezoid(np.r_[b[:i],np.interp(.1,a,b)],np.r_[a[:i],.1])/.1
metrics=read(O/'evaluation/slip/metrics.csv');trials=read(O/'evaluation/slip/trials.csv');ths=read(O/'evaluation/slip/thresholds.csv');fmet=read(O/'evaluation/future/metrics.csv');draws=read(O/'evaluation/bootstrap/PAIRED_DRAWS.csv')
res={'checks':{},'runs':[],'selection_tie_differences':[],'errors':[]};store={};maxerr=defaultdict(float)
def chk(name,ok,detail=None):
 if not ok:res['errors'].append({'check':name,'detail':detail})
 res['checks'][name]=res['checks'].get(name,True) and bool(ok)
def compare(name,a,b,tol=1e-6):
 try:a=float(a);b=float(b)
 except:return
 if np.isnan(a) and np.isnan(b):return
 err=abs(a-b);maxerr[name]=max(maxerr[name],err);chk(name,np.isfinite(err) and err<=tol,{'a':a,'b':b})
def epstats(t,st,s,th,k):
 # Independent rolling-window alarm, with all consecutive segment boundaries explicit.
 alarm=np.array([i+1>=k and (k==1 or np.all(np.diff(t[i-k+1:i+1])==1)) and np.all(s[i-k+1:i+1]>=th) for i in range(len(t))]);br=np.r_[True,np.diff(t)!=1];prev=np.r_[False,alarm[:-1]] & ~br;starts=alarm&~prev;gross=st==2;static=st==0
 out=dict(static_frames=int(static.sum()),static_fp=int((alarm&static).sum()),gross_frames=int(gross.sum()),gross_tp=int((alarm&gross).sum()),false_starts=int((starts&static).sum()),alarm_starts=int(starts.sum()),events=0,event_hits=0,left_censored_events=0,right_boundary_events=0,delay_sum=0,delay_n=0,preexisting_alarm_events=0,preexisting_alarm_delay0_events=0)
 for i in np.flatnonzero(gross & (br|~np.r_[False,gross[:-1]])):
  j=i+1
  while j<len(t) and not br[j] and gross[j]:j+=1
  left=br[i];out['left_censored_events']+=int(left);out['right_boundary_events']+=int(j==len(t) or br[j]);hit=np.flatnonzero(alarm[i:j])
  if not left:
   out['events']+=1;prior=alarm[i-1];out['preexisting_alarm_events']+=int(prior)
   if len(hit):out['event_hits']+=1;out['delay_sum']+=int(hit[0]);out['delay_n']+=1;out['preexisting_alarm_delay0_events']+=int(prior and hit[0]==0)
 return out
for fold in range(1,5):
 for seed in mt.SEEDS:
  dp=O/f'prepared/p{fold}_s{seed}/prepared.pt';data=torch.load(dp,map_location='cpu',weights_only=False);norm=data['normalizer'];roles=data['roles'];chk('role_whitelist',set(roles)=={'fit','selection','calibration','validation'});r14=torch.load(data['provenance']['round14_prepared'],map_location='cpu',weights_only=False)
  for role,d in roles.items():
   chk('round14_x_y_endpoints_exact',all(torch.equal(d[k],r14['roles'][role][k]) for k in ('x','y','t','y_current')));chk('causal_shape',d['x'].shape[1:]==(9,195) and d['t'].min()>=13);chk('known_stages',set(d['stage'].tolist())<={0,1,2})
   for r2,d2 in roles.items():
    if role!=r2:chk('group_isolation',not(set(d['leakage_group'])&set(d2['leakage_group'])))
  fit=roles['fit'];xn=fit['x'].double();yn=fit['y'].double()
  chk('fit_normalizers_exact',all(torch.equal(norm[k],v.float()) for k,v in {'x_mean':xn.mean((0,1)),'x_std':xn.std((0,1),unbiased=False).clamp_min(1e-6),'y_mean':yn.mean(0),'y_std':yn.std(0,unbiased=False).clamp_min(1e-6)}.items()))
  st=fit['stage'].numpy();w=fit['slip_weight'].numpy();N=((st==0)|(st==2)).sum();eps=np.array(fit['episode_id'])
  for cl in (0,2):
   ee=np.unique(eps[st==cl]);expected=N/(2*len(ee))
   for e in ee:compare('class_trial_weight',w[(st==cl)&(eps==e)].sum(),expected,tol=1e-3)
  chk('incipient_zero_weight',np.all(w[st==1]==0))
  for group in mt.GROUPS:
   ckpath=O/f'formal/{group}/p{fold}_s{seed}/best.pth';ck=torch.load(ckpath,map_location='cpu',weights_only=False);latest=torch.load(ckpath.with_name('latest.pth'),map_location='cpu',weights_only=False);history=latest['history'];v=[r['selection_metric'] for r in history];best=int(np.argmin(v) if group=='F' else np.argmax(v));chk('earliest_best_history',best==ck['epoch']==ck['best_epoch']==latest['best_epoch']);chk('checkpoint_input_sha',ck['identity']['data_sha256']==sha(dp));chk('full_optimizer_scope',len(ck['optimizer']['param_groups'][0]['params'])==sum(len(list(getattr(mt.init_model(seed),n).parameters())) for n in mt.active_names(group)))
   initial=mt.init_model(seed);chk('inactive_initial_unchanged',all(torch.equal(ck['model'][n],v) for n,v in initial.state_dict().items() if n.split('.')[0] not in mt.active_names(group)))
   initial.load_state_dict(ck['model']);initial.cuda(0).eval();d=roles['selection'];x=(d['x']-norm['x_mean'])/norm['x_std'];sl=[];fu=[]
   with torch.no_grad():
    for ids in torch.arange(len(x)).split(1024):
     s,f=initial(x[ids].cuda(0));sl.append(torch.sigmoid(s).cpu());fu.append(f.cpu()*norm['y_std']+norm['y_mean'])
   score=torch.cat(sl).numpy();future=torch.cat(fu).numpy();pz=np.load(O/f'evaluation/predictions/{group}/p{fold}_s{seed}/selection.npz')
   if group in ('S','J'):
    compare('selection_replay_prediction',np.max(np.abs(score-pz['score'])),0,tol=2e-6);raw=mt.slip_pauc(d['stage'].numpy(),score);std=pauc(d['stage'].numpy(),score);compare('selection_metric_replay',raw,ck['best_metric'],tol=1e-7)
    if abs(raw-std)>1e-10:res['selection_tie_differences'].append({'group':group,'fold':fold,'seed':seed,'implemented':raw,'tie_correct':std})
   else:compare('selection_metric_replay',np.abs(future-d['y'].numpy()).mean(),ck['best_metric'],tol=2e-6)
   for role in ('calibration','validation'):
    z=np.load(O/f'evaluation/predictions/{group}/p{fold}_s{seed}/{role}.npz');d=roles[role];chk('prediction_endpoint_labels',np.array_equal(z['t'],d['t'].numpy()) and np.array_equal(z['stage'],d['stage'].numpy()) and np.array_equal(z['episode_id'],d['episode_id']));store[(group,fold,seed,role)]={k:z[k] for k in z.files};store[(group,fold,seed,role)]['y']=d['y'].numpy();store[(group,fold,seed,role)]['pc']=d['x'][:,-1,192:].numpy();store[(group,fold,seed,role)]['yc']=d['y_current'].numpy()
    if group in ('S','J'):
     rr=[r for r in metrics if r['group']==group and int(r['fold'])==fold and int(r['seed'])==seed and r['role']==role];chk('all31_workpoints',len(rr)==31 and len({r['policy'] for r in rr})==31)
     rank=pauc(z['stage'],z['score']);compare('rank_pauc',rank,rr[0]['pAUC']);m=z['stage']!=1;compare('rank_AP',average_precision_score(z['stage'][m]==2,z['score'][m]),rr[0]['AP'])
     tt=[r for r in trials if r['group']==group and int(r['fold'])==fold and int(r['seed'])==seed and r['role']==role]
     for ep in np.unique(z['episode_id']):
      ids=np.flatnonzero(z['episode_id']==ep);ids=ids[np.argsort(z['t'][ids])]
      for r in [r for r in tt if r['episode']==ep]:
       stats=epstats(z['t'][ids],z['stage'][ids],z['score'][ids],float(r['threshold']),int(r['k']))
       for k,v in stats.items():compare('alarm_event_'+k,v,r[k],tol=0)
    if group in ('F','J'):
     y=d['y'].numpy();pred=z['future'];pc=d['x'][:,-1,192:].numpy();yc=d['y_current'].numpy();rr=[r for r in fmet if r['group']==group and r['predictor']==group and int(r['fold'])==fold and int(r['seed'])==seed and r['role']==role and r['stratum']=='all'];
     for r in rr:
      h=(1,5,10).index(int(r['horizon']));a=('x','y','z').index(r['axis']);e=pred[:,h,a]-y[:,h,a];anc=pc[:,a]-yc[:,a];delta=(pred[:,h,a]-pc[:,a])-(y[:,h,a]-yc[:,a]);compare('future_mae',np.abs(e).mean(),r['future_mae']);compare('future_rmse',np.sqrt((e*e).mean()),r['future_rmse']);compare('future_change',np.abs(delta).mean(),r['deployed_change_mae']);compare('future_cross_identity',np.mean(e**2),np.mean(anc**2)+np.mean(delta**2)+np.mean(2*anc*delta),tol=2e-4)
   res['runs'].append({'group':group,'fold':fold,'seed':seed,'best_epoch':best});del initial
  print('verified',fold,seed,file=sys.stderr,flush=True)
# shared complete-group CI independently from raw validation arrays; check every draw for linear metrics, first 25 for ranking.
for fold in range(1,5):
 groups=np.unique(store[('S',fold,mt.SEEDS[0],'validation')]['leakage_group']);rng=np.random.default_rng(170000+fold);lookup={(int(r['draw']),r['metric']):r['difference'] for r in draws if int(r['fold'])==fold}
 for di in range(2000):
  counts=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups));wm=dict(zip(groups,counts));vals=defaultdict(list)
  for seed in mt.SEEDS:
   for group in mt.GROUPS:
    z=store[(group,fold,seed,'validation')];w=np.array([wm[g] for g in z['leakage_group']]);
    if group in ('S','J'):
     th=float(next(r['threshold'] for r in ths if r['group']==group and int(r['fold'])==fold and int(r['seed'])==seed and r['policy']=='FPR0.05'));static=z['stage']==0;gross=z['stage']==2;alarm=z['score']>=th
     vals[group+'_fpr'].append(np.sum(w*static*alarm)/np.sum(w*static) if np.sum(w*static) else np.nan);vals[group+'_recall'].append(np.sum(w*gross*alarm)/np.sum(w*gross) if np.sum(w*gross) else np.nan)
     if di<25:vals[group+'_pauc'].append(pauc(z['stage'],z['score'],w))
    if group in ('F','J'):vals[group+'_future'].append(np.average(np.abs(z['future']-z['y']).mean((1,2)),weights=w))
  for metric,ka,kb in [('J_minus_S_FPR_at_cal_FPR05','J_fpr','S_fpr'),('J_minus_S_recall_at_cal_FPR05','J_recall','S_recall'),('J_minus_F_future_MAE','J_future','F_future')]+([('J_minus_S_pAUC','J_pauc','S_pauc')] if di<25 else []):
   a=np.mean(vals[ka])-np.mean(vals[kb]);b=lookup[(di,metric)];compare('paired_CI_'+metric,a,float(b) if b else np.nan,tol=2e-6)
res['max_absolute_errors']=dict(maxerr);res['status']='pass' if not res['errors'] else 'fail';print(json.dumps(res,indent=2))
