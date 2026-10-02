import sys,json,csv,hashlib
from pathlib import Path
import numpy as np,torch
C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round17_htt_shared_temporal_multitask');O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round17_htt_shared_temporal_multitask');sys.path.insert(0,str(C));import multitask_train as mt
torch.set_num_threads(4);rows=list(csv.DictReader(open(O/'evaluation/slip/thresholds.csv')));out={'thresholds_checked':0,'threshold_mismatches':[],'recovery':{},'fit_assets':[]}
for fold in range(1,5):
 for seed in mt.SEEDS:
  d=torch.load(O/f'prepared/p{fold}_s{seed}/prepared.pt',map_location='cpu',weights_only=False);norm,w,table,joint=mt.fit_assets(d['roles'],seed);out['fit_assets'].append({'fold':fold,'seed':seed,'normalizers_exact':all(torch.equal(v,d['normalizer'][k]) for k,v in norm.items()),'weights_exact':torch.equal(w,d['roles']['fit']['slip_weight']),'lambda_exact':joint==d['joint_weight']})
  for g in ('S','J'):
   z=np.load(O/f'evaluation/predictions/{g}/p{fold}_s{seed}/calibration.npz');rr=[r for r in rows if r['group']==g and int(r['fold'])==fold and int(r['seed'])==seed];pre={}
   for k in (1,2,4):
    es=[]
    for ep in np.unique(z['episode_id']):
     ids=np.flatnonzero(z['episode_id']==ep);ids=ids[np.argsort(z['t'][ids])];t=z['t'][ids];s=z['score'][ids];q=np.full(len(ids),-np.inf)
     for i in range(k-1,len(ids)):
      if k==1 or np.all(np.diff(t[i-k+1:i+1])==1):q[i]=min(s[i-k+1:i+1])
     es.append((q,z['stage'][ids]))
    ts=np.unique(np.r_[np.nextafter(1.,np.inf),np.concatenate([q[(st!=1)&np.isfinite(q)] for q,st in es])]);sn=np.array([(st==0).sum() for q,st in es]);gn=np.array([(st==2).sum() for q,st in es]);fp=np.array([np.sum(q[st==0,None]>=ts[None,:],axis=0) for q,st in es]);tp=np.array([np.sum(q[st==2,None]>=ts[None,:],axis=0) for q,st in es]);frame=fp.sum(0)/sn.sum();rec=tp.sum(0)/gn.sum();macro=(fp[sn>0]/sn[sn>0,None]).mean(0);anyr=(fp[sn>0]>0).mean(0);pre[k]=(ts,frame,rec,macro,anyr)
   raw={}
   for r in rr:
    policy=r['policy'];k=int(r['k']);family=r['family'];ts,frame,rec,macro,anyr=pre[k]
    if policy=='fixed_0.5':th=.5
    elif policy=='maxBA':i=max(range(len(ts)),key=lambda i:((1-frame[i]+rec[i])/2,-frame[i],ts[i]));th=ts[i]
    elif family=='historical':alpha=float(policy[3:]);ii=np.flatnonzero(frame<=alpha+1e-15);i=max(ii,key=lambda i:(rec[i],-frame[i],ts[i]));th=ts[i];raw[policy]=th
    elif family=='confirm4_reference':th=raw[policy.split('|')[0]]
    else:
     arr=macro if family=='trial_macro_static_FPR' else anyr;ii=np.flatnonzero(arr<=float(r['alpha'])+1e-15);i=max(ii,key=lambda i:(rec[i],-arr[i],ts[i]));th=ts[i]
    out['thresholds_checked']+=1
    if th!=float(r['threshold']):out['threshold_mismatches'].append({'fold':fold,'seed':seed,'g':g,'policy':policy,'expected':th,'actual':r['threshold']})
def eq(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
base=O/'recovery_real_v3'
for kind in ('latest.pth','best.pth'):
 a=torch.load(base/'full'/kind,map_location='cpu',weights_only=False);b=torch.load(base/'resumed'/kind,map_location='cpu',weights_only=False);out['recovery'][kind]={k:eq(a[k],b[k]) for k in a};out['recovery'][kind]['sha_full']=hashlib.sha256((base/'full'/kind).read_bytes()).hexdigest();out['recovery'][kind]['sha_resumed']=hashlib.sha256((base/'resumed'/kind).read_bytes()).hexdigest()
out['parameter_counts']={g:sum(p.numel() for n in mt.active_names(g) for p in getattr(mt.init_model(20260914),n).parameters()) for g in mt.GROUPS};print(json.dumps(out,indent=2))
