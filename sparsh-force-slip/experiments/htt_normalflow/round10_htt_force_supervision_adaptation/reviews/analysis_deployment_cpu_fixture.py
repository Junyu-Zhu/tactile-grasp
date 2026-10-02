import sys,json,tempfile,csv,subprocess,importlib.util
from pathlib import Path
import torch,numpy as np
C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round10_htt_force_supervision_adaptation')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
tr=load('reviewtrain',C/'fusion/train.py');be=load('reviewbench',C/'deployment/benchmark.py');ev=load('revieweval',C/'evaluation/evaluate.py')
x=torch.zeros(2,9,199);x[:,:,:192]=torch.randn(2,9,192);x[:,:,192:195]=torch.arange(9)[None,:,None];x[:,5:,195:198]=5;x[:,5:,198]=1
norm={'mean':torch.zeros(198),'std':torch.ones(198)}
lag=x.clone();lag[:,1:,192:195]=x[:,:-1,192:195];lag[:,5:,195:198]=lag[:,5:,192:195]-lag[:,:4,192:195];lag=tr.normalize(lag,norm)
assert torch.equal(lag[0,:,192],torch.tensor([0,0,1,2,3,4,5,6,7])) and torch.equal(lag[0,5:,195],torch.tensor([4,5,5,5]))
assert torch.equal(lag[:,:,:192],x[:,:,:192]) and torch.equal(lag[:,:,198],x[:,:,198]);zero=x.clone();zero[:,:,192:198]=0
model=tr.TemporalHead('V_temporal').eval().requires_grad_(False)
with torch.inference_mode():assert torch.equal(model(x),model(lag)) and torch.equal(model(x),model(zero))
bases=[(x[:1,i,:192],x[:1,i,192:195]) for i in range(9)];raw,out=be.assemble(bases,norm,'F_history');assert torch.equal(raw,x[:1]);assert torch.equal(out,x[:1])
with tempfile.TemporaryDirectory(prefix='r10-review-') as td:
 p=Path(td);ce=p/'current';fo=p/'force';o=p/'out';ce.mkdir();fo.mkdir();trials=[];metrics=[];fr=[]
 for f in range(1,5):
  fold=f'htt_leave_p{f}'
  for seed in ev.SEEDS:
   metrics.append(dict(fold=fold,seed=seed,role='calibration',point='FPR0.05',rule='raw',group='F_history_new',static_fpr=.05))
   for t in range(4):
    eid=f'trial{t}';n=0 if t==3 else 10
    for group,fp,fn in [('V_temporal',t,1),('F_history_old',t+2,1),('F_history_new',t+1,t)]:
     trials.append(dict(fold=fold,seed=seed,episode=eid,group=group,role='validation',point='FPR0.05',rule='raw',fp=fp if n else 0,tn=n-fp if n else 0,fn=fn,tp=10-fn,false_starts=1))
    for variant in ('old','new'):
     for axis in ('shear_x','shear_y','normal'):fr.append(dict(fold=fold,seed=seed,episode_id=eid,variant=variant,axis=axis,role='validation',population='all',task='slip_force',mae=(t+2 if variant=='old' else t+1)))
 ev.csvout(ce/'trials.csv',trials);ev.csvout(ce/'metrics.csv',metrics);ev.js(ce/'summary.json',dict(status='complete',new_runs=12,synthetic=False,output_hashes={q.name:ev.sha(q) for q in ce.glob('*.csv')}));ev.csvout(fo/'per_trial_axis.csv',fr);ev.js(fo/'AUDIT.json',dict(status='pass',test_consumed=False,output_hashes={'per_trial_axis.csv':ev.sha(fo/'per_trial_axis.csv')}))
 subprocess.run([sys.executable,str(C/'analysis/associate.py'),'--force-trials',str(fo),'--current-evaluation',str(ce),'--output',str(o)],check=True)
 avg=ev.readcsv(o/'seedmean_trial.csv');assert len(avg)==4*4*4 and all(float(r['delta_MAE'])==-1 for r in avg)
 assert all(abs(float(r['delta_static_fpr'])+.1)<1e-12 for r in avg if r['episode']!='trial3')
 ex=ev.readcsv(o/'exclusions.csv');assert len(ex)==12 and all(r['reason']=='no_static' for r in ex)
 corr=ev.readcsv(o/'correlations.csv');assert all(int(r['n_trials'])<=4 for r in corr)
print(json.dumps({'status':'pass','CPU_only':True,'checks':['causal lag1 exact index/delta','visual and valid unchanged','V bitwise invariant','deployment ninebase exact assembly','association seedmean and new-old signs','missing static retained undefined/excluded','within-fold complete trial counts']}))
