import sys,json,importlib.util
from pathlib import Path
import numpy as np,torch
C=Path('/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round17_htt_shared_temporal_multitask');O=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round17_htt_shared_temporal_multitask');sys.path.insert(0,str(C));import multitask_train as mt
torch.set_num_threads(4);out={'endpoint_targets':[],'mask':{},'gap':{}};total=0
for fold in range(1,5):
 for seed in mt.SEEDS:
  d=torch.load(O/f'prepared/p{fold}_s{seed}/prepared.pt',map_location='cpu',weights_only=False);supp=json.load(open(d['provenance']['support']));entries={e['episode_id']:e for e in supp['entries']};ok=True
  for role,v in d['roles'].items():
   for ep in set(v['episode_id']):
    ids=np.array([i for i,e in enumerate(v['episode_id']) if e==ep]);t=v['t'][ids].numpy();arr=np.load(entries[ep]['force_native_n_path']);ok &= (t.min()>=13 and (t+10).max()<len(arr) and np.array_equal(v['y'][ids].numpy(),arr[t[:,None]+np.array([1,5,10])].astype(np.float32)) and np.array_equal(v['y_current'][ids].numpy(),arr[t].astype(np.float32)));total+=len(t)
  out['endpoint_targets'].append({'fold':fold,'seed':seed,'same_episode_horizons_current_exact':bool(ok)})
# Gradient-only checks using existing fit endpoints; no optimizer step, no checkpoint or training.
v=d['roles']['fit'];n=d['normalizer'];ids=torch.arange(32);x=(v['x'][ids]-n['x_mean'])/n['x_std'];y=(v['y'][ids]-n['y_mean'])/n['y_std'];stage=v['stage'][ids];w=v['slip_weight'][ids];m=mt.init_model(20260916);ls,lf,logit,_=mt.loss_parts(m,x,y,stage,w);mask=(stage==0)|(stage==2)
manual=(torch.nn.functional.binary_cross_entropy_with_logits(logit[mask],(stage[mask]==2).float(),reduction='none')*w[mask]).sum()/mask.sum();out['mask']['denominator_exact']=torch.equal(ls,manual)
for name,l in [('slip',ls),('future',lf)]:
 grad=torch.autograd.grad(l,tuple(m.gru.parameters()),retain_graph=True);out['mask'][name+'_backbone_finite_nonzero']=all(torch.isfinite(g).all().item() for g in grad) and sum(float(g.abs().sum()) for g in grad)>0
ls,lf,_,_=mt.loss_parts(m,x,y,torch.ones_like(stage),torch.zeros_like(w));grad=torch.autograd.grad(lf,tuple(m.slip.parameters()),allow_unused=True);out['mask']['incipient_slip_loss_absent']=ls is None;out['mask']['future_no_slip_head_gradient']=all(g is None for g in grad)
spec=importlib.util.spec_from_file_location('r13',C.parent/'round13_trial_level_alarm_calibration/evaluation/r13_evaluate.py');r=importlib.util.module_from_spec(spec);sys.modules['r13']=r;spec.loader.exec_module(r)
e=r.Episode('synthetic_gap','g','p',np.array([13,14,18,19,20,21]),np.array([2,2,2,0,2,2]),np.array([.9,.9,.9,.9,.9,.1]));alarm,starts=r.state_alarm(e,.5,2);s=r.episode_stats(e,.5,2);out['gap']={'alarm':alarm.tolist(),'expected':[False,True,False,True,True,False],'pass':alarm.tolist()==[False,True,False,True,True,False] and s['left_censored_events']==2 and s['right_boundary_events']==2 and s['preexisting_alarm_events']==1 and s['delay_sum']==0,'stats':s};out['checked_endpoints']=total;print(json.dumps(out,indent=2))
