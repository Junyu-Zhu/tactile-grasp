import os
os.environ['XFORMERS_DISABLED']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import time,json,importlib.util,sys
from pathlib import Path
import torch
here=Path(__file__).resolve().parent
rr=Path('/vla1/zjy/sparsh_runs/force_slip_htt_normalflow')
torch.set_num_threads(4)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
r18=load('root_r18',here.parent/'round18_htt_force_conditioned_film/train.py')
r20=load('root_r20',here.parent/'round20_htt_contact_state_transition/train.py')
d=torch.load(rr/'round17_htt_shared_temporal_multitask/prepared/p1_s20260914/prepared.pt',map_location='cpu',weights_only=False)
r=d['roles']['fit'];sel=d['roles']['selection'];mask=(r['stage']==0)|(r['stage']==2)
x=r['x'][mask].float();mu=x.mean((0,1));sd=x.std((0,1),unbiased=False).clamp_min(1e-6);x=(x-mu)/sd;xs=(sel['x'].float()-mu)/sd;y=(r['stage'][mask]==2).float()
out=[]
for group in ('V0','C0','M0','K-current'):
 torch.manual_seed(20260914)
 model=(r20.init_model(group,20260914) if group=='K-current' else r18.init_model(group,20260914)).cuda()
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 if group=='K-current':
  f=torch.load(rr/'round14_htt_future_force_dual/prepared/p1_s20260914/prepared.pt',map_location='cpu',weights_only=False);nf=r20.normalizer(f['roles']['fit']);xx=r20.norm_x(f['roles']['fit']['x'],nf);yy=r20.true_change(f['roles']['fit'])/nf['delta_scale'];ss=r20.norm_x(f['roles']['selection']['x'],nf)
 else:xx,yy,ss=x,y,xs
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();model.train();losses=[]
 for ids in torch.arange(len(xx)).split(256):
  opt.zero_grad(set_to_none=True);pred=model(xx[ids].cuda());target=yy[ids].cuda();loss=torch.nn.functional.smooth_l1_loss(pred,target) if group=='K-current' else torch.nn.functional.binary_cross_entropy_with_logits(pred,target)
  assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()))
 model.eval()
 with torch.inference_mode():
  for z in ss.split(256):assert torch.isfinite(model(z.cuda())).all()
 torch.cuda.synchronize();row={'group':group,'fit_count':len(xx),'selection_count':len(ss),'train_plus_selection_seconds':time.perf_counter()-start,'max_memory':torch.cuda.max_memory_allocated(),'finite':True};out.append(row);print(json.dumps(row),flush=True)
 del model,opt
 torch.cuda.empty_cache()
res={'scope':'budget_only_one_epoch_no_formal_weights','gpu':'physical GPU0','rows':out,'limitations':['old endpoint support','BCE unweighted timing proxy only','no checkpoint IO; add separate allowance','not semantic smoke or formal result']}
(here/'root_frozen_benchmark.json').write_text(json.dumps(res,indent=2)+'\n')
