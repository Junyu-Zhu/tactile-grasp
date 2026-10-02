#!/usr/bin/env python3
"""Frozen-input HTT temporal-head adaptation; no upstream model in optimizer."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('XFORMERS_DISABLED', '1')
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'protocol.json'
ROLES=('train','validation','calibration')
GROUPS=('V_balanced','F_history_balanced','F_residual_balanced')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(t,p)

def atomic_save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);os.replace(t,p)

def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def configure(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.set_num_threads(4)

class TemporalHead(nn.Module):
 def __init__(self,group):
  super().__init__()
  if group not in GROUPS:raise ValueError(group)
  self.group=group
  # Common modules are created first, giving identical same-seed GRU/risk initialization.
  self.gru=nn.GRU(128,128,batch_first=True)
  self.risk=nn.Linear(128,1)
  if group in ('V_balanced','F_residual_balanced'):
   self.visual=nn.Sequential(nn.Linear(193,128),nn.GELU())
   self.interaction=nn.Sequential(nn.Linear(128,74),nn.GELU(),nn.Linear(74,128))
  else:
   self.visual=nn.Sequential(nn.Linear(192,96),nn.GELU())
   self.force=nn.Sequential(nn.Linear(7,32),nn.GELU())
   self.interaction=nn.Sequential(nn.Linear(128,98),nn.GELU(),nn.Linear(98,128))
  if group=='F_residual_balanced':
   self.residual_bound=2.0
   self.force_gru=nn.GRU(4,16,batch_first=True)
   self.force_correction=nn.Sequential(nn.Linear(144,32),nn.GELU(),nn.Linear(32,1))

 def components(self,x):
  if x.ndim!=3 or x.shape[1:]!=(9,199):raise ValueError('expected [N,9,199]')
  if self.group in ('V_balanced','F_residual_balanced'):
   v=self.visual(torch.cat([x[:,:,:192],x[:,:,198:199]],-1))
   hidden=self.gru(v+torch.tanh(self.interaction(v)))[0][:,-1]
   base=self.risk(hidden).squeeze(-1)
   if self.group=='V_balanced':return {'base_logit':base,'residual_logit':torch.zeros_like(base),'logit':base}
   f=self.force_gru(torch.cat([x[:,:,192:195],x[:,:,198:199]],-1))[0][:,-1]
   raw=self.force_correction(torch.cat([hidden,f],-1)).squeeze(-1)
   residual=self.residual_bound*torch.tanh(raw)
   return {'base_logit':base,'residual_logit':residual,'logit':base+residual}
  condition=torch.cat([x[:,:,192:195],torch.zeros_like(x[:,:,195:198]),x[:,:,198:]],-1)
  v=self.visual(x[:,:,:192]);f=self.force(condition);joined=torch.cat([v,f],-1)
  fused=joined+torch.sigmoid(f.mean(-1,keepdim=True))*torch.tanh(self.interaction(joined))
  logit=self.risk(self.gru(fused)[0][:,-1]).squeeze(-1)
  return {'base_logit':logit,'residual_logit':torch.zeros_like(logit),'logit':logit}

 def forward(self,x):
  return self.components(x)['logit']


def trial_weights(role):
 stage=role['stage'];ids=torch.nonzero((stage==0)|(stage==2)).flatten()
 if not len(ids):raise ValueError('no primary train')
 by={}
 for i in ids.tolist():by.setdefault(role['episode_id'][i],{}).setdefault(int(stage[i]),[]).append(i)
 n=len(ids);j=len(by);w=torch.zeros(len(stage),dtype=torch.float64);records=[]
 for eid,classes in sorted(by.items()):
  for c,ix in sorted(classes.items()):
   value=n/(j*len(classes)*len(ix));w[ix]=value
   records.append({'episode_id':eid,'stage':c,'n':len(ix),'class_count':len(classes),'frame_weight':value,'total_weight':value*len(ix),'trial_total_weight':n/j})
 if not torch.allclose(w.sum(),torch.tensor(float(n),dtype=torch.float64)):raise ValueError('weight normalization')
 return w.float(),records

def fit_normalizer(role):
 primary=(role['stage']==0)|(role['stage']==2)
 x=role['x'][primary].double();mean=x[:,:,:198].reshape(-1,198).mean(0)
 std=x[:,:,:198].reshape(-1,198).std(0,unbiased=False)
 delta=x[:,5:,195:198].reshape(-1,3)
 mean[195:198]=delta.mean(0);std[195:198]=delta.std(0,unbiased=False)
 return {'mean':mean.float(),'std':std.clamp_min(1e-6).float()}

def normalize(x,norm):
 y=x.clone();y[:,:,:198]=(y[:,:,:198]-norm['mean'])/norm['std'];y[:,:5,195:198]=0
 if not torch.isfinite(y).all():raise ValueError('nonfinite normalized input')
 return y

def low_fpr_auc(y,p):
 y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.float64)
 if set(y.tolist())!={0,1} or not np.isfinite(p).all():raise ValueError('selection needs both finite classes')
 order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
 ends=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]
 tp=np.cumsum(ys)[ends]/y.sum();fp=np.cumsum(1-ys)[ends]/(len(y)-y.sum())
 fp=np.r_[0.,fp];tp=np.r_[0.,tp];limit=.1;keep=fp<limit
 xs=np.r_[fp[keep],limit];ys=np.r_[tp[keep],np.interp(limit,fp,tp)]
 return float(np.trapz(ys,xs)/limit)

def validate_data(data,audit,path,expected_fold,expected_seed):
 if data.get('schema')!='round9_htt_temporal_prepared_v1':raise ValueError('prepared schema')
 if data['fold']!=expected_fold or data['seed']!=expected_seed:raise ValueError('prepared fold/seed identity')
 if audit.get('status')!='pass' or audit['output_hashes'].get(str(path.resolve()))!=sha(path):raise ValueError('prepared audit/hash')
 for k in ('frozen_visual','no_force_visual_path','role_group_disjoint','gt_force_not_input'):
  if data['upstream_audit'].get(k) is not True:raise ValueError('missing upstream check '+k)
 for name in ROLES:
  r=data['roles'][name];x=r['x'];n=len(x)
  if x.dtype!=torch.float32 or x.shape!=(n,9,199) or not torch.isfinite(x).all():raise ValueError('input schema')
  if any(len(r[k])!=n for k in ('stage','t','episode_id','leakage_group')):raise ValueError('identity length')
  if not torch.isin(r['stage'],torch.tensor([0,1,2])).all() or (r['t']<13).any():raise ValueError('stage/history')
  if x[:,:5,195:199].count_nonzero() or not torch.all(x[:,5:,198]==1):raise ValueError('aux validity')
  if not torch.allclose(x[:,5:,195:198],x[:,5:,192:195]-x[:,:4,192:195],atol=1e-6,rtol=1e-6):raise ValueError('delta construction')
  keys=list(zip(r['episode_id'],r['t'].tolist()))
  if len(set(keys))!=n:raise ValueError('duplicate endpoints')
  if not all(set(r['leakage_group']).isdisjoint(data['roles'][other]['leakage_group']) for other in ROLES if other!=name):raise ValueError('role leakage')

def infer(model,x,device,batch=256):
 model.eval();pred=[]
 with torch.inference_mode():
  for lo in range(0,len(x),batch):pred.append(torch.sigmoid(model(x[lo:lo+batch].to(device))).cpu())
 return torch.cat(pred)

def train(a):
 configure(a.seed);protocol=json.loads(PROTOCOL.read_text());out=a.output;out.mkdir(parents=True,exist_ok=True)
 if not a.smoke and not a.execute_formal:raise ValueError('formal execution requires --execute-formal')
 data=torch.load(a.data,map_location='cpu',weights_only=False);audit=json.loads(a.data.with_name('audit.json').read_text())
 validate_data(data,audit,a.data,a.fold,a.seed)
 if data.get('experiment')!='round10_new_force' or not data.get('force_replacement_audit',{}).get('pass'):raise ValueError('requires accepted new-force prepared input')
 for proof in data['force_replacement_audit']['proofs']:
  if sha(proof['path'])!=proof['sha256']:raise ValueError('upstream replacement identity changed')
 if not a.smoke and data['force_replacement_audit'].get('smoke'):raise ValueError('formal cannot use smoke force')
 if a.epochs is not None and not a.smoke:raise ValueError('epoch override is smoke only')
 max_epochs=a.epochs or protocol['max_epochs'];batch=protocol['batch_size']
 config={'schema':'round11_htt_fusion_run_v1','group':a.group,'fold':a.fold,'seed':a.seed,'smoke':a.smoke,'max_epochs':max_epochs,'batch_size':batch,'prepared':{'path':str(a.data.resolve()),'sha256':sha(a.data)},'prepared_audit':{'path':str(a.data.with_name('audit.json').resolve()),'sha256':sha(a.data.with_name('audit.json'))},'sources':{str(PROTOCOL):sha(PROTOCOL),str(Path(__file__).resolve()):sha(__file__)},'protocol':protocol}
 cp=out/'config.json'
 if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('existing config identity mismatch')
 atomic_json(cp,config)
 if (out/'summary.json').exists():
  summary=json.loads((out/'summary.json').read_text())
  if summary.get('status')=='complete':
   for name,h in summary['output_hashes'].items():
    if sha(out/name)!=h:raise ValueError('accepted artifact changed')
   print(json.dumps({'status':'reused','output':str(out)}));return
 norm=fit_normalizer(data['roles']['train']);xs={k:normalize(data['roles'][k]['x'],norm) for k in ROLES}
 stage=data['roles']['train']['stage'];ids=torch.nonzero((stage==0)|(stage==2)).flatten()
 ys=(stage==2).float();counts=torch.bincount(ys[ids].long(),minlength=2)
 if (counts==0).any():raise ValueError('training lacks a class')
 weights,weight_records=trial_weights(data['roles']['train']);atomic_json(out/'TRAIN_WEIGHTS.json',{'schema':'round11_trial_class_weights_v1','primary_frames':len(ids),'records':weight_records,'total_weight':float(weights.double().sum()),'train_only':True});model=TemporalHead(a.group).to(a.device)
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};init_hash=state_hash(initial)
 optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['optimizer']['lr'],weight_decay=protocol['optimizer']['weight_decay'])
 history=[];best=-1.;best_epoch=0;start=1;wait=0;gradient_audit={}
 latest=out/'latest.pth'
 if latest.exists():
  old=torch.load(latest,map_location='cpu',weights_only=False)
  if old['config']!=config or old['initial_state_sha256']!=init_hash:raise ValueError('resume identity')
  model.load_state_dict(old['model_state']);optimizer.load_state_dict(old['optimizer_state'])
  for state in optimizer.state.values():
   for k,v in state.items():
    if torch.is_tensor(v):state[k]=v.to(a.device)
  history=old['history'];best=old['best_score'];best_epoch=old['best_epoch'];wait=old['wait'];start=old['epoch']+1;gradient_audit=old['gradient_audit']
 began=time.time()
 for epoch in range(start,max_epochs+1):
  if wait>=protocol['patience']:break
  model.train();perm=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(a.seed+epoch))]
  losses=[]
  for lo in range(0,len(perm),batch):
   ix=perm[lo:lo+batch];x=xs['train'][ix].to(a.device);y=ys[ix].to(a.device)
   check=not gradient_audit
   if check:x.requires_grad_(True)
   optimizer.zero_grad(set_to_none=True);logits=model(x)
   loss=(nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights[ix].to(a.device)).mean()
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
   loss.backward()
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('missing/nonfinite gradient')
   if check:
    g=x.grad.detach();gradient_audit={'all_gradients_finite':True,'visual_gradient_nonzero':bool(g[:,:,:192].count_nonzero()),'force_gradient_absmax':float(g[:,:,192:195].abs().max()),'delta_gradient_absmax':float(g[:,:,195:198].abs().max()),'valid_gradient_absmax':float(g[:,:,198].abs().max())}
    if not gradient_audit['visual_gradient_nonzero']:raise ValueError('visual disconnected')
    if a.group=='V_balanced' and (gradient_audit['force_gradient_absmax']!=0 or gradient_audit['delta_gradient_absmax']!=0):raise ValueError('force leakage into V')
    if a.group in ('F_history_balanced','F_residual_balanced') and gradient_audit['delta_gradient_absmax']!=0:raise ValueError('delta leaked into F_history')
    if a.group!='V_balanced' and gradient_audit['force_gradient_absmax']==0:raise ValueError('force disconnected')
    if a.group=='F_residual_balanced' and not hasattr(model,'residual_bound'):raise ValueError('missing residual bound')
   optimizer.step();losses.append(float(loss.detach()))
  val=infer(model,xs['validation'],a.device);s=data['roles']['validation']['stage'];mask=(s==0)|(s==2)
  score=low_fpr_auc((s[mask]==2).numpy(),val[mask].numpy());improved=score>best
  if improved:best=score;best_epoch=epoch;wait=0
  else:wait+=1
  history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_low_fpr_auc':score})
  state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  checkpoint={'config':config,'model_state':state,'optimizer_state':optimizer.state_dict(),'normalizer':norm,'epoch':epoch,'best_score':best,'best_epoch':best_epoch,'wait':wait,'history':history,'initial_state_sha256':init_hash,'gradient_audit':gradient_audit}
  atomic_save(latest,checkpoint)
  if improved:atomic_save(out/'best.pth',checkpoint)
  print(json.dumps({'epoch':epoch,'loss':history[-1]['train_loss'],'validation_low_fpr_auc':score,'best_epoch':best_epoch}),flush=True)
  if a.stop_after_epochs and epoch>=a.stop_after_epochs:
   print(json.dumps({'status':'interrupted_for_recovery_proof','epoch':epoch}));return
 best_cp=torch.load(out/'best.pth',map_location='cpu',weights_only=False);model.load_state_dict(best_cp['model_state']);model.eval()
 roundtrip=TemporalHead(a.group).to(a.device);roundtrip.load_state_dict(best_cp['model_state']);roundtrip.eval()
 probe=xs['validation'][:16].to(a.device)
 with torch.inference_mode():
  if not torch.equal(model(probe),roundtrip(probe)):raise ValueError('checkpoint roundtrip failed')
 parameter_changes={k:not torch.equal(initial[k],v.cpu()) for k,v in model.state_dict().items()}
 if not all(parameter_changes.values()):raise ValueError('a trainable parameter tensor did not update')
 diagnostics={};outputs={}
 for role in ROLES:
  r=data['roles'][role];pred=infer(model,xs[role],a.device);prob=pred.numpy()
  if not np.isfinite(prob).all() or prob.min()<0 or prob.max()>1:raise ValueError('invalid probability')
  if len(np.unique(prob))<2:raise ValueError('constant probability collapse')
  diagnostics[role]={'std':float(prob.std()),'mean':float(prob.mean()),'unique':len(np.unique(prob))}
  target=out/f'predictions_{role}.csv';tmp=target.with_suffix('.csv.tmp')
  with tmp.open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['episode_id','t','stage','p_slip','role','fold','seed','group','leakage_group'])
   w.writerows(zip(r['episode_id'],r['t'].tolist(),r['stage'].tolist(),prob.tolist(),[role]*len(prob),[a.fold]*len(prob),[a.seed]*len(prob),[a.group]*len(prob),r['leakage_group']))
  os.replace(tmp,target);outputs[target.name]=sha(target)
 for name in ['best.pth','latest.pth','config.json','TRAIN_WEIGHTS.json']:outputs[name]=sha(out/name)
 if sha(a.data)!=config['prepared']['sha256']:raise ValueError('frozen input changed')
 total=sum(p.numel() for p in model.parameters());inactive=96 if a.group=='F_history_balanced' else 0
 summary={'schema':'round11_htt_fusion_summary_v1','status':'complete','group':a.group,'fold':a.fold,'seed':a.seed,'smoke':a.smoke,'config_sha256':sha(cp),'best_epoch':best_epoch,'best_score':best,'epochs':len(history),'history':history,'normalizer_fit_role':'train_static_gross','loss_weighting':'per_trial_per_present_class_mean1','loss_weights_sha256':sha(out/'TRAIN_WEIGHTS.json'),'initial_state_sha256':init_hash,'parameters':total,'structurally_inactive_parameters':inactive,'effective_parameters':total-inactive,'parameter_tensors_updated':parameter_changes,'gradient_audit':gradient_audit,'checkpoint_roundtrip_exact':True,'upstream_readonly_by_cache_architecture':True,'prepared_unchanged':True,'optimizer_only_new_head':True,'probability_diagnostics':diagnostics,'elapsed_seconds_this_invocation':time.time()-began,'output_hashes':outputs}
 atomic_json(out/'summary.json',summary);print(json.dumps({'status':'complete','output':str(out),'best_epoch':best_epoch,'epochs':len(history)}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--group',choices=GROUPS,required=True);p.add_argument('--fold',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');p.add_argument('--execute-formal',action='store_true');p.add_argument('--epochs',type=int);p.add_argument('--stop-after-epochs',type=int);train(p.parse_args())
