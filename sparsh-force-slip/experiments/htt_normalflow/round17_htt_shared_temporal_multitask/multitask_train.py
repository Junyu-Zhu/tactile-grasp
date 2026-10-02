#!/usr/bin/env python3
"""Round-17 cache preparation and resumable S/F/J trainer."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch import nn

torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)

HORIZONS=(1,5,10); GROUPS=("S","F","J"); SEEDS=(20260914,20260915,20260916)

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def stable_json(x):return json.dumps(x,sort_keys=True,separators=(",",":"))
def atomic_json(x,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".",suffix=".tmp");os.close(fd)
 try:Path(tmp).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def atomic_save(x,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".",suffix=".tmp");os.close(fd)
 try:torch.save(x,tmp);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def state_hash(state,names=None):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):
  if names is None or any(k.startswith(q+".") for q in names):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

class SharedModel(nn.Module):
 def __init__(self):
  super().__init__();self.visual=nn.Linear(192,45);self.force=nn.Linear(3,3);self.gru=nn.GRU(48,48,batch_first=True);self.slip=nn.Linear(48,1);self.future=nn.Linear(48,9);self.act=nn.GELU()
 def encode(self,x):
  z=torch.cat((self.act(self.visual(x[...,:192])),self.act(self.force(x[...,192:195]))),-1)
  return self.gru(z)[0][:,-1]
 def forward(self,x):
  h=self.encode(x);return self.slip(h).squeeze(-1),self.future(h).reshape(-1,3,3)

MODULES=("visual","force","gru","slip","future")
def module_seed(seed,name):return int(hashlib.sha256(f"round17:{seed}:{name}".encode()).hexdigest()[:8],16)
def init_model(seed):
 m=SharedModel()
 for name in MODULES:
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(module_seed(seed,name))
   for p in getattr(m,name).parameters():
    nn.init.xavier_uniform_(p) if p.ndim>=2 else nn.init.zeros_(p)
 return m
def active_names(group):return {"S":("visual","force","gru","slip"),"F":("visual","force","gru","future"),"J":MODULES}[group]

def assert_roles(roles):
 seen={}
 for role,d in roles.items():
  if role=="test":raise ValueError("test role forbidden")
  groups=set(d["leakage_group"])
  for old,oldg in seen.items():
   if groups&oldg:raise ValueError(f"leakage overlap {role}/{old}")
  seen[role]=groups

def fit_assets(roles,seed):
 fit=roles["fit"];x=fit["x"].double();y=fit["y"].double()
 norm={"x_mean":x.mean((0,1)).float(),"x_std":x.std((0,1),unbiased=False).clamp_min(1e-6).float(),"y_mean":y.mean(0).float(),"y_std":y.std(0,unbiased=False).clamp_min(1e-6).float()}
 stage=fit["stage"].long();mask=stage.eq(0)|stage.eq(2);n=int(mask.sum());weights=torch.zeros(len(stage),dtype=torch.float64);table=[]
 if not bool((stage.eq(0)&mask).any()) or not bool((stage.eq(2)&mask).any()):raise ValueError("fit lacks static or gross")
 for c in (0,2):
  ids=torch.nonzero(stage.eq(c),as_tuple=False).flatten();eps=np.asarray(fit["episode_id"],object)[ids.numpy()];unique=sorted(set(eps.tolist()));jc=len(unique)
  for ep in unique:
   local=ids[torch.tensor(eps==ep)];nij=len(local);w=n/(2*jc*nij);weights[local]=w;table.append({"class":c,"episode":ep,"endpoints":nij,"trials_in_class":jc,"weight":w,"contribution":w*nij})
 sums={str(c):float(weights[stage.eq(c)].sum()) for c in (0,2)}
 if not all(math.isclose(v,n/2,rel_tol=1e-10,abs_tol=1e-8) for v in sums.values()):raise AssertionError(sums)
 model=init_model(seed).eval();ls_sum=lf_sum=0.;slip_n=future_n=0
 with torch.no_grad():
  for ids in torch.arange(len(x)).split(256):
   xx=(fit["x"][ids]-norm["x_mean"])/norm["x_std"];yy=(fit["y"][ids]-norm["y_mean"])/norm["y_std"]
   slog,fp=model(xx);sm=fit["stage"][ids].eq(0)|fit["stage"][ids].eq(2)
   if sm.any():
    target=fit["stage"][ids][sm].eq(2).float();ls_sum+=float((nn.functional.binary_cross_entropy_with_logits(slog[sm],target,reduction="none")*weights[ids][sm].float()).sum());slip_n+=int(sm.sum())
   lf_sum+=float(nn.functional.smooth_l1_loss(fp,yy,beta=1,reduction="sum"));future_n+=yy.numel()
 ls0=ls_sum/slip_n;lf0=lf_sum/future_n;lam=float(np.clip(ls0/lf0,.25,4.0));change=(fit["y"][:,2]-fit["y_current"]).abs().amax(1).numpy();q=np.quantile(change,[.25,.75],method="linear")
 stable=change<=q[0];changing=(change>=q[1])&(~stable);transition=~(stable|changing)
 return norm,weights.float(),table,{"slip_loss_initial":ls0,"future_loss_initial":lf0,"lambda_future":lam,"formula":"clip(Ls0/Lf0,0.25,4.0)","class_weight_sums":sums,"reliable_fit_endpoints":n,"fit_change_strata":{"definition":"max_axis_abs_GT_h10_minus_GT_current","quantile_method":"linear","tie_policy":"stable_first; changing excludes stable; transition remainder","stable_le_n":float(q[0]),"changing_ge_n":float(q[1]),"thresholds_equal":bool(q[0]==q[1]),"threshold_is_zero":bool(q[0]==0 or q[1]==0),"fit_endpoints":len(change),"fit_support":{"stable":int(stable.sum()),"transition":int(transition.sum()),"changing":int(changing.sum())}}}

def enrich_round14(r14_path,parent_path,output,fold,seed):
 if seed not in SEEDS:raise ValueError(seed)
 r14=torch.load(r14_path,map_location="cpu",weights_only=False);parent=torch.load(parent_path,map_location="cpu",weights_only=False)
 if sha(parent_path)!=r14["provenance"]["prepared_sha256"]:raise ValueError("parent prepared hash mismatch")
 for k,v in r14["provenance"]["immutable_upstream"].items():
  if sha(v["path"])!=v["sha256"]:raise ValueError(f"upstream mismatch {k}")
 maps={}
 for outer,d in parent["roles"].items():
  q={}
  for i,(e,t) in enumerate(zip(d["episode_id"],d["t"].tolist())):
   key=(e,int(t))
   if key in q:raise ValueError(f"duplicate endpoint {key}")
   q[key]=(int(d["stage"][i]),d["probe"][i])
  maps[outer]=q
 roles={}
 for role,d in r14["roles"].items():
  outer="train" if role in ("fit","selection") else role;st=[];probe=[]
  for e,t in zip(d["episode_id"],d["t"].tolist()):
   if (e,int(t)) not in maps[outer]:raise ValueError(f"unmatched {role} {e} {t}")
   a,b=maps[outer][(e,int(t))];st.append(a);probe.append(b)
  roles[role]={**d,"stage":torch.tensor(st,dtype=torch.long),"probe":probe}
 assert_roles(roles)
 norm,weights,table,joint=fit_assets(roles,seed);roles["fit"]["slip_weight"]=weights
 support={}
 for role,d in roles.items():
  support[role]={"endpoints":len(d["t"]),"episodes":len(set(d["episode_id"])),"leakage_groups":len(set(d["leakage_group"])),"stage_counts":{str(c):int(d["stage"].eq(c).sum()) for c in (0,1,2)},"unknown_stage_endpoints":int((~(d["stage"].eq(0)|d["stage"].eq(1)|d["stage"].eq(2))).sum()),"stage_episode_counts":{str(c):len(set(e for e,s in zip(d["episode_id"],d["stage"].tolist()) if s==c)) for c in (0,1,2)}}
 if min(support["selection"]["stage_counts"][str(c)] for c in (0,2))==0:raise ValueError("selection unsupported")
 data={"schema":"round17_common_cache_v1","fold":fold,"seed":seed,"horizons":HORIZONS,"roles":roles,"normalizer":norm,"class_trial_table":table,"joint_weight":joint,"support":support,"provenance":{"round14_prepared":str(Path(r14_path).resolve()),"round14_sha256":sha(r14_path),"parent_prepared":str(Path(parent_path).resolve()),"parent_sha256":sha(parent_path),"immutable_upstream":r14["provenance"]["immutable_upstream"],"support":r14["provenance"]["support"],"support_sha256":r14["provenance"]["support_sha256"],"optimizer_scope":"new_round17_modules_only; cached upstream immutable"}}
 atomic_save(data,output);return {"output":str(output),"sha256":sha(output),"fold":fold,"seed":seed,"support":support,"joint_weight":joint,"normalizer_sha256":hashlib.sha256(b"".join(norm[k].numpy().tobytes() for k in ("x_mean","x_std","y_mean","y_std"))).hexdigest()}

def slip_pauc(stage,score,max_fpr=.1):
 stage=np.asarray(stage);score=np.asarray(score);neg=stage==0;pos=stage==2
 if not neg.any() or not pos.any():return float("nan")
 order=np.argsort(-score,kind="stable");y=pos[order];isneg=neg[order];tp=np.r_[0,np.cumsum(y)];fp=np.r_[0,np.cumsum(isneg)];tpr=tp/pos.sum();fpr=fp/neg.sum();keep=fpr<=max_fpr
 xf=fpr[keep];yf=tpr[keep]
 if xf[-1]<max_fpr:
  j=np.flatnonzero(fpr>max_fpr)
  if len(j):
   j=j[0];i=j-1;frac=(max_fpr-fpr[i])/(fpr[j]-fpr[i]);xf=np.r_[xf,max_fpr];yf=np.r_[yf,tpr[i]+frac*(tpr[j]-tpr[i])]
  else:xf=np.r_[xf,max_fpr];yf=np.r_[yf,yf[-1]]
 return float(np.trapezoid(yf,xf)/max_fpr)

def loss_parts(model,x,y,stage,weight):
 slog,fp=model(x);sm=stage.eq(0)|stage.eq(2);ls=None
 if sm.any():ls=(nn.functional.binary_cross_entropy_with_logits(slog[sm],stage[sm].eq(2).float(),reduction="none")*weight[sm]).sum()/sm.sum()
 lf=nn.functional.smooth_l1_loss(fp,y,beta=1.0);return ls,lf,slog,fp

def train(data,out,group,fold,seed,device="cpu",max_epochs=60,patience=10,interrupt_after=None):
 if group not in GROUPS:raise ValueError(group)
 assert_roles(data["roles"]);out=Path(out);out.mkdir(parents=True,exist_ok=True);norm=data["normalizer"];fit=data["roles"]["fit"];sel=data["roles"]["selection"]
 def nx(d):(None)
 xfit=(fit["x"]-norm["x_mean"])/norm["x_std"];yfit=(fit["y"]-norm["y_mean"])/norm["y_std"]
 xsel=(sel["x"]-norm["x_mean"])/norm["x_std"]
 model=init_model(seed).to(device);names=active_names(group);params=[p for n in names for p in getattr(model,n).parameters()];opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-4);gen=torch.Generator().manual_seed(seed+1700)
 identity={"schema":"round17_run_identity_v1","group":group,"fold":fold,"seed":seed,"data_sha256":data["data_sha256"],"protocol_sha256":sha(Path(__file__).with_name("PROTOCOL.md")),"source_sha256":sha(__file__),"active_modules":list(names),"batch_size":256,"lr":1e-3,"weight_decay":1e-4,"max_epochs":max_epochs,"patience":patience,"selector":"future_mae" if group=="F" else "slip_pauc_0_0.1","lambda_future":data["joint_weight"]["lambda_future"]}
 latest=out/"latest.pth";best=out/"best.pth";start=0;best_metric=float("inf") if group=="F" else -float("inf");best_epoch=-1;wait=0;history=[];skipped=0;optimizer_steps=0;slip_seen=0;future_seen=0
 if latest.exists():
  ck=torch.load(latest,map_location="cpu",weights_only=False)
  if ck["identity"]!=identity:raise ValueError("resume identity mismatch")
  model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
  for st in opt.state.values():
   for k,v in st.items():
    if torch.is_tensor(v):st[k]=v.to(device)
  gen.set_state(ck["generator"]);random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"]);start=ck["epoch"]+1;best_metric=ck["best_metric"];best_epoch=ck["best_epoch"];wait=ck["wait"];history=ck["history"];skipped=ck["skipped_empty_slip_steps"];optimizer_steps=ck["optimizer_steps"];slip_seen=ck["slip_supervised_endpoints_seen"];future_seen=ck["future_supervised_endpoints_seen"]
 if wait>=patience or start>=max_epochs:
  result={"status":"complete","best_epoch":best_epoch,"best_metric":best_metric,"epochs":len(history),"resume_without_extra_epoch":True,"identity":identity};atomic_json(result,out/"summary.json");return result
 if start==0:random.seed(seed);np.random.seed(seed)
 lam=data["joint_weight"]["lambda_future"]
 for epoch in range(start,max_epochs):
  model.train();perm=torch.randperm(len(xfit),generator=gen);sls=[];fls=[]
  for ids in perm.split(256):
   xx=xfit[ids].to(device);yy=yfit[ids].to(device);st=fit["stage"][ids].to(device);ww=fit["slip_weight"][ids].to(device);opt.zero_grad(set_to_none=True);ls,lf,_,_=loss_parts(model,xx,yy,st,ww)
   if group=="S" and ls is None:skipped+=1;continue
   loss=ls if group=="S" else lf if group=="F" else (lf*lam if ls is None else ls+lam*lf)
   if not torch.isfinite(loss):raise ValueError("nonfinite loss")
   loss.backward()
   if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):raise ValueError("nonfinite gradient")
   opt.step();optimizer_steps+=1
   if group in ("S","J"):slip_seen+=int((st.eq(0)|st.eq(2)).sum())
   if group in ("F","J"):future_seen+=len(ids)
   if ls is not None:sls.append(float(ls.detach()))
   fls.append(float(lf.detach()))
  model.eval()
  with torch.no_grad():
   scores=[];fp=[]
   for ids in torch.arange(len(xsel)).split(1024):
    sl,fu=model(xsel[ids].to(device));scores.append(torch.sigmoid(sl).cpu());fp.append(fu.cpu())
   scores=torch.cat(scores);fp=torch.cat(fp)*norm["y_std"]+norm["y_mean"]
  metric=float((fp-sel["y"]).abs().mean()) if group=="F" else slip_pauc(sel["stage"].numpy(),scores.numpy())
  improved=metric<best_metric if group=="F" else metric>best_metric
  if improved:best_metric=metric;best_epoch=epoch;wait=0
  else:wait+=1
  history.append({"epoch":epoch,"fit_slip_loss":float(np.mean(sls)) if sls else None,"fit_future_loss":float(np.mean(fls)) if fls else None,"selection_metric":metric})
  ck={"identity":identity,"model":model.state_dict(),"optimizer":opt.state_dict(),"generator":gen.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"normalizer":norm,"joint_weight":data["joint_weight"],"epoch":epoch,"best_metric":best_metric,"best_epoch":best_epoch,"wait":wait,"history":history,"skipped_empty_slip_steps":skipped,"optimizer_steps":optimizer_steps,"slip_supervised_endpoints_seen":slip_seen,"future_supervised_endpoints_seen":future_seen}
  atomic_save(ck,latest)
  if improved:atomic_save(ck,best)
  if interrupt_after is not None and epoch>=interrupt_after:
   result={"status":"interrupted","epochs":len(history),"identity":identity};atomic_json(result,out/"summary.json");return result
  if wait>=patience:break
 result={"status":"complete","best_epoch":best_epoch,"best_metric":best_metric,"epochs":len(history),"skipped_empty_slip_steps":skipped,"optimizer_steps":optimizer_steps,"slip_supervised_endpoints_seen":slip_seen,"future_supervised_endpoints_seen":future_seen,"active_parameters":sum(p.numel() for p in params),"canonical_parameters":sum(p.numel() for p in model.parameters()),"latest_active_state_sha256":state_hash(torch.load(latest,map_location="cpu",weights_only=False)["model"],names),"identity":identity};atomic_json(result,out/"summary.json");return result

def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True)
 q=sp.add_parser("prepare");q.add_argument("--round14",type=Path,required=True);q.add_argument("--parent",type=Path,required=True);q.add_argument("--output",type=Path,required=True);q.add_argument("--fold",type=int,required=True);q.add_argument("--seed",type=int,required=True)
 q=sp.add_parser("train");q.add_argument("--data",type=Path,required=True);q.add_argument("--output",type=Path,required=True);q.add_argument("--group",choices=GROUPS,required=True);q.add_argument("--fold",type=int,required=True);q.add_argument("--seed",type=int,required=True);q.add_argument("--device",default="cpu");q.add_argument("--max-epochs",type=int,default=60);q.add_argument("--patience",type=int,default=10);q.add_argument("--interrupt-after",type=int)
 a=p.parse_args()
 if a.cmd=="prepare":print(stable_json(enrich_round14(a.round14,a.parent,a.output,a.fold,a.seed)))
 else:
  d=torch.load(a.data,map_location="cpu",weights_only=False);d["data_sha256"]=sha(a.data);print(stable_json(train(d,a.output,a.group,a.fold,a.seed,a.device,a.max_epochs,a.patience,a.interrupt_after)))
if __name__=="__main__":main()
