#!/usr/bin/env python3
"""Minimal resumable Round-14 future-force trainer; formal dispatch is root-controlled."""
from __future__ import annotations
import argparse, hashlib, json, os, random, tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

HORIZONS=(1,5,10); GROUPS=("V","F_concat","F_dual")

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def atomic_save(obj,path):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".",suffix=".tmp"); os.close(fd)
 try: torch.save(obj,tmp); os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def atomic_json(obj,path):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+".",suffix=".tmp"); os.close(fd)
 try:
  Path(tmp).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)
def stable_json(obj): return json.dumps(obj,sort_keys=True,separators=(",",":"))
def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

class FutureModel(nn.Module):
 def __init__(self,group):
  super().__init__(); self.group=group
  if group=="V": self.visual=nn.Sequential(nn.Linear(192,48),nn.GELU())
  elif group=="F_concat": self.joint=nn.Sequential(nn.Linear(195,48),nn.GELU())
  elif group=="F_dual":
   self.visual=nn.Sequential(nn.Linear(192,45),nn.GELU()); self.force=nn.Sequential(nn.Linear(3,3),nn.GELU())
  else: raise ValueError(group)
  self.gru=nn.GRU(48,48,batch_first=True); self.out=nn.Linear(48,9)
 def forward(self,x):
  if self.group=="V": z=self.visual(x[...,:192])
  elif self.group=="F_concat": z=self.joint(x[...,:195])
  else: z=torch.cat((self.visual(x[...,:192]),self.force(x[...,192:195])),dim=-1)
  return self.out(self.gru(z)[0][:,-1]).reshape(-1,3,3)

def init_model(group,seed):
 torch.manual_seed(seed); m=FutureModel(group)
 for n,p in m.named_parameters():
  if p.ndim>=2: nn.init.xavier_uniform_(p)
  else: nn.init.zeros_(p)
 return m

def assert_role_isolation(roles):
 seen={}
 for role,d in roles.items():
  if role=="test": raise ValueError("test role forbidden")
  groups=set(d["leakage_group"])
  for other,old in seen.items():
   overlap=groups&old
   if overlap: raise ValueError(f"leakage overlap {role}/{other}: {sorted(overlap)[:2]}")
  seen[role]=groups

def fit_norm(x,y):
 xm=x.mean((0,1)); xs=x.std((0,1),unbiased=False).clamp_min(1e-6)
 ym=y.mean(0); ys=y.std(0,unbiased=False).clamp_min(1e-6)
 return {"x_mean":xm,"x_std":xs,"y_mean":ym,"y_std":ys}
def norm_xy(x,y,n): return (x-n["x_mean"])/n["x_std"],(y-n["y_mean"])/n["y_std"]

def load_formal_cache(prepared,support):
 """Bind R10 causal features to same-episode future targets without reading test."""
 d=torch.load(prepared,map_location="cpu",weights_only=False); s=json.load(open(support))
 for key in ("force_prediction_manifest","force_checkpoint","visual_checkpoint","source_checkpoint"):
  ref=d["provenance"][key]
  if sha(ref["path"])!=ref["sha256"]: raise ValueError(f"upstream identity mismatch: {key}")
 if set(d["roles"])-{"train","calibration","validation"}: raise ValueError("unexpected/test role")
 entry={e["episode_id"]:e for e in s["entries"]}; train_sub={e["leakage_group"]:e["role"] for e in s["entries"] if e["outer_role"]=="train"}; target_cache={}
 roles={}
 for outer,r in d["roles"].items():
  buckets=("fit","selection") if outer=="train" else (outer,)
  for bucket in buckets:
   ids=[]; ys=[]; y_current=[]
   for i,(eid,t,g) in enumerate(zip(r["episode_id"],r["t"].tolist(),r["leakage_group"])):
    if outer=="train" and train_sub[g]!=bucket: continue
    e=entry[eid]
    if eid not in target_cache: target_cache[eid]=np.load(e["force_native_n_path"])
    arr=target_cache[eid]
    if t<13 or t+max(HORIZONS)>=len(arr): continue
    ids.append(i); y_current.append(np.asarray(arr[t],dtype=np.float32)); ys.append(np.asarray(arr[[t+h for h in HORIZONS]],dtype=np.float32))
   if not ids: raise ValueError(f"empty role {bucket}")
   roles[bucket]={"x":r["x"][ids,:,:195].clone(),"y_current":torch.tensor(np.stack(y_current)),"y":torch.tensor(np.stack(ys)),"episode_id":[r["episode_id"][i] for i in ids],"leakage_group":[r["leakage_group"][i] for i in ids],"t":r["t"][ids].clone()}
 assert_role_isolation(roles)
 return {"schema":"round14_future_cache_v1","horizons":HORIZONS,"roles":roles,"provenance":{"prepared":str(Path(prepared).resolve()),"prepared_sha256":sha(prepared),"support":str(Path(support).resolve()),"support_sha256":sha(support),"immutable_upstream":{k:d["provenance"][k] for k in ("force_prediction_manifest","force_checkpoint","visual_checkpoint","source_checkpoint")},"optimizer_scope":"new_future_model_only_cached_upstream_not_instantiated"}}

def train(data,out,group,fold,seed,max_epochs=60,patience=10,interrupt_after=None,device="cpu"):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); assert group in GROUPS; assert_role_isolation(data["roles"])
 fit=data["roles"]["fit"]; sel=data["roles"]["selection"]; n=fit_norm(fit["x"],fit["y"])
 xfit,yfit=norm_xy(fit["x"],fit["y"],n); xsel,ysel=norm_xy(sel["x"],sel["y"],n)
 random.seed(seed); np.random.seed(seed); model=init_model(group,seed).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
 gen=torch.Generator().manual_seed(seed+1400); latest=out/"latest.pth"; best=out/"best.pth"
 norm_identity=hashlib.sha256(b"".join(n[k].numpy().tobytes() for k in ("x_mean","x_std","y_mean","y_std"))).hexdigest()
 protocol_path=Path(__file__).with_name("PROTOCOL.md")
 ident={"group":group,"fold":fold,"seed":seed,"horizons":list(HORIZONS),"max_epochs":max_epochs,"patience":patience,"batch_size":256,"lr":1e-3,"weight_decay":1e-4,"round14_data_sha256":data.get("round14_data_sha256","synthetic"),"parent_prepared_sha256":data.get("provenance",{}).get("prepared_sha256","synthetic"),"support_sha256":data.get("provenance",{}).get("support_sha256","synthetic"),"normalizer_sha256":norm_identity,"source_sha256":sha(__file__),"protocol_sha256":sha(protocol_path) if protocol_path.exists() else "synthetic"}
 start=0; best_metric=float("inf"); best_epoch=-1; wait=0; history=[]
 if latest.exists():
  ck=torch.load(latest,map_location="cpu",weights_only=False)
  if ck["identity"]!=ident: raise ValueError("resume identity mismatch")
  model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
  for state in opt.state.values():
   for k,v in state.items():
    if torch.is_tensor(v): state[k]=v.to(device)
  gen.set_state(ck["generator"].cpu()); random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"])
  start=ck["epoch"]+1; best_metric=ck["best_metric"]; best_epoch=ck["best_epoch"]; wait=ck["wait"]; history=ck["history"]
 if wait>=patience or start>=max_epochs:
  result={"status":"complete","best_epoch":best_epoch,"best_metric_native_n_mae":best_metric,"epochs":len(history),"state_sha256":state_hash(model.state_dict()),"parameters":sum(p.numel() for p in model.parameters()),"identity":ident,"resume_without_extra_epoch":True}; atomic_json(result,out/"summary.json"); return result
 for epoch in range(start,max_epochs):
  model.train(); perm=torch.randperm(len(xfit),generator=gen); losses=[]
  for ids in perm.split(256):
   opt.zero_grad(set_to_none=True); pred=model(xfit[ids].to(device)); loss=nn.functional.smooth_l1_loss(pred,yfit[ids].to(device),beta=1.0)
   if not torch.isfinite(loss): raise ValueError("nonfinite loss")
   loss.backward()
   if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError("nonfinite gradient")
   opt.step(); losses.append(float(loss))
  model.eval()
  with torch.no_grad():
   pred_native=model(xsel.to(device)).cpu()*n["y_std"]+n["y_mean"]
   metric=float((pred_native-sel["y"]).abs().mean())
  improved=metric<best_metric
  if improved: best_metric=metric; best_epoch=epoch; wait=0
  else: wait+=1
  history.append({"epoch":epoch,"fit_loss":float(np.mean(losses)),"selection_mae":metric})
  ck={"identity":ident,"model":model.state_dict(),"optimizer":opt.state_dict(),"generator":gen.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"normalizer":n,"epoch":epoch,"best_metric":best_metric,"best_epoch":best_epoch,"wait":wait,"history":history}
  atomic_save(ck,latest)
  if improved: atomic_save(ck,best)
  if interrupt_after is not None and epoch>=interrupt_after:
   result={"status":"interrupted","latest":str(latest),"identity":ident,"epochs":len(history)}; atomic_json(result,out/"summary.json"); return result
  if wait>=patience: break
 result={"status":"complete","best_epoch":best_epoch,"best_metric_native_n_mae":best_metric,"epochs":len(history),"state_sha256":state_hash(torch.load(latest,map_location="cpu",weights_only=False)["model"]),"parameters":sum(p.numel() for p in model.parameters()),"identity":ident}; atomic_json(result,out/"summary.json"); return result

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
 q=sub.add_parser("prepare"); q.add_argument("--prepared",type=Path,required=True); q.add_argument("--support",type=Path,required=True); q.add_argument("--output",type=Path,required=True)
 q=sub.add_parser("train"); q.add_argument("--data",type=Path,required=True); q.add_argument("--output",type=Path,required=True); q.add_argument("--group",choices=GROUPS,required=True); q.add_argument("--fold",type=int,required=True); q.add_argument("--seed",type=int,required=True); q.add_argument("--device",default="cpu"); q.add_argument("--max-epochs",type=int,default=60); q.add_argument("--patience",type=int,default=10); q.add_argument("--interrupt-after",type=int)
 a=p.parse_args()
 if a.cmd=="prepare": atomic_save(load_formal_cache(a.prepared,a.support),a.output); print(stable_json({"status":"complete","output":str(a.output),"sha256":sha(a.output)}))
 else:
  d=torch.load(a.data,map_location="cpu",weights_only=False); d["round14_data_sha256"]=sha(a.data); print(stable_json(train(d,a.output,a.group,a.fold,a.seed,a.max_epochs,a.patience,a.interrupt_after,a.device)))
if __name__=="__main__": main()
