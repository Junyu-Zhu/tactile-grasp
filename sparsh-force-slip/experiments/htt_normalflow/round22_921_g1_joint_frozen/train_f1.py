#!/usr/bin/env python3
"""F1 frozen current-state force-change trainer for K-V/K-F/K-VF."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,json,os,random,tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
GROUPS=("K-V","K-F","K-VF");WIDTHS={"K-V":192,"K-F":3,"K-VF":195};EXPECTED_PARAMS={"K-V":23817,"K-F":14745,"K-VF":23961};SEEDS=(20260914,20260915,20260916);ROLES=("fit","selection","calibration","validation");HORIZONS=(1,5,10);BATCH=256
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic_save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:torch.save(x,t);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:Path(t).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def task_lock(out):
 out.mkdir(parents=True,exist_ok=True);h=(out/"TRAIN.lock").open("a+")
 try:fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:h.close();raise RuntimeError("task output lock already owned")
 return h
class Model(nn.Module):
 def __init__(self,group):
  super().__init__();self.group=group;self.projection=nn.Sequential(nn.Linear(WIDTHS[group],48),nn.GELU());self.gru=nn.GRU(48,48,batch_first=True);self.readout=nn.Linear(48,9)
 def input_current(self,x):
  z=x[:,-1]
  if self.group=="K-V":z=z[:,:192]
  elif self.group=="K-F":z=z[:,192:195]
  return z
 def forward(self,x):
  z=self.projection(self.input_current(x)).unsqueeze(1).repeat(1,2,1);return self.readout(self.gru(z)[0][:,-1]).reshape(-1,3,3)
def init_model(group,seed):
 torch.manual_seed(seed);m=Model(group)
 for name,p in m.named_parameters():nn.init.zeros_(p) if name.startswith("readout.") or p.ndim==1 else nn.init.xavier_uniform_(p)
 if sum(p.numel() for p in m.parameters())!=EXPECTED_PARAMS[group]:raise ValueError("parameter contract")
 return m
def validate(d,path,fold,seed,support_inventory):
 if d.get("schema")!="round14_future_cache_v1" or tuple(d.get("horizons",()))!=HORIZONS:raise ValueError("future cache identity")
 inventory=json.loads(Path(support_inventory).read_text());matches=[r for r in inventory.get("support",[]) if (r.get("fold"),r.get("seed"))==(fold,seed)]
 if len(matches)!=1 or Path(matches[0]["future_cache"]).resolve()!=Path(path).resolve() or matches[0]["future_sha256"]!=sha(path):raise ValueError("future cache fold/seed inventory mismatch")
 seen=set()
 for role in ROLES:
  r=d["roles"][role]
  if r["x"].shape[1:]!=(9,195) or r["y"].shape[1:]!=(3,3) or r["y_current"].shape[1:]!=(3,) or not all(torch.isfinite(r[k]).all() for k in ("x","y","y_current")):raise ValueError(f"bad {role}")
  groups=set(r["leakage_group"])
  if seen&groups:raise ValueError("role leakage")
  seen|=groups
 if "test" in d["roles"]:raise ValueError("test forbidden")
def delta(r):return r["y"]-r["y_current"][:,None,:]
def assets(fit,group):
 x=fit["x"][:,:,:192] if group=="K-V" else fit["x"][:,:,192:195] if group=="K-F" else fit["x"]
 return {"x_mean":x.mean((0,1)),"x_std":x.std((0,1),unbiased=False).clamp_min(1e-6),"delta_scale":delta(fit).std((0,1),unbiased=False).clamp_min(1e-6)}
def normalize_x(x,n,group):
 z=x[:,:,:192] if group=="K-V" else x[:,:,192:195] if group=="K-F" else x
 z=(z-n["x_mean"])/n["x_std"]
 if group=="K-V":return torch.cat((z,torch.zeros((*z.shape[:2],3),dtype=z.dtype)),2)
 if group=="K-F":return torch.cat((torch.zeros((*z.shape[:2],192),dtype=z.dtype),z),2)
 return z
def predict(m,x,n,device,shrink=1.):
 m.eval();out=[]
 with torch.inference_mode():
  for z in x.split(2048):out.append(m(z.to(device)).cpu()*n["delta_scale"]*shrink)
 return torch.cat(out)
def train(a):
 if a.group not in GROUPS or a.fold not in range(1,5) or a.seed not in SEEDS:raise ValueError("grid")
 if a.formal:
  gate=json.loads(a.authorization.read_text()) if a.authorization and a.authorization.is_file() else {}
  if gate.get("g1_formal_authorized") is not True or gate.get("budget_pass") is not True:raise ValueError("formal F1 blocked")
  if not a.protocol or not a.protocol.is_file() or (a.max_epochs,a.patience)!=(60,10) or a.interrupt_after_commit>=0 or "formal" not in a.output.parts or "smoke" in a.output.parts:raise ValueError("formal protocol/budget/smoke override")
  protocol=json.loads(a.protocol.read_text())
  if protocol.get("sources",{}).get(Path(__file__).name,{}).get("sha256")!=sha(__file__):raise ValueError("formal protocol source mismatch")
 held=task_lock(a.output);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);d=torch.load(a.data,map_location="cpu",weights_only=False);validate(d,a.data,a.fold,a.seed,a.support_inventory);n=assets(d["roles"]["fit"],a.group);x={r:normalize_x(d["roles"][r]["x"],n,a.group) for r in ROLES};y=delta(d["roles"]["fit"])/n["delta_scale"]
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);m=init_model(a.group,a.seed).to(a.device);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);gen=torch.Generator().manual_seed(a.seed+2000)
 nh=hashlib.sha256(b"".join(n[k].numpy().tobytes() for k in ("x_mean","x_std","delta_scale"))).hexdigest();ident={"schema":"round22_f1_identity_v1","group":a.group,"fold":a.fold,"seed":a.seed,"data_sha256":sha(a.data),"source_sha256":sha(__file__),"normalizer_sha256":nh,"normalizer_fit":"all_nine_positions_fit_only","current_forward":"last_position_repeated_twice","target":"true_future_minus_true_current_scale_only_no_center","absolute_anchor":"shared_frozen_predicted_current_force","max_epochs":a.max_epochs,"patience":a.patience,"batch":BATCH,"selector":"earliest_strict_min_selection_native_delta_mae","formal":a.formal,"authorization_sha256":sha(a.authorization) if a.formal else None,"protocol_sha256":sha(a.protocol) if a.formal else None}
 ident["support_inventory_sha256"]=sha(a.support_inventory);config=a.output/"config.json";manifest=a.output/"COMMIT.json";(a.output/"commits").mkdir(exist_ok=True);start=0;best=float("inf");best_epoch=-1;wait=0;history=[];steps=0;best_entry=None
 if config.exists() and json.loads(config.read_text())!=ident:raise ValueError("existing config identity mismatch")
 if manifest.exists():
  cm=json.loads(manifest.read_text());cp=a.output/cm["latest"]["path"]
  if sha(cp)!=cm["latest"]["sha256"]:raise ValueError("commit hash")
  ck=torch.load(cp,map_location="cpu",weights_only=False)
  if ck["identity"]!=ident:raise ValueError("resume identity")
  m.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
  for st in opt.state.values():
   for k,v in st.items():
    if torch.is_tensor(v):st[k]=v.to(a.device)
  gen.set_state(ck["generator"]);random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"]);start=ck["epoch"]+1;best=ck["best_metric"];best_epoch=ck["best_epoch"];wait=ck["wait"];history=ck["history"];steps=ck["optimizer_steps"];best_entry=cm["best"]
 atomic_json(config,ident)
 for epoch in range(start,a.max_epochs):
  if wait>=a.patience:break
  m.train();losses=[]
  for ids in torch.randperm(len(y),generator=gen).split(BATCH):
   opt.zero_grad(set_to_none=True);loss=nn.functional.smooth_l1_loss(m(x["fit"][ids].to(a.device)),y[ids].to(a.device),beta=1)
   if not torch.isfinite(loss):raise FloatingPointError("loss")
   loss.backward()
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError("gradient")
   opt.step();steps+=1;losses.append(float(loss.detach()))
  pred=predict(m,x["selection"],n,a.device);metric=float((pred-delta(d["roles"]["selection"])).abs().mean());improved=metric<best;best,best_epoch,wait=(metric,epoch,0) if improved else (best,best_epoch,wait+1);history.append({"epoch":epoch,"fit_loss":float(np.mean(losses)),"selection_native_delta_mae":metric})
  ck={"identity":ident,"model":copy.deepcopy(m.state_dict()),"optimizer":opt.state_dict(),"generator":gen.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"normalizer":n,"epoch":epoch,"best_metric":best,"best_epoch":best_epoch,"wait":wait,"history":history,"optimizer_steps":steps};cp=a.output/"commits"/f"epoch-{epoch:04d}.pth";atomic_save(cp,ck);entry={"path":str(cp.relative_to(a.output)),"sha256":sha(cp),"epoch":epoch};best_entry=entry if improved else best_entry;atomic_json(manifest,{"schema":"round22_checkpoint_commit_v1","epoch":epoch,"latest":entry,"best":best_entry})
  if a.interrupt_after_commit==epoch:return {"status":"interrupted","epoch":epoch,"identity":ident}
 cm=json.loads(manifest.read_text());bp=a.output/cm["best"]["path"]
 if sha(bp)!=cm["best"]["sha256"]:raise ValueError("best commit hash")
 bestck=torch.load(bp,map_location="cpu",weights_only=False);atomic_save(a.output/"best.pth",bestck);result={"schema":"round22_f1_summary_v1","status":"complete","group":a.group,"fold":a.fold,"seed":a.seed,"best_epoch":best_epoch,"best_metric_native_delta_mae":best,"epochs":len(history),"optimizer_steps":steps,"parameters":sum(p.numel() for p in m.parameters()),"identity":ident};atomic_json(a.output/"summary.json",result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--group",choices=GROUPS,required=True);p.add_argument("--fold",type=int,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--support-inventory",type=Path,default=HERE/"G1_PREPARE.json");p.add_argument("--device",default="cpu");p.add_argument("--max-epochs",type=int,default=60);p.add_argument("--patience",type=int,default=10);p.add_argument("--formal",action="store_true");p.add_argument("--authorization",type=Path);p.add_argument("--protocol",type=Path);p.add_argument("--interrupt-after-commit",type=int,default=-1);print(json.dumps(train(p.parse_args())))
if __name__=="__main__":main()
