#!/usr/bin/env python3
"""Frozen E1/E2 trainer for V/C/M/MB on the exact Round-22 current support."""
from __future__ import annotations
import argparse,copy,fcntl,hashlib,importlib.util,json,os,random,tempfile,time
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch import nn

HERE=Path(__file__).resolve().parent;R18_PATH=HERE.parent/"round18_htt_force_conditioned_film"/"train.py"
spec=importlib.util.spec_from_file_location("r22_frozen_r18",R18_PATH);R18=importlib.util.module_from_spec(spec);spec.loader.exec_module(R18)
GROUPS=("V","C","M","MB");SEEDS=(20260914,20260915,20260916);ROLES=("fit","selection","calibration","validation");BATCH=256
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
def lock(out):
 out.mkdir(parents=True,exist_ok=True);h=(out/"TRAIN.lock").open("a+")
 try:fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:h.close();raise RuntimeError("task output lock already owned")
 return h
class Head(nn.Module):
 def __init__(self,group):
  super().__init__();self.group=group;self.visual_ln=nn.LayerNorm(192);self.gru=nn.GRU(195 if group=="C" else 192,128,batch_first=True);self.risk=nn.Linear(128,1)
  if group in ("M","MB"):self.film_hidden=nn.Linear(3,14);self.film_out=nn.Linear(14,384)
 def conditioned_visual(self,x):
  visual=self.visual_ln(x[...,:192])
  if self.group not in ("M","MB"):return visual,torch.zeros_like(visual),torch.zeros_like(visual)
  gamma,beta=self.film_out(nn.functional.gelu(self.film_hidden(x[...,192:195]))).chunk(2,-1)
  if self.group=="MB":gamma=.25*torch.tanh(gamma);beta=.5*torch.tanh(beta)
  return (1+gamma)*visual+beta,gamma,beta
 def components(self,x):
  v,g,b=self.conditioned_visual(x);f=torch.cat((v,x[...,192:195]),-1) if self.group=="C" else v;return {"logit":self.risk(self.gru(f)[0][:,-1]).squeeze(-1),"gamma":g,"beta":b}
 def forward(self,x):return self.components(x)["logit"]
def init_model(group,seed):
 m=Head(group)
 for name in ("visual_ln","gru","risk","film_hidden"):
  if not hasattr(m,name):continue
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(R18.module_seed(seed,name))
   for pn,p in getattr(m,name).named_parameters():
    if p.ndim>=2:nn.init.xavier_uniform_(p)
    elif name=="visual_ln" and pn=="weight":nn.init.ones_(p)
    else:nn.init.zeros_(p)
 if group in ("M","MB"):nn.init.zeros_(m.film_out.weight);nn.init.zeros_(m.film_out.bias)
 return m
def validate(d,path,fold,seed,support_inventory):
 if d.get("schema")!="round22_e3_current_common_gt_v1" or (d.get("fold"),d.get("seed"))!=(fold,seed):raise ValueError("cache identity")
 inventory=json.loads(Path(support_inventory).read_text());matches=[r for r in inventory.get("runs",[]) if (r.get("fold"),r.get("seed"))==(fold,seed)]
 if len(matches)!=1 or Path(matches[0]["prepared"]).resolve()!=Path(path).resolve() or matches[0]["prepared_sha256"]!=sha(path):raise ValueError("current cache fold/seed inventory mismatch")
 seen=set()
 for role in ROLES:
  r=d["roles"][role]
  if r["x"].shape[1:]!=(9,195) or not torch.isfinite(r["x"]).all() or (r["t"]<13).any():raise ValueError(f"bad {role}")
  g=set(r["leakage_group"])
  if seen&g:raise ValueError("role leakage")
  seen|=g
 if "test" in d["roles"]:raise ValueError("test forbidden")
def fit_assets(d):
 r=d["roles"]["fit"];primary=torch.nonzero(r["stage"].eq(0)|r["stage"].eq(2)).flatten();x=r["x"][primary].double();norm={"mean":x.reshape(-1,195).mean(0).float(),"std":x.reshape(-1,195).std(0,unbiased=False).clamp_min(1e-6).float()};w=torch.zeros(len(r["stage"]),dtype=torch.float64);n=len(primary)
 for c in (0,2):
  ids=torch.nonzero(r["stage"].eq(c)).flatten();eps=np.asarray(r["episode_id"],object)[ids.numpy()]
  for e in sorted(set(eps.tolist())):q=ids[torch.from_numpy(eps==e)];w[q]=n/(2*len(set(eps.tolist()))*len(q))
 return primary,norm,w.float()
def infer(m,x,device):
 m.eval();out=[]
 with torch.inference_mode():
  for z in x.split(1024):out.append(torch.sigmoid(m(z.to(device))).cpu())
 return torch.cat(out)
def low_fpr_auc(stage,probability,limit=.1):
 labels=(stage==2).numpy().astype(np.int64);scores=probability.numpy().astype(np.float64);order=np.argsort(-scores,kind="stable");labels=labels[order];scores=scores[order];ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1];tpr=np.r_[0.,np.cumsum(labels)[ends]/labels.sum()];fpr=np.r_[0.,np.cumsum(1-labels)[ends]/(len(labels)-labels.sum())];i=np.flatnonzero(fpr<limit)[-1];xs=fpr[fpr<=limit];ys=tpr[fpr<=limit]
 if not np.any(fpr==limit):q=(limit-fpr[i])/(fpr[i+1]-fpr[i]);xs=np.r_[fpr[:i+1],limit];ys=np.r_[tpr[:i+1],tpr[i]+q*(tpr[i+1]-tpr[i])]
 return float(np.trapezoid(ys,xs)/limit)
def train(a):
 if (a.group,a.fold,a.seed) not in {(g,f,s) for g in GROUPS for f in range(1,5) for s in SEEDS}:raise ValueError("grid")
 if a.formal:
  gate=json.loads(a.authorization.read_text()) if a.authorization and a.authorization.is_file() else {}
  if gate.get("g1_formal_authorized") is not True or gate.get("budget_pass") is not True:raise ValueError("formal frozen dispatch blocked")
  if not a.protocol or not a.protocol.is_file() or (a.max_epochs,a.patience)!=(60,10) or a.interrupt_after_commit>=0 or "formal" not in a.output.parts or "smoke" in a.output.parts:raise ValueError("formal protocol/budget/smoke override")
  protocol=json.loads(a.protocol.read_text())
  if protocol.get("sources",{}).get(Path(__file__).name,{}).get("sha256")!=sha(__file__):raise ValueError("formal protocol source mismatch")
 task_lock=lock(a.output);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 d=torch.load(a.data,map_location="cpu",weights_only=False);validate(d,a.data,a.fold,a.seed,a.support_inventory);primary,norm,weights=fit_assets(d);x={r:(d["roles"][r]["x"]-norm["mean"])/norm["std"] for r in ROLES};target=d["roles"]["fit"]["stage"].eq(2).float()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);m=init_model(a.group,a.seed).to(a.device);initial=copy.deepcopy(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);gen=torch.Generator().manual_seed(a.seed+1800)
 ident={"schema":"round22_frozen_identity_v1","package":"E2" if a.group=="MB" else "E1","group":a.group,"fold":a.fold,"seed":a.seed,"data_sha256":sha(a.data),"support_inventory_sha256":sha(a.support_inventory),"source_sha256":sha(__file__),"r18_source_sha256":sha(R18_PATH),"normalizer_sha256":hashlib.sha256(norm["mean"].numpy().tobytes()+norm["std"].numpy().tobytes()).hexdigest(),"max_epochs":a.max_epochs,"patience":a.patience,"batch":BATCH,"selector":"corrected_internal_selection_pAUC_0_0.1_earliest_strict_max","formal":a.formal,"authorization_sha256":sha(a.authorization) if a.formal else None,"protocol_sha256":sha(a.protocol) if a.formal else None}
 config=a.output/"config.json";manifest=a.output/"COMMIT.json";(a.output/"commits").mkdir(exist_ok=True);start=0;best=-float("inf");best_epoch=-1;wait=0;history=[];steps=0;best_entry=None
 if config.exists() and json.loads(config.read_text())!=ident:raise ValueError("existing config identity mismatch")
 if manifest.exists():
  cm=json.loads(manifest.read_text());ck=torch.load(a.output/cm["latest"]["path"],map_location="cpu",weights_only=False)
  if sha(a.output/cm["latest"]["path"])!=cm["latest"]["sha256"] or ck["identity"]!=ident:raise ValueError("resume identity")
  m.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
  for state in opt.state.values():
   for k,v in state.items():
    if torch.is_tensor(v):state[k]=v.to(a.device)
  gen.set_state(ck["generator"]);random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"]);start=ck["epoch"]+1;best=ck["best_metric"];best_epoch=ck["best_epoch"];wait=ck["wait"];history=ck["history"];steps=ck["optimizer_steps"];best_entry=cm["best"]
 atomic_json(config,ident)
 for epoch in range(start,a.max_epochs):
  if wait>=a.patience:break
  m.train();losses=[]
  for ids in primary[torch.randperm(len(primary),generator=gen)].split(BATCH):
   opt.zero_grad(set_to_none=True);logit=m(x["fit"][ids].to(a.device));loss=(nn.functional.binary_cross_entropy_with_logits(logit,target[ids].to(a.device),reduction="none")*weights[ids].to(a.device)).mean()
   if not torch.isfinite(loss):raise FloatingPointError("loss")
   loss.backward()
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError("gradient")
   opt.step();steps+=1;losses.append(float(loss.detach()))
  sel=d["roles"]["selection"];mask=sel["stage"].eq(0)|sel["stage"].eq(2);metric=low_fpr_auc(sel["stage"][mask],infer(m,x["selection"],a.device)[mask]);improved=metric>best;best,best_epoch,wait=(metric,epoch,0) if improved else (best,best_epoch,wait+1);history.append({"epoch":epoch,"fit_loss":float(np.mean(losses)),"selection_pAUC_0_0.1":metric})
  ck={"identity":ident,"model":copy.deepcopy(m.state_dict()),"optimizer":opt.state_dict(),"generator":gen.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"normalizer":norm,"epoch":epoch,"best_metric":best,"best_epoch":best_epoch,"wait":wait,"history":history,"optimizer_steps":steps};cp=a.output/"commits"/f"epoch-{epoch:04d}.pth";atomic_save(cp,ck);entry={"path":str(cp.relative_to(a.output)),"sha256":sha(cp),"epoch":epoch};best_entry=entry if improved else best_entry;atomic_json(manifest,{"schema":"round22_checkpoint_commit_v1","latest":entry,"best":best_entry,"epoch":epoch})
  if a.interrupt_after_commit==epoch:return {"status":"interrupted","epoch":epoch,"identity":ident}
 cm=json.loads(manifest.read_text());bp=a.output/cm["best"]["path"]
 if sha(bp)!=cm["best"]["sha256"]:raise ValueError("best commit hash")
 bestck=torch.load(bp,map_location="cpu",weights_only=False);atomic_save(a.output/"best.pth",bestck);result={"schema":"round22_frozen_summary_v1","status":"complete","package":ident["package"],"group":a.group,"fold":a.fold,"seed":a.seed,"epochs":len(history),"best_epoch":best_epoch,"best_metric":best,"optimizer_steps":steps,"parameters":sum(p.numel() for p in m.parameters()),"identity":ident};atomic_json(a.output/"summary.json",result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--group",choices=GROUPS,required=True);p.add_argument("--fold",type=int,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--support-inventory",type=Path,default=HERE/"PREPARE_CURRENT_SUPPORT.json");p.add_argument("--device",default="cpu");p.add_argument("--max-epochs",type=int,default=60);p.add_argument("--patience",type=int,default=10);p.add_argument("--formal",action="store_true");p.add_argument("--authorization",type=Path);p.add_argument("--protocol",type=Path);p.add_argument("--interrupt-after-commit",type=int,default=-1);print(json.dumps(train(p.parse_args())))
if __name__=="__main__":main()
