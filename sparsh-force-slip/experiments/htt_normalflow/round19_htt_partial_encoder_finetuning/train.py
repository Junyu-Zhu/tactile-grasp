#!/usr/bin/env python3
"""Round-19 V2/M2 trainer: cached frozen prefix, live trainable MAE blocks 10/11."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, os, random, sys, tempfile, time
from pathlib import Path
import numpy as np
os.environ.setdefault("XFORMERS_DISABLED","1");os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch import nn

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path[:0]=[str(HERE.parent),str(REPO/"scripts"),str(HERE.parent/"round5_force_conditioned_slip"/"current")]
import phase2_b_multitask as p2
from sources import load_decoupled_decoder,load_r3_slip_branch

GROUPS=("V2","M2");SEEDS=(20260914,20260915,20260916);ROLES=("fit","selection","calibration","validation")
BATCH=64;ACCUM=4;EFFECTIVE_BATCH=256;MAX_EPOCHS=60;PATIENCE=10
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def state_hash(s):
 h=hashlib.sha256()
 for k,v in sorted(s.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+".",suffix=".tmp");os.close(fd)
 try:
  Path(t).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
  if os.environ.get("ROUND19_CRASH_BEFORE_REPLACE")==p.name and int(os.environ.get("ROUND19_CRASH_EPOCH","-1"))==int(x.get("epoch",-2)):
   os._exit(79)
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def atomic_save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+".",suffix=".tmp");os.close(fd)
 try:
  torch.save(x,t)
  if os.environ.get("ROUND19_CRASH_BEFORE_REPLACE")==p.name and int(os.environ.get("ROUND19_CRASH_EPOCH","-1"))==int(x.get("epoch",-2)):
   os._exit(79)
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def load_r18():
 p=HERE.parent/"round18_htt_force_conditioned_film"/"train.py";s=importlib.util.spec_from_file_location("r18_train",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
R18=load_r18()

class Model(nn.Module):
 def __init__(self,group,seed,source,r3):
  super().__init__();base,_=p2.load_b_checkpoint(source,torch.device("cpu"));e=base.encoder
  self.blocks=copy.deepcopy(e.blocks[10:]).requires_grad_(True);self.norm=copy.deepcopy(e.norm).requires_grad_(False)
  decoder,_=load_decoupled_decoder(source);branch=load_r3_slip_branch(decoder,r3)
  self.pooler=copy.deepcopy(branch.slip_pooler).requires_grad_(False);self.trunk=copy.deepcopy(branch.slip_trunk).requires_grad_(False)
  self.head=R18.init_model("M0" if group=="M2" else "V0",seed)
 def train(self,mode=True):
  super().train(mode);self.norm.eval();self.pooler.eval();self.trunk.eval();return self
 def visual(self,z):
  for b in self.blocks:z=b(z)
  z=self.norm(z)[:,1:]
  return self.trunk(self.pooler(z).squeeze(1))
 def components(self,z,force):
  n,h=z.shape[:2];v=self.visual(z.reshape(-1,*z.shape[2:])).reshape(n,h,192);x=torch.cat((v,force),-1);return self.head.components(x)
 def forward(self,z,force):return self.components(z,force)["logit"]

class Data:
 def __init__(self,prepared,index):
  self.d=torch.load(prepared,map_location="cpu",weights_only=False);idx=json.loads(index.read_text());self.paths={x["episode_id"]:x["path"] for x in idx["entries"]};self.mm={}
  if "test" in self.d["roles"]:raise ValueError("test forbidden")
 def batch(self,role,ids):
  r=self.d["roles"][role];zs=[]
  for i in ids.tolist():
   eid=r["episode_id"][i];t=int(r["t"][i]);
   if eid not in self.mm:self.mm[eid]=np.load(self.paths[eid],mmap_mode="r",allow_pickle=False)
   # Nine base steps s=t-8..t, each prefix token represents raw pair (s,s-5).
   zs.append(np.array(self.mm[eid][t-8:t+1],copy=True))
  return torch.from_numpy(np.stack(zs)),r["x"][ids,:,192:195].clone()

def fit_assets(d):
 fit=d["roles"]["fit"];mask=fit["stage"].eq(0)|fit["stage"].eq(2);primary=torch.nonzero(mask,as_tuple=False).flatten();force=fit["x"][primary,:,192:195].double();norm={"mean":force.reshape(-1,3).mean(0).float(),"std":force.reshape(-1,3).std(0,unbiased=False).clamp_min(1e-6).float()};n=len(primary);w=torch.zeros(len(mask),dtype=torch.float64);records=[]
 for c in (0,2):
  ids=torch.nonzero(fit["stage"].eq(c)).flatten();eps=np.asarray(fit["episode_id"],object)[ids.numpy()];u=sorted(set(eps.tolist()))
  for ep in u:
   q=ids[torch.from_numpy(eps==ep)];v=n/(2*len(u)*len(q));w[q]=v;records.append({"class":c,"episode":ep,"endpoints":len(q),"weight":v})
 return primary,norm,w.float(),records
def low_fpr_auc(stage,probability,limit=.1):
 labels=(stage==2).numpy().astype(np.int64);scores=probability.numpy().astype(np.float64)
 if set(labels.tolist())!={0,1} or not np.isfinite(scores).all():raise ValueError("selection lacks two finite classes")
 order=np.argsort(-scores,kind="stable");labels=labels[order];scores=scores[order];ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1]
 tpr=np.r_[0.,np.cumsum(labels)[ends]/labels.sum()];fpr=np.r_[0.,np.cumsum(1-labels)[ends]/(len(labels)-labels.sum())]
 if np.any(fpr==limit):
  keep=fpr<=limit;xs=fpr[keep];ys=tpr[keep]
 else:
  below=np.flatnonzero(fpr<limit);i=below[-1]
  if i==len(fpr)-1:xs=np.r_[fpr,limit];ys=np.r_[tpr,tpr[-1]]
  else:
   q=(limit-fpr[i])/(fpr[i+1]-fpr[i]);xs=np.r_[fpr[:i+1],limit];ys=np.r_[tpr[:i+1],tpr[i]+q*(tpr[i+1]-tpr[i])]
 return float(np.trapezoid(ys,xs)/limit)
def infer(model,data,role,norm,device):
 out=[];model.eval()
 with torch.inference_mode():
  for ids in torch.arange(len(data.d["roles"][role]["t"])).split(BATCH):
   z,f=data.batch(role,ids);f=(f-norm["mean"])/norm["std"];out.append(torch.sigmoid(model(z.to(device),f.to(device))).cpu())
 return torch.cat(out)

def train(a):
 if a.group not in GROUPS or a.seed not in SEEDS:raise ValueError("grid")
 if a.formal and ("formal" not in a.output.parts or "smoke" in a.output.parts):raise ValueError("formal path")
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 data=Data(a.data,a.prefix_index);primary,norm,weights,records=fit_assets(data.d);target=data.d["roles"]["fit"]["stage"].eq(2).float()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 model=Model(a.group,a.seed,a.source,a.visual_checkpoint).to(a.device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};trainable=[p for p in model.parameters() if p.requires_grad]
 enc=list(model.blocks.parameters());head=list(model.head.parameters());opt=torch.optim.AdamW([{"params":enc,"lr":1e-5},{"params":head,"lr":1e-3}],weight_decay=1e-4);gen=torch.Generator().manual_seed(a.seed+1900)
 deps={str(Path(x).resolve()):sha(Path(x)) for x in (__file__,R18.__file__,p2.__file__,sys.modules[load_decoupled_decoder.__module__].__file__)}
 identity={"schema":"round19_run_identity_v1","group":a.group,"fold":a.fold,"seed":a.seed,"data":str(a.data),"data_sha256":sha(a.data),"prefix_index":str(a.prefix_index),"prefix_index_sha256":sha(a.prefix_index),"source_sha256":sha(a.source),"visual_checkpoint_sha256":sha(a.visual_checkpoint),"source_sha256_code":sha(__file__),"dependency_hashes":deps,"protocol_sha256":sha(HERE/"PROTOCOL.md"),"micro_batch":BATCH,"accumulation":ACCUM,"effective_batch":EFFECTIVE_BATCH,"head_lr":1e-3,"encoder_lr":1e-5,"max_epochs":a.max_epochs,"patience":a.patience,"formal":a.formal}
 a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/"config.json",identity);atomic_json(a.output/"TRAIN_WEIGHTS.json",{"records":records})
 latest=a.output/"latest.pth";bestp=a.output/"best.pth";manifest=a.output/"COMMIT.json";(a.output/"commits").mkdir(exist_ok=True);start=0;best=-float("inf");best_epoch=-1;wait=0;history=[];steps=0;best_commit=None
 if manifest.exists():
  committed=json.loads(manifest.read_text());cp=a.output/committed["latest"]["path"]
  if sha(cp)!=committed["latest"]["sha256"]:raise ValueError("committed latest hash mismatch")
  ck=torch.load(cp,map_location="cpu",weights_only=False)
  if ck["identity"]!=identity or ck["initial_state_sha256"]!=state_hash(initial):raise ValueError("resume identity")
  model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"])
  for s in opt.state.values():
   for k,v in s.items():
    if torch.is_tensor(v):s[k]=v.to(a.device)
  gen.set_state(ck["generator"]);random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"]);torch.set_rng_state(ck["torch_rng"]);torch.cuda.set_rng_state_all(ck["cuda_rng"])
  start=ck["epoch"]+1;best=ck["best_metric"];best_epoch=ck["best_epoch"];wait=ck["wait"];history=ck["history"];steps=ck["optimizer_steps"];best_commit=committed["best"]
  atomic_save(latest,ck)
  if best_commit:
   bp=a.output/best_commit["path"]
   if sha(bp)!=best_commit["sha256"]:raise ValueError("committed best hash mismatch")
   atomic_save(bestp,torch.load(bp,map_location="cpu",weights_only=False))
 began=time.perf_counter();peak=0
 for epoch in range(start,a.max_epochs):
  if wait>=a.patience:break
  model.train();perm=primary[torch.randperm(len(primary),generator=gen)];losses=[]
  for effective_ids in perm.split(EFFECTIVE_BATCH):
   opt.zero_grad(set_to_none=True);effective_loss=0.
   for ids in effective_ids.split(BATCH):
    z,f=data.batch("fit",ids);f=(f-norm["mean"])/norm["std"];logit=model(z.to(a.device),f.to(a.device));loss=(nn.functional.binary_cross_entropy_with_logits(logit,target[ids].to(a.device),reduction="none")*weights[ids].to(a.device)).sum()/len(effective_ids)
    if not torch.isfinite(loss):raise FloatingPointError("loss")
    loss.backward();effective_loss+=float(loss.detach())
   losses.append(effective_loss)
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):raise FloatingPointError("gradient")
   opt.step();opt.zero_grad(set_to_none=True);steps+=1
  sel=data.d["roles"]["selection"];mask=sel["stage"].eq(0)|sel["stage"].eq(2);prob=infer(model,data,"selection",norm,a.device);metric=low_fpr_auc(sel["stage"][mask],prob[mask]);improved=metric>best
  if improved:best=metric;best_epoch=epoch;wait=0
  else:wait+=1
  history.append({"epoch":epoch,"fit_loss":float(np.mean(losses)),"selection_pAUC_0_0.1":metric,"selection_scores":prob.tolist()})
  ck={"identity":identity,"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"optimizer":opt.state_dict(),"generator":gen.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"normalizer":norm,"epoch":epoch,"global_step":steps,"best_metric":best,"best_epoch":best_epoch,"wait":wait,"history":history,"optimizer_steps":steps,"initial_state_sha256":state_hash(initial)}
  # The atomic manifest is the sole recovery pointer; latest/best are repairable convenience copies.
  commit=a.output/"commits"/f"epoch-{epoch:04d}.pth";atomic_save(commit,ck);entry={"path":str(commit.relative_to(a.output)),"sha256":sha(commit),"epoch":epoch}
  if improved:best_commit=entry
  atomic_json(manifest,{"schema":"round19_checkpoint_commit_v1","epoch":epoch,"latest":entry,"best":best_commit})
  atomic_save(latest,ck)
  if improved:atomic_save(bestp,ck)
  peak=max(peak,torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0);print(json.dumps({"epoch":epoch,"loss":history[-1]["fit_loss"],"pauc":metric,"best":best_epoch}),flush=True)
  if a.interrupt_after is not None and epoch>=a.interrupt_after:atomic_json(a.output/"summary.json",{"status":"interrupted","epoch":epoch,"identity":identity});return
 bestck=torch.load(bestp,map_location="cpu",weights_only=False);changes={k:not torch.equal(initial[k],v) for k,v in bestck["model"].items()};atomic_json(a.output/"summary.json",{"schema":"round19_training_summary_v1","status":"complete","group":a.group,"fold":a.fold,"seed":a.seed,"best_epoch":best_epoch,"best_metric":best,"epochs":len(history),"optimizer_steps":steps,"wall_seconds":time.perf_counter()-began,"peak_cuda_allocated":peak,"trainable_parameters":sum(p.numel() for p in trainable),"encoder_trainable_parameters":sum(p.numel() for p in enc),"head_trainable_parameters":sum(p.numel() for p in head),"changes_at_best":changes,"identity":identity})

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--visual-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--group",choices=GROUPS,required=True);p.add_argument("--fold",type=int,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--max-epochs",type=int,default=MAX_EPOCHS);p.add_argument("--patience",type=int,default=PATIENCE);p.add_argument("--interrupt-after",type=int);p.add_argument("--formal",action="store_true");train(p.parse_args())
