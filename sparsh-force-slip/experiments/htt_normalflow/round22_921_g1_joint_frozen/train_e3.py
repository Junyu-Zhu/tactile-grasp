#!/usr/bin/env python3
"""E3 A/B/C/D trainer built on the accepted R19 partial-encoder boundary."""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, importlib.util, json, os, random, sys, tempfile, time
from pathlib import Path
import numpy as np
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
R19_DIR=HERE.parent/"round19_htt_partial_encoder_finetuning"
sys.path[:0]=[str(R19_DIR),str(HERE.parent),str(REPO/"scripts"),str(HERE.parent/"round5_force_conditioned_slip"/"current")]
import phase2_b_multitask as p2
from sources import load_decoupled_decoder,load_r3_slip_branch

def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
R18=load_module("r22_r18",HERE.parent/"round18_htt_force_conditioned_film"/"train.py")

GROUPS=("A","B","C","D"); AUX_GROUPS=("B","D"); FORCE_INPUT_GROUPS=("C","D")
SEEDS=(20260914,20260915,20260916); ROLES=("fit","selection","calibration","validation")
BATCH=64; ACCUM=4; EFFECTIVE_BATCH=256; MAX_EPOCHS=60; PATIENCE=10

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
 try:Path(t).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def atomic_save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent,prefix=p.name+".",suffix=".tmp");os.close(fd)
 try:torch.save(x,t);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def acquire_task_lock(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);handle=(output/"TRAIN.lock").open("a+")
 try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:handle.close();raise RuntimeError("task output lock already owned")
 handle.seek(0);handle.truncate();handle.write(json.dumps({"pid":os.getpid(),"started_at":time.time()}));handle.flush();os.fsync(handle.fileno());return handle

class Model(nn.Module):
 def __init__(self,group,seed,source,r3):
  super().__init__();self.group=group;base,_=p2.load_b_checkpoint(source,torch.device("cpu"));e=base.encoder
  self.blocks=copy.deepcopy(e.blocks[10:]).requires_grad_(True);self.norm=copy.deepcopy(e.norm).requires_grad_(False)
  decoder,_=load_decoupled_decoder(source);branch=load_r3_slip_branch(decoder,r3)
  self.pooler=copy.deepcopy(branch.slip_pooler).requires_grad_(False);self.trunk=copy.deepcopy(branch.slip_trunk).requires_grad_(False)
  self.head=R18.init_model("M0" if group in FORCE_INPUT_GROUPS else "V0",seed)
  self.aux=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,3)) if group in AUX_GROUPS else None
  if self.aux is not None:
   torch.manual_seed(seed+2203)
   for p in self.aux.parameters(): nn.init.zeros_(p) if p.ndim==1 else nn.init.xavier_uniform_(p)
 def train(self,mode=True):
  super().train(mode);self.norm.eval();self.pooler.eval();self.trunk.eval();return self
 def visual(self,z):
  for b in self.blocks:z=b(z)
  z=self.norm(z)[:,1:]
  return self.trunk(self.pooler(z).squeeze(1))
 def components(self,z,force):
  n,h=z.shape[:2];v=self.visual(z.reshape(-1,*z.shape[2:])).reshape(n,h,192)
  x=torch.cat((v,force),-1) if self.group in FORCE_INPUT_GROUPS else torch.cat((v,torch.zeros_like(force)),-1)
  c=self.head.components(x);c["visual"]=v;c["aux_force"]=self.aux(v[:,-1]) if self.aux is not None else None;return c
 def forward(self,z,force):return self.components(z,force)["logit"]

class Data:
 def __init__(self,prepared,index):
  self.d=torch.load(prepared,map_location="cpu",weights_only=False);idx=json.loads(Path(index).read_text());self.paths={x["episode_id"]:x["path"] for x in idx["entries"]};self.mm={}
  if "test" in self.d["roles"]:raise ValueError("test forbidden")
 def batch(self,role,ids):
  r=self.d["roles"][role];zs=[]
  for i in ids.tolist():
   eid=r["episode_id"][i];t=int(r["t"][i]);
   if eid not in self.mm:self.mm[eid]=np.load(self.paths[eid],mmap_mode="r",allow_pickle=False)
   zs.append(np.array(self.mm[eid][t-8:t+1],copy=True))
  return torch.from_numpy(np.stack(zs)),r["x"][ids,:,192:195].clone(),r["y_current"][ids].clone()

def fit_assets(d):
 fit=d["roles"]["fit"];mask=fit["stage"].eq(0)|fit["stage"].eq(2);primary=torch.nonzero(mask,as_tuple=False).flatten();f=fit["x"][primary,:,192:195].double();gt=fit["y_current"][primary].double()
 norm={"force_mean":f.reshape(-1,3).mean(0).float(),"force_std":f.reshape(-1,3).std(0,unbiased=False).clamp_min(1e-6).float(),"aux_mean":gt.mean(0).float(),"aux_std":gt.std(0,unbiased=False).clamp_min(1e-6).float()}
 n=len(primary);w=torch.zeros(len(mask),dtype=torch.float64)
 for c in (0,2):
  ids=torch.nonzero(fit["stage"].eq(c)).flatten();eps=np.asarray(fit["episode_id"],object)[ids.numpy()]
  for ep in sorted(set(eps.tolist())):
   q=ids[torch.from_numpy(eps==ep)];w[q]=n/(2*len(set(eps.tolist()))*len(q))
 return primary,norm,w.float()
def low_fpr_auc(stage,probability,limit=.1):
 labels=(stage==2).numpy().astype(np.int64);scores=probability.numpy().astype(np.float64);order=np.argsort(-scores,kind="stable");labels=labels[order];scores=scores[order];ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1];tpr=np.r_[0.,np.cumsum(labels)[ends]/labels.sum()];fpr=np.r_[0.,np.cumsum(1-labels)[ends]/(len(labels)-labels.sum())];i=np.flatnonzero(fpr<limit)[-1];xs=fpr[fpr<=limit];ys=tpr[fpr<=limit]
 if not np.any(fpr==limit):q=(limit-fpr[i])/(fpr[i+1]-fpr[i]);xs=np.r_[fpr[:i+1],limit];ys=np.r_[tpr[:i+1],tpr[i]+q*(tpr[i+1]-tpr[i])]
 return float(np.trapezoid(ys,xs)/limit)
def infer(model,data,role,norm,device):
 out=[];model.eval()
 with torch.inference_mode():
  for ids in torch.arange(len(data.d["roles"][role]["t"])).split(BATCH):
   z,f,_=data.batch(role,ids);f=(f-norm["force_mean"])/norm["force_std"];out.append(torch.sigmoid(model(z.to(device),f.to(device))).cpu())
 return torch.cat(out)

def train(a):
 total_started=time.perf_counter();print(json.dumps({"event":"startup","group":a.group,"fold":a.fold,"seed":a.seed}),flush=True)
 if a.group not in GROUPS or a.seed not in SEEDS:raise ValueError("grid")
 if a.formal:
  if a.authorization is None or not a.authorization.is_file():raise ValueError("formal E3 blocked: missing root authorization")
  authorization=json.loads(a.authorization.read_text())
  if authorization.get("e3_formal_authorized") is not True or authorization.get("budget_pass") is not True:raise ValueError("formal E3 blocked by root authorization or budget")
  if a.protocol is None or not a.protocol.is_file():raise ValueError("formal E3 blocked: missing locked protocol")
 task_lock=acquire_task_lock(a.output)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 data=Data(a.data,a.prefix_index);primary,norm,weights=fit_assets(data.d);target=data.d["roles"]["fit"]["stage"].eq(2).float();print(json.dumps({"event":"data_ready","elapsed_seconds":time.perf_counter()-total_started,"fit_primary":len(primary)}),flush=True)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 model=Model(a.group,a.seed,a.source,a.visual_checkpoint).to(a.device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};print(json.dumps({"event":"model_ready","elapsed_seconds":time.perf_counter()-total_started}),flush=True)
 enc=list(model.blocks.parameters());head=list(model.head.parameters())+([] if model.aux is None else list(model.aux.parameters()));trainable=enc+head
 opt=torch.optim.AdamW([{"params":enc,"lr":1e-5},{"params":head,"lr":1e-3}],weight_decay=1e-4);gen=torch.Generator().manual_seed(a.seed+2200)
 norm_sha=hashlib.sha256(b"".join(norm[k].numpy().tobytes() for k in sorted(norm))).hexdigest();deps={"r18_train":sha(R18.__file__),"phase2_b_multitask":sha(p2.__file__),"sources":sha(sys.modules[load_decoupled_decoder.__module__].__file__)}
 identity={"schema":"round22_e3_identity_v2","group":a.group,"fold":a.fold,"seed":a.seed,"data_sha256":sha(a.data),"prefix_index_sha256":sha(a.prefix_index),"source_sha256":sha(a.source),"visual_checkpoint_sha256":sha(a.visual_checkpoint),"source_sha256_code":sha(__file__),"dependency_hashes":deps,"normalizer_sha256":norm_sha,"micro_batch":BATCH,"accumulation":ACCUM,"effective_batch":EFFECTIVE_BATCH,"head_lr":1e-3,"encoder_lr":1e-5,"aux_weight":.1,"aux_reduction":"mean_over_endpoints_and_three_axes","max_epochs":a.max_epochs,"patience":a.patience,"formal":a.formal,"authorization_sha256":sha(a.authorization) if a.formal else None,"protocol_sha256":sha(a.protocol) if a.formal else None}
 a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/"config.json",identity);latest=a.output/"latest.pth";bestp=a.output/"best.pth";manifest=a.output/"COMMIT.json";(a.output/"commits").mkdir(exist_ok=True)
 start=0;best=-float("inf");best_epoch=-1;wait=0;history=[];steps=0;best_commit=None
 if manifest.exists():
  committed=json.loads(manifest.read_text());cp=a.output/committed["latest"]["path"]
  if sha(cp)!=committed["latest"]["sha256"]:raise ValueError("committed latest hash mismatch")
  ck=torch.load(cp,map_location="cpu",weights_only=False)
  if ck["identity"]!=identity or ck["initial_state_sha256"]!=state_hash(initial):raise ValueError("resume identity mismatch")
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
  model.train();perm=primary[torch.randperm(len(primary),generator=gen)];slips=[];auxes=[];totals=[]
  train_started=time.perf_counter()
  for effective_number,effective_ids in enumerate(perm.split(EFFECTIVE_BATCH)):
   opt.zero_grad(set_to_none=True);sl=0.;au=0.
   for ids in effective_ids.split(BATCH):
    z,f,gt=data.batch("fit",ids);f=(f-norm["force_mean"])/norm["force_std"];c=model.components(z.to(a.device),f.to(a.device));sloss=(nn.functional.binary_cross_entropy_with_logits(c["logit"],target[ids].to(a.device),reduction="none")*weights[ids].to(a.device)).sum()/len(effective_ids);loss=sloss;aloss=torch.zeros((),device=a.device)
    if c["aux_force"] is not None:
     agt=(gt.to(a.device)-norm["aux_mean"].to(a.device))/norm["aux_std"].to(a.device);aloss=nn.functional.smooth_l1_loss(c["aux_force"],agt,beta=1,reduction="mean")*(len(ids)/len(effective_ids));loss=loss+.1*aloss
    if not torch.isfinite(loss):raise FloatingPointError("loss")
    loss.backward();sl+=float(sloss.detach());au+=float(aloss.detach())
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):raise FloatingPointError("gradient")
   opt.step();steps+=1;slips.append(sl);auxes.append(au);totals.append(sl+.1*au)
   if effective_number==0:print(json.dumps({"event":"first_optimizer_step","epoch":epoch,"elapsed_seconds":time.perf_counter()-total_started,"step_seconds":time.perf_counter()-train_started}),flush=True)
  train_seconds=time.perf_counter()-train_started;selection_started=time.perf_counter()
  sel=data.d["roles"]["selection"];mask=sel["stage"].eq(0)|sel["stage"].eq(2);prob=infer(model,data,"selection",norm,a.device);metric=low_fpr_auc(sel["stage"][mask],prob[mask]);improved=metric>best
  if improved:best=metric;best_epoch=epoch;wait=0
  else:wait+=1
  selection_seconds=time.perf_counter()-selection_started;history.append({"epoch":epoch,"fit_slip_loss":float(np.mean(slips)),"fit_force_aux_loss":float(np.mean(auxes)),"fit_total_loss":float(np.mean(totals)),"selection_pAUC_0_0.1":metric,"train_seconds":train_seconds,"selection_seconds":selection_seconds})
  ck={"identity":identity,"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"optimizer":opt.state_dict(),"generator":gen.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"normalizer":norm,"epoch":epoch,"best_metric":best,"best_epoch":best_epoch,"wait":wait,"history":history,"optimizer_steps":steps,"initial_state_sha256":state_hash(initial)}
  commit=a.output/"commits"/f"epoch-{epoch:04d}.pth";atomic_save(commit,ck);entry={"path":str(commit.relative_to(a.output)),"sha256":sha(commit),"epoch":epoch}
  if improved:best_commit=entry
  atomic_json(a.output/"COMMIT.json",{"schema":"round22_checkpoint_commit_v1","epoch":epoch,"latest":entry,"best":best_commit});atomic_save(latest,ck)
  if improved:atomic_save(bestp,ck)
  checkpoint_seconds=time.perf_counter()-selection_started-selection_seconds;peak=max(peak,torch.cuda.max_memory_allocated());print(json.dumps({"event":"epoch_complete","epoch":epoch,"seconds":time.perf_counter()-began,"checkpoint_seconds":checkpoint_seconds,**history[-1]}),flush=True)
 committed=json.loads(manifest.read_text());best_entry=committed["best"];best_source=a.output/best_entry["path"]
 if sha(best_source)!=best_entry["sha256"]:raise ValueError("final committed best hash mismatch")
 bestck=torch.load(best_source,map_location="cpu",weights_only=False);atomic_save(bestp,bestck);changes={k:not torch.equal(initial[k],v) for k,v in bestck["model"].items()}
 result={"schema":"round22_e3_summary_v1","status":"complete","group":a.group,"fold":a.fold,"seed":a.seed,"best_epoch":best_epoch,"best_metric":best,"epochs":len(history),"optimizer_steps":steps,"wall_seconds":time.perf_counter()-began,"peak_cuda_allocated":peak,"trainable_parameters":sum(p.numel() for p in trainable),"encoder_trainable_parameters":sum(p.numel() for p in enc),"auxiliary_parameters":0 if model.aux is None else sum(p.numel() for p in model.aux.parameters()),"changes_at_best":changes,"identity":identity};atomic_json(a.output/"summary.json",result);return result

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--visual-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--group",choices=GROUPS,required=True);p.add_argument("--fold",type=int,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--max-epochs",type=int,default=MAX_EPOCHS);p.add_argument("--patience",type=int,default=PATIENCE);p.add_argument("--formal",action="store_true");p.add_argument("--authorization",type=Path);p.add_argument("--protocol",type=Path);print(json.dumps(train(p.parse_args())))
