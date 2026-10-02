#!/usr/bin/env python3
"""Preregistered offline shared-encoder gradient diagnostic for one B/D run."""
from __future__ import annotations
import argparse,importlib.util,json,math,random,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;R22=HERE.parent/"round22_921_g1_joint_frozen"
def load_train():
 s=importlib.util.spec_from_file_location("r23_grad_train",R22/"train_e3.py");m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
tr=load_train()
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--ids",type=Path,required=True);p.add_argument("--run",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();inv=json.loads(a.inventory.read_text());rr=next(x for x in inv["runs"] if x["run"]==a.run)
 if rr["group"] not in tr.AUX_GROUPS:raise ValueError("gradient diagnostic only B/D")
 fixed=json.loads(a.ids.read_text());ds=next(x for x in fixed["datasets"] if x["fold"]==rr["fold"] and x["seed"]==rr["seed"])
 if ds["data_sha256"]!=rr["data_sha256"] or ds["batch_size"]!=64:raise ValueError("diagnostic ID identity")
 indices=torch.tensor([x["fit_index"] for x in ds["endpoints"]],dtype=torch.long);stages=["initialization","epoch_0","epoch_9"];run=Path(rr["output"]);commit=json.loads((run/"COMMIT.json").read_text());best=run/commit["best"]["path"];bestck=torch.load(best,map_location="cpu",weights_only=False);norm=bestck["normalizer"]
 data=tr.Data(rr["data"],rr["prefix_index"]);_,_,weights=tr.fit_assets(data.d);fit=data.d["roles"]["fit"];target=fit["stage"].eq(2).float();z,f,gt=data.batch("fit",indices);f=(f-norm["force_mean"])/norm["force_std"]
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 rows=[]
 for stage in stages:
  random.seed(rr["seed"]);np.random.seed(rr["seed"]);torch.manual_seed(rr["seed"]);torch.cuda.manual_seed_all(rr["seed"]);model=tr.Model(rr["group"],rr["seed"],rr["source"],rr["visual_checkpoint"])
  cp=None
  if stage!="initialization":
   epoch=0 if stage=="epoch_0" else 9;cp=run/"commits"/f"epoch-{epoch:04d}.pth"
   if not cp.exists():rows.append({"run":a.run,"group":rr["group"],"fold":rr["fold"],"seed":rr["seed"],"position":stage,"status":"unavailable","reason":"preregistered checkpoint absent; no substitution"});continue
   model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=False)["model"],strict=True)
  model.to(a.device).train();params=list(model.blocks.parameters());c=model.components(z.to(a.device),f.to(a.device));sl=(torch.nn.functional.binary_cross_entropy_with_logits(c["logit"],target[indices].to(a.device),reduction="none")*weights[indices].to(a.device)).sum()/64
  agt=(gt.to(a.device)-norm["aux_mean"].to(a.device))/norm["aux_std"].to(a.device);aux=torch.nn.functional.smooth_l1_loss(c["aux_force"],agt,beta=1,reduction="mean");gs=torch.autograd.grad(sl,params,retain_graph=True,allow_unused=True);ga=torch.autograd.grad(.1*aux,params,allow_unused=True)
  sv=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)]);av=torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,ga)]);sn=float(sv.norm());an=float(av.norm());cos=float(torch.dot(sv,av)/(sv.norm()*av.norm())) if sn and an else None
  rows.append({"run":a.run,"group":rr["group"],"fold":rr["fold"],"seed":rr["seed"],"position":stage,"status":"available","model_mode":"formal_train_mode; frozen norm/pooler/trunk remain eval by Model.train override","checkpoint":str(cp) if cp else None,"checkpoint_sha256":tr.sha(cp) if cp else None,"batch_ids_sha256":tr.sha(a.ids),"slip_loss":float(sl),"force_aux_loss_unweighted":float(aux),"grad_slip_l2":sn,"grad_0p1_force_aux_l2":an,"cosine":cos,"slip_finite":math.isfinite(sn),"aux_finite":math.isfinite(an),"cosine_finite":cos is not None and math.isfinite(cos),"slip_zero_norm":sn==0,"aux_zero_norm":an==0})
  del model,c,gs,ga,sv,av,sl,aux;torch.cuda.empty_cache()
 out={"schema":"round23_e3_gradient_diagnostic_v1","status":"complete","run":a.run,"rows":rows,"offline_no_optimizer_step":True,"test_consumed":False};tr.atomic_json(a.output,out);print(json.dumps(out))
if __name__=="__main__":main()
