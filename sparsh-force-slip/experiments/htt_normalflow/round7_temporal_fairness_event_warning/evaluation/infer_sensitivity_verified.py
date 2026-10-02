#!/usr/bin/env python3
"""Frozen-checkpoint Round-7 force-input sensitivity inference."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os
from pathlib import Path
import numpy as np
import torch

GROUPS=("B_force","C_force_delta");SEEDS=(20260914,20260915,20260916);ROLES=("selection","calibration","outer")

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
 return h.hexdigest()

def atomic_json(path:Path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)

def load_module(path:Path):
 spec=importlib.util.spec_from_file_location("round7_train_bound",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def intervention(x:torch.Tensor,group:str,name:str)->torch.Tensor:
 result=x.clone();cols=list(range(769,772))+(list(range(772,775)) if group=="C_force_delta" else [])
 if name=="zero_fit_mean":result[:,:,cols]=0
 elif name=="causal_lag1":
  old=result[:,:,cols].clone();result[:,1:,cols]=old[:,:-1];result[:,0,cols]=0
  if group=="C_force_delta":
   valid=result[:,:,775].clone();result[:,1:,775]=valid[:,:-1];result[:,0,775]=0
 else:raise ValueError(name)
 if not torch.isfinite(result).all():raise FloatingPointError(name)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--training-source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu");p.add_argument("--batch-size",type=int,default=256);a=p.parse_args()
 train=load_module(a.training_source.resolve());torch.use_deterministic_algorithms(True)
 if a.device.startswith("cuda"):
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 payload=torch.load(a.prepared,map_location="cpu",weights_only=False);audit=train.validate_prepared(payload,a.prepared);horizons=audit["horizons"]
 sources={str(a.prepared.resolve()):sha256(a.prepared),str(a.training_source.resolve()):sha256(a.training_source),str(Path(__file__).resolve()):sha256(Path(__file__).resolve())};artifacts=[];backend_checks=[]
 for group in GROUPS:
  for seed in SEEDS:
   folder=a.formal_root/f"future_{group}_{seed}";summary_path=folder/"summary.json";summary=json.loads(summary_path.read_text());sources[str(summary_path.resolve())]=sha256(summary_path)
   if summary.get("status")!="complete" or not summary.get("formal") or summary.get("group")!=group or int(summary.get("seed"))!=seed:raise ValueError(f"invalid parent {folder}")
   best=Path(summary["artifacts"]["best"]["path"])
   if sha256(best)!=summary["artifacts"]["best"]["sha256"]:raise ValueError(f"checkpoint hash {folder}")
   sources[str(best.resolve())]=sha256(best);checkpoint=torch.load(best,map_location="cpu",weights_only=False)
   model=train.MultiWindowGRU(776,128,len(horizons));model.load_state_dict(checkpoint["model_state"],strict=True);model.to(a.device).eval()
   for role in ROLES:
    source=payload["timelines"][role];x=train.assemble_inputs(payload,source,group)
    check_n=min(32,len(x));unperturbed=train.predict(model,x[:check_n],a.device,check_n);formal_path=Path(summary["artifacts"]["predictions"][role]["timeline"]["path"]);formal=list(csv.DictReader(formal_path.open()))[:check_n];expected=np.asarray([[float(r[f"p_future_H{h}_raw"]) for h in horizons] for r in formal]);error=float(np.max(np.abs(unperturbed-expected)));backend_checks.append({"group":group,"seed":seed,"role":role,"rows":check_n,"max_abs_error":error,"pass":error<=2e-5})
    if error>2e-5:raise RuntimeError(f"unperturbed backend mismatch {group}/{seed}/{role}: {error}")
    for name in ("zero_fit_mean","causal_lag1"):
     altered=intervention(x,group,name);prob=train.predict(model,altered,a.device,a.batch_size)
     path=a.output/name/f"future_{group}_{seed}"/f"predictions_timeline_{role}.csv";path.parent.mkdir(parents=True,exist_ok=True);spec=train.atomic_predictions(path,source,prob,horizons,timeline=True)
     artifacts.append({"intervention":name,"group":group,"seed":seed,"role":role,**spec})
 manifest={"schema":"round7_force_input_sensitivity_v1","status":"complete","formal_round7_run":False,"descriptive_intervention_only":True,"interventions":{"zero_fit_mean":"force-derived normalized values set to fit-mean zero; validity mask retained","causal_lag1":"force-derived values shifted one causal history slot; first slot fit-mean zero; C auxiliary validity shifted and first invalid"},"groups":list(GROUPS),"seeds":list(SEEDS),"roles":list(ROLES),"horizons":horizons,"deterministic_backend":{"use_deterministic_algorithms":True,"tf32":False,"cudnn_benchmark":False,"unperturbed_tolerance":2e-5,"checks":backend_checks},"source_hashes":sources,"artifacts":artifacts}
 atomic_json(a.output/"SENSITIVITY_MANIFEST.json",manifest);print(json.dumps({"status":"complete","artifacts":len(artifacts)}))
if __name__=="__main__":main()
