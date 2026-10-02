#!/usr/bin/env python3
"""Verify the inference backend against bound unperturbed formal predictions."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os
from pathlib import Path
import numpy as np,torch
GROUPS=("B_force","C_force_delta");SEEDS=(20260914,20260915,20260916);ROLES=("selection","calibration","outer")
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(8*1024*1024),b""):h.update(c)
 return h.hexdigest()
def module(path):
 s=importlib.util.spec_from_file_location("bound_train_verify",path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def atomic(path,v):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--training-source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--rows",type=int,default=32);a=p.parse_args()
 tr=module(a.training_source);payload=torch.load(a.prepared,map_location="cpu",weights_only=False);audit=tr.validate_prepared(payload,a.prepared);hs=audit["horizons"];checks=[];sources={str(a.prepared.resolve()):sha256(a.prepared),str(a.training_source.resolve()):sha256(a.training_source),str(Path(__file__).resolve()):sha256(Path(__file__).resolve())}
 for g in GROUPS:
  for seed in SEEDS:
   folder=a.formal_root/f"future_{g}_{seed}";summary_path=folder/"summary.json";summary=json.loads(summary_path.read_text());sources[str(summary_path.resolve())]=sha256(summary_path);best=Path(summary["artifacts"]["best"]["path"]);sources[str(best.resolve())]=sha256(best)
   model=tr.MultiWindowGRU(776,128,len(hs));model.load_state_dict(torch.load(best,map_location="cpu",weights_only=False)["model_state"],strict=True);model.eval()
   for role in ROLES:
    source=payload["timelines"][role];n=min(a.rows,len(source["t"]));subset={k:(v[:n] if torch.is_tensor(v) or isinstance(v,(list,tuple)) else v) for k,v in source.items()};x=tr.assemble_inputs(payload,subset,g);prob=tr.predict(model,x,"cpu",n)
    pred_path=Path(summary["artifacts"]["predictions"][role]["timeline"]["path"]);sources[str(pred_path.resolve())]=sha256(pred_path);stored=list(csv.DictReader(pred_path.open()))[:n];expected=np.asarray([[float(r[f"p_future_H{h}_raw"]) for h in hs] for r in stored]);error=float(np.max(np.abs(prob-expected)))
    checks.append({"group":g,"seed":seed,"role":role,"rows":n,"max_abs_error":error,"pass":error<=2e-5})
 result={"schema":"round7_sensitivity_backend_equivalence_v1","status":"pass" if all(x["pass"] for x in checks) else "fail","tolerance":2e-5,"backend":"CPU checkpoint replay against formal timeline probabilities","checks":checks,"source_hashes":sources};atomic(a.output,result);print(json.dumps({"status":result["status"],"max_abs_error":max(x["max_abs_error"] for x in checks)}))
 if result["status"]!="pass":raise SystemExit(1)
if __name__=="__main__":main()
