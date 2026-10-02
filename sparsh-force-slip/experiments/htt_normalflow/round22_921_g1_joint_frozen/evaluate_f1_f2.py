#!/usr/bin/env python3
"""Evaluate accepted F1 predictions and fixed F2 0.5 shrink without retraining."""
from __future__ import annotations
import argparse,hashlib,json,tempfile,os
from pathlib import Path
import torch
import train_f1 as F1
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:Path(t).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def features(r,group):
 z=r["x"][:,-1]
 return z[:,:192] if group=="K-V" else z[:,192:195] if group=="K-F" else z
def ridge_fit(r,group,alpha=1e-6):
 x=features(r,group).double();y=F1.delta(r).reshape(len(x),-1).double();xm=x.mean(0);ym=y.mean(0);xc=x-xm;w=torch.linalg.solve(xc.T@xc+alpha*torch.eye(x.shape[1],dtype=x.dtype),xc.T@(y-ym));return {"weight":w,"x_mean":xm,"y_mean":ym,"alpha":alpha}
def ridge_predict(r,group,m):return ((features(r,group).double()-m["x_mean"])@m["weight"]+m["y_mean"]).float().reshape(-1,3,3)
def metrics(r,pred):
 truth=F1.delta(r);anchor=r["x"][:,-1,192:195];stratum10=truth[:,2].abs().amax(1);masks={"all":torch.ones(len(truth),dtype=torch.bool),"stable_10frame_max_axis_le_0.25N":stratum10<=.25,"transition_10frame":(stratum10>.25)&(stratum10<1),"changing_10frame_max_axis_ge_1N":stratum10>=1};out={}
 for name,mask in masks.items():
  if not mask.any():out[name]={"endpoints":0};continue
  err=(pred[mask]-truth[mask]).abs();absolute=(anchor[mask,None,:]+pred[mask]-r["y"][mask]).abs();out[name]={"endpoints":int(mask.sum()),"delta_mae_n":float(err.mean()),"absolute_future_mae_n":float(absolute.mean()),"predicted_delta_abs_mean_n":float(pred[mask].abs().mean())}
 out["per_horizon_delta_mae_n"]={str(h):float((pred[:,i]-truth[:,i]).abs().mean()) for i,h in enumerate(F1.HORIZONS)};return out
def endpoint_hash(r):
 h=hashlib.sha256()
 for e,t,g in zip(r["episode_id"],r["t"].tolist(),r["leakage_group"]):h.update(f"{e}\0{int(t)}\0{g}\n".encode())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu");a=p.parse_args();summary=json.loads((a.run/"summary.json").read_text());identity=summary["identity"]
 if summary.get("status")!="complete" or identity.get("data_sha256")!=sha(a.data) or identity.get("source_sha256")!=sha(Path(F1.__file__)):raise ValueError("unaccepted F1 identity")
 d=torch.load(a.data,map_location="cpu",weights_only=False);F1.validate(d,a.data,summary["fold"],summary["seed"],Path(F1.__file__).with_name("G1_PREPARE.json"));cm=json.loads((a.run/"COMMIT.json").read_text());bp=a.run/cm["best"]["path"]
 if sha(bp)!=cm["best"]["sha256"]:raise ValueError("best commit hash")
 ck=torch.load(bp,map_location="cpu",weights_only=False)
 if ck.get("identity")!=identity:raise ValueError("checkpoint/summary identity mismatch")
 model=F1.init_model(summary["group"],summary["seed"]);model.load_state_dict(ck["model"]);model.to(a.device);n=ck["normalizer"];ridge=ridge_fit(d["roles"]["fit"],summary["group"]);predictions={};results={}
 for role,r in d["roles"].items():
  x=F1.normalize_x(r["x"],n,summary["group"]);raw=F1.predict(model,x,n,a.device);hold=torch.zeros_like(raw);rp=ridge_predict(r,summary["group"],ridge);variants={"F1_raw":raw,"hold_delta_zero":hold,"matched_current_ridge_1e-6":rp}
  if summary["group"]=="K-VF":variants["F2_fixed_half_shrink"]=.5*raw
  predictions[role]={k:v for k,v in variants.items()};results[role]={k:metrics(r,v) for k,v in variants.items()}
 a.output.mkdir(parents=True,exist_ok=False);endpoint_hashes={role:endpoint_hash(r) for role,r in d["roles"].items()};torch.save({"schema":"round22_f1_f2_predictions_v1","source_run":str(a.run),"source_best_sha256":sha(bp),"data_sha256":sha(a.data),"source_trainer_sha256":sha(F1.__file__),"evaluator_sha256":sha(__file__),"endpoint_hashes":endpoint_hashes,"group":summary["group"],"predictions":predictions},a.output/"predictions.pt");report={"schema":"round22_f1_f2_evaluation_v1","status":"entrypoint_smoke_complete","scope":"means/window strata only; full formal evaluation still requires per-axis RMSE, per-trial and paired CI, error-cancellation, and case tables","group":summary["group"],"fold":summary["fold"],"seed":summary["seed"],"data_sha256":sha(a.data),"source_trainer_sha256":sha(F1.__file__),"evaluator_sha256":sha(__file__),"endpoint_hashes":endpoint_hashes,"f2_applicable":summary["group"]=="K-VF","f2_coefficient":.5 if summary["group"]=="K-VF" else None,"f2_selected_without_validation":True if summary["group"]=="K-VF" else None,"ridge_alpha":1e-6,"ridge_rule":"raw current input with centered closed-form intercept; no validation tuning","dynamic_input":identity["group"],"absolute_anchor_for_all_groups":"shared frozen predicted current force x[t,192:195]","results":results,"predictions_sha256":sha(a.output/"predictions.pt"),"source_summary_sha256":sha(a.run/"summary.json")};atomic_json(a.output/"evaluation.json",report);print(json.dumps(report))
if __name__=="__main__":main()
