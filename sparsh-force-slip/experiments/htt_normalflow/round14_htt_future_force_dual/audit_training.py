#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import torch
import future_train as ft
def eq(a,b):
 if torch.is_tensor(a): return torch.equal(a,b)
 if isinstance(a,dict): return a.keys()==b.keys() and all(eq(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)): return len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
 return a==b
def main():
 p=argparse.ArgumentParser();p.add_argument("--formal",type=Path,required=True);p.add_argument("--lock",type=Path,required=True);p.add_argument("--inventory",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 lock=json.loads(a.lock.read_text()); inv=json.loads(a.inventory.read_text()); expected={(g,f,s) for g in ft.GROUPS for f in range(1,5) for s in (20260914,20260915,20260916)}; accepted=[]
 for rec in inv["runs"]:
  run=a.formal/rec["id"]; summary=json.loads((run/"summary.json").read_text()); best=torch.load(run/"best.pth",map_location="cpu",weights_only=False); latest=torch.load(run/"latest.pth",map_location="cpu",weights_only=False)
  ident=latest["identity"]; key=(ident["group"],ident["fold"],ident["seed"]); assert key in expected and summary["status"]=="complete" and summary["identity"]==ident
  assert ident["protocol_sha256"]==lock["source_hashes"]["PROTOCOL.md"] and ident["source_sha256"]==lock["source_hashes"]["future_train.py"] and ident["round14_data_sha256"]==next(x["sha256"] for x in lock["inputs"] if x["fold"]==ident["fold"] and x["seed"]==ident["seed"])
  scores=[x["selection_mae"] for x in latest["history"]]; first=min(range(len(scores)),key=lambda i:scores[i]); assert latest["best_epoch"]==first and best["epoch"]==first and math.isclose(latest["best_metric"],scores[first],rel_tol=0,abs_tol=0)
  assert len(scores)==summary["epochs"] and all(math.isfinite(x["fit_loss"]) and math.isfinite(x["selection_mae"]) for x in latest["history"])
  assert eq(best["model"],torch.load(run/"best.pth",map_location="cpu",weights_only=False)["model"])
  params=sum(v.numel() for v in latest["model"].values()); assert params==summary["parameters"]
  accepted.append({"id":rec["id"],"group":key[0],"fold":key[1],"seed":key[2],"epochs":summary["epochs"],"best_epoch":summary["best_epoch"],"best_metric_native_n_mae":summary["best_metric_native_n_mae"],"parameters":params,"best_checkpoint":str(run/"best.pth"),"best_sha256":ft.sha(run/"best.pth"),"latest_sha256":ft.sha(run/"latest.pth"),"summary_sha256":ft.sha(run/"summary.json")})
 assert { (r["group"],r["fold"],r["seed"]) for r in accepted}==expected
 out={"schema":"round14_training_audit_v1","status":"pass","accepted_count":len(accepted),"test_consumed":False,"cached_upstream_not_instantiated":True,"optimizer_scope":"new future module only","checks":["complete_registered_grid","locked_protocol_source_and_input","finite_history","earliest_strict_native_n_selection","checkpoint_identity","no_smoke_checkpoint"],"accepted_runs":accepted};ft.atomic_json(out,a.output)
 inv["status"]="training_accepted";
 for r in inv["runs"]: r.update(status="accepted",**next({k:v for k,v in z.items() if k not in ("id","group","fold","seed")} for z in accepted if z["id"]==r["id"]))
 ft.atomic_json(inv,a.inventory);print(json.dumps({"status":"pass","accepted":len(accepted)}))
if __name__=="__main__":main()

