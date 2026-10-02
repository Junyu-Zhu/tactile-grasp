#!/usr/bin/env python3
"""Finalize the mixed-method prediction grid without invalidating 46 accepted exports."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256(Path(p).read_bytes());return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--prediction-root",type=Path,required=True);a=p.parse_args();inv=json.loads(a.inventory.read_text());predictor=sha(HERE/"predict_e3.py");repair=sha(HERE/"repair_prediction_direct.py");rows=[];issues=[]
 for r in inv["runs"]:
  out=a.prediction_root/r["group"]/f"p{r['fold']}_s{r['seed']}"
  try:
   s=json.loads((out/"SUMMARY.json").read_text());direct=r["run"]=="D/p4_s20260915";valid=s.get("status")=="complete" and s.get("run")==r["run"] and sha(out/"PREDICTIONS.npz")==s.get("prediction_sha256") and s.get("source_hashes",{}).get("inventory")==sha(a.inventory) and s.get("source_hashes",{}).get("predictor")==predictor and s.get("selection_replay_abs_error",1)<=2e-6 and ((not direct and s.get("schema")=="round23_e3_prediction_v1") or (direct and s.get("schema")=="round23_e3_prediction_v1_direct_repair" and s.get("source_hashes",{}).get("direct_repair")==repair))
   if not valid:issues.append(r["run"])
   rows.append({"run":r["run"],"valid":valid,"method":s.get("prediction_method","deduplicated_visual"),"prediction_sha256":s.get("prediction_sha256"),"selection_replay_abs_error":s.get("selection_replay_abs_error")})
  except Exception as e:issues.append(f"{r['run']}: {e}")
 out={"schema":"round23_e3_prediction_status_v2","status":"complete" if not issues and len(rows)==48 else "incomplete","complete":sum(x["valid"] for x in rows),"runs":48,"methods":{"deduplicated_visual":47,"direct_repeated_window_exception":1},"exception_run":"D/p4_s20260915","predictor_sha256":predictor,"direct_repair_sha256":repair,"issues":issues,"rows":rows,"test_consumed":False};(HERE/"PREDICTION_STATUS.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out));raise SystemExit(0 if out["status"]=="complete" else 1)
if __name__=="__main__":main()
