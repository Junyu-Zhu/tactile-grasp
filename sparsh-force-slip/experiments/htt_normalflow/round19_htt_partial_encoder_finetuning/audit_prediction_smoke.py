#!/usr/bin/env python3
"""Independent direct-score replay for one exported formal prediction artifact."""
import argparse,json
from pathlib import Path
import numpy as np,torch
import train as tr
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--prefix-index",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--visual-checkpoint",type=Path,required=True);p.add_argument("--run",type=Path,required=True);p.add_argument("--prediction",type=Path,required=True);p.add_argument("--group",choices=tr.GROUPS,required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 ps=json.loads((a.prediction/"SUMMARY.json").read_text());arr=np.load(a.prediction/"PREDICTIONS.npz",allow_pickle=False);manifest=json.loads((a.run/"COMMIT.json").read_text());best=a.run/manifest["best"]["path"];ck=torch.load(best,map_location="cpu",weights_only=False);data=tr.Data(a.prepared,a.prefix_index);model=tr.Model(a.group,a.seed,a.source,a.visual_checkpoint);model.load_state_dict(ck["model"]);model.to(a.device).eval();norm=ck["normalizer"];maxerr=0.;rows={}
 with torch.inference_mode():
  for role in tr.ROLES:
   score=arr[f"{role}_score"]
   if len(score)!=len(data.d["roles"][role]["t"]) or not np.isfinite(score).all() or score.min()<0 or score.max()>1:raise ValueError("score support")
   ids=torch.tensor(sorted(set((0,len(score)-1))));z,f=data.batch(role,ids);f=(f-norm["mean"])/norm["std"];direct=torch.sigmoid(model(z.to(a.device),f.to(a.device))).cpu().numpy();err=float(np.max(np.abs(direct-score[ids.numpy()])));maxerr=max(maxerr,err);rows[role]={"rows":len(score),"min":float(score.min()),"max":float(score.max()),"direct_replay_max_abs":err}
 checks={"summary_complete":ps["status"]=="complete","prediction_hash":tr.sha(a.prediction/"PREDICTIONS.npz")==ps["prediction_sha256"],"best_hash":tr.sha(best)==ps["best_commit_sha256"],"four_roles":set(rows)==set(tr.ROLES),"direct_replay_within_fp32_batch_tolerance":maxerr<=2e-6,"test_not_consumed":ps["test_consumed"] is False}
 result={"schema":"round19_prediction_smoke_audit_v1","status":"pass" if all(checks.values()) else "fail","checks":checks,"roles":rows,"direct_replay_max_abs":maxerr,"test_consumed":False};tr.atomic_json(a.output,result);print(json.dumps(result,indent=2));assert result["status"]=="pass"
if __name__=="__main__":main()
