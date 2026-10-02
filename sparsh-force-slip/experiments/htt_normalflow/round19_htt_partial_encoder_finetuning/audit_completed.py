#!/usr/bin/env python3
"""Incremental read-only audit for every run currently marked complete."""
import argparse,json
from pathlib import Path
import torch
import train as tr
def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--large",type=Path,required=True);p.add_argument("--r17",type=Path,required=True);p.add_argument("--r3",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();formal=json.loads((a.local/"FORMAL_STATUS.json").read_text());targets=[r for r in formal["runs"] if r["status"]=="complete"];rows=[];fail=[]
 for rec in targets:
  group,tail=rec["run"].split("/");fold=int(tail.split("_")[0][1:]);seed=int(tail.split("_s")[1]);dp=a.r17/f"p{fold}_s{seed}"/"prepared.pt";data=torch.load(dp,map_location="cpu",weights_only=False);sel=data["roles"]["selection"];mask=sel["stage"].eq(0)|sel["stage"].eq(2);run=a.large/"formal"/group/tail
  try:
   summary=json.loads((run/"summary.json").read_text());manifest=json.loads((run/"COMMIT.json").read_text());best=run/manifest["best"]["path"]
   if tr.sha(run/"summary.json")!=rec["summary_sha256"] or tr.sha(best)!=manifest["best"]["sha256"]:raise ValueError("accepted hashes")
   latest=run/manifest["latest"]["path"]
   if tr.sha(latest)!=manifest["latest"]["sha256"]:raise ValueError("latest hash")
   ck=torch.load(best,map_location="cpu",weights_only=False);last=torch.load(latest,map_location="cpu",weights_only=False);hist=last["history"];metrics=[tr.low_fpr_auc(sel["stage"][mask],torch.tensor(x["selection_scores"])[mask]) for x in hist];best_epoch=max(range(len(metrics)),key=lambda i:metrics[i])
   if last["best_epoch"]!=summary["best_epoch"] or best_epoch!=last["best_epoch"] or manifest["best"]["epoch"]!=best_epoch or summary["epochs"]!=len(hist) or max(abs(x-y["selection_pAUC_0_0.1"]) for x,y in zip(metrics,hist))>1e-12:raise ValueError("selection replay")
   initial=tr.Model(group,seed,a.source,a.r3/f"fold_p{fold}"/f"seed_{seed}"/"best.pth").state_dict()
   if tr.state_hash(initial)!=ck["initial_state_sha256"]:raise ValueError("initial hash")
   changes={k:not torch.equal(initial[k],v) for k,v in ck["model"].items()};bad=[k for k,v in changes.items() if v and not (k.startswith("blocks.") or k.startswith("head."))]
   if bad or not any(v for k,v in changes.items() if k.startswith("blocks.")) or not any(v for k,v in changes.items() if k.startswith("head.")) or changes!=summary["changes_at_best"]:raise ValueError("freeze boundary")
   rows.append({"run":rec["run"],"epochs":len(hist),"best_epoch":best_epoch,"best_metric":metrics[best_epoch],"summary_sha256":rec["summary_sha256"],"best_sha256":manifest["best"]["sha256"],"latest_sha256":manifest["latest"]["sha256"],"selection_replay_max_abs":max(abs(x-y["selection_pAUC_0_0.1"]) for x,y in zip(metrics,hist)),"frozen_norm_pooler_trunk_exact":True,"blocks_and_head_changed":True})
  except Exception as e:fail.append(f"{rec['run']}:{e}")
 result={"schema":"round19_incremental_training_audit_v1","status":"pass" if len(rows)==len(targets) and not fail else "fail","formal_status_sha256":tr.sha(a.local/"FORMAL_STATUS.json"),"complete_runs_seen":len(targets),"audited_runs":len(rows),"failures":fail,"rows":rows,"test_consumed":False};tr.atomic_json(a.output,result);print(json.dumps(result,indent=2));assert result["status"]=="pass"
if __name__=="__main__":main()
