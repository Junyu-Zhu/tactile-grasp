#!/usr/bin/env python3
"""Read-only audit of all formal Round-19 commits and selection histories."""
import argparse,json,math
from pathlib import Path
import torch
import train as tr
def main():
 p=argparse.ArgumentParser();p.add_argument("--local",type=Path,default=Path(__file__).resolve().parent);p.add_argument("--large",type=Path,required=True);p.add_argument("--r17",type=Path,required=True);p.add_argument("--r3",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();formal=json.loads((a.local/"FORMAL_STATUS.json").read_text());rows=[];fail=[]
 if formal.get("status")!="complete" or formal.get("status_counts")!={"complete":24}:fail.append("formal_status")
 for fold in range(1,5):
  for seed in tr.SEEDS:
   dp=a.r17/f"p{fold}_s{seed}"/"prepared.pt";data=torch.load(dp,map_location="cpu",weights_only=False);sel=data["roles"]["selection"];mask=sel["stage"].eq(0)|sel["stage"].eq(2)
   for group in tr.GROUPS:
    run=a.large/"formal"/group/f"p{fold}_s{seed}";summary=json.loads((run/"summary.json").read_text());manifest=json.loads((run/"COMMIT.json").read_text());best=run/manifest["best"]["path"]
    try:
     if summary.get("status")!="complete":raise ValueError("summary")
     if tr.sha(best)!=manifest["best"]["sha256"]:raise ValueError("best hash")
     latest=run/manifest["latest"]["path"]
     if tr.sha(latest)!=manifest["latest"]["sha256"]:raise ValueError("latest hash")
     ck=torch.load(best,map_location="cpu",weights_only=False);last=torch.load(latest,map_location="cpu",weights_only=False);identity=ck["identity"]
     if (identity["group"],identity["fold"],identity["seed"],identity["data_sha256"],identity["source_sha256_code"],identity["protocol_sha256"],identity["formal"])!=(group,fold,seed,tr.sha(dp),tr.sha(a.local/"train.py"),tr.sha(a.local/"PROTOCOL.md"),True):raise ValueError("identity")
     hist=last["history"]
     if not hist or len(hist)>tr.MAX_EPOCHS or any(len(x["selection_scores"])!=len(sel["t"]) for x in hist):raise ValueError("history")
     recomputed=[tr.low_fpr_auc(sel["stage"][mask],torch.tensor(x["selection_scores"])[mask]) for x in hist]
     if any(abs(v-x["selection_pAUC_0_0.1"])>1e-12 for v,x in zip(recomputed,hist)):raise ValueError("selection replay")
     best_epoch=max(range(len(recomputed)),key=lambda i:recomputed[i])
     if last["best_epoch"]!=best_epoch or summary["best_epoch"]!=best_epoch or manifest["best"]["epoch"]!=best_epoch or summary["epochs"]!=len(hist) or abs(summary["best_metric"]-recomputed[best_epoch])>1e-12:raise ValueError("earliest strict best")
     initial=tr.Model(group,seed,a.source,a.r3/f"fold_p{fold}"/f"seed_{seed}"/"best.pth").state_dict()
     if tr.state_hash(initial)!=ck["initial_state_sha256"]:raise ValueError("initial state hash")
     independent_changes={k:not torch.equal(initial[k],v) for k,v in ck["model"].items()};bad=[k for k,v in independent_changes.items() if v and not (k.startswith("blocks.") or k.startswith("head."))]
     if bad or not any(v for k,v in independent_changes.items() if k.startswith("blocks.")) or not any(v for k,v in independent_changes.items() if k.startswith("head.")):raise ValueError("gradient boundary")
     if independent_changes!=summary["changes_at_best"]:raise ValueError("summary change map")
     rows.append({"group":group,"fold":fold,"seed":seed,"epochs":len(hist),"best_epoch":best_epoch,"best_metric":recomputed[best_epoch],"optimizer_steps":summary["optimizer_steps"],"best_commit_sha256":tr.sha(best),"selection_replay_max_abs":max(abs(v-x["selection_pAUC_0_0.1"]) for v,x in zip(recomputed,hist)),"only_blocks_and_head_changed":True})
    except Exception as e:fail.append(f"{group}/p{fold}_s{seed}:{e}")
 result={"schema":"round19_training_audit_v1","status":"pass" if len(rows)==24 and not fail else "fail","runs":len(rows),"failures":fail,"rows":rows,"test_consumed":False};tr.atomic_json(a.output,result);print(json.dumps(result,indent=2));assert result["status"]=="pass"
if __name__=="__main__":main()
