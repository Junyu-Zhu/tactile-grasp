#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
import multitask_train as mt

def main():
 p=argparse.ArgumentParser();p.add_argument("--r14-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--local",type=Path,default=Path(__file__).parent);p.add_argument("--refresh",action="store_true");a=p.parse_args();(a.output/"prepared").mkdir(parents=True,exist_ok=True);rows=[]
 for fold in range(1,5):
  for seed in mt.SEEDS:
   r14=a.r14_root/"prepared"/f"p{fold}_s{seed}"/"prepared.pt";d=torch.load(r14,map_location="cpu",weights_only=False);parent=Path(d["provenance"]["prepared"]);out=a.output/"prepared"/f"p{fold}_s{seed}"/"prepared.pt"
   if out.exists() and not a.refresh:
    old=torch.load(out,map_location="cpu",weights_only=False)
    if old.get("provenance",{}).get("round14_sha256")!=mt.sha(r14):raise ValueError(f"stale prepared {out}")
    row={"output":str(out),"sha256":mt.sha(out),"fold":fold,"seed":seed,"support":old["support"],"joint_weight":old["joint_weight"],"normalizer_sha256":"existing_verified"}
   else:row=mt.enrich_round14(r14,parent,out,fold,seed)
   rows.append(row);print(mt.stable_json({"prepared":f"p{fold}_s{seed}","sha256":row["sha256"],"selection":row["support"]["selection"],"lambda":row["joint_weight"]["lambda_future"]}),flush=True)
 # Common init and precise selection support audit.
 for row in rows:
  s=row["support"]["selection"];ns=s["stage_counts"]["0"];row["selection_identifiability"]={"static_endpoints":ns,"gross_endpoints":s["stage_counts"]["2"],"static_trials":s["stage_episode_counts"]["0"],"gross_trials":s["stage_episode_counts"]["2"],"one_false_positive_quantum":1/ns,"fpr_1pct_resolvable_by_count":ns>=100,"fpr_5pct_resolvable_by_count":ns>=20,"fpr_10pct_resolvable_by_count":ns>=10}
  audit={"schema":"round17_support_audit_v1","status":"pass","prepared":rows,"notes":["selection is internal train only","1/Nstatic is the smallest positive empirical frame-FPR increment; zero FP can retain gross recall and is not automatically never-alarm","low resolution is reported and never repaired with outer roles"]};mt.atomic_json(audit,a.local/"SUPPORT_AUDIT.json")
 identities=[]
 for row in rows:
  d=torch.load(row["output"],map_location="cpu",weights_only=False);identities.append({"fold":d["fold"],"seed":d["seed"],"cache":row["sha256"],"round14":d["provenance"]["round14_sha256"],"parent":d["provenance"]["parent_sha256"],"upstream":d["provenance"]["immutable_upstream"],"module_init_hashes":{n:mt.state_hash(mt.init_model(d["seed"]).state_dict(),(n,)) for n in mt.MODULES}})
 lock={"schema":"round17_identity_lock_v1","status":"locked","forbidden_upstream":"Round16 ToucHD/H/T->H force","identities":identities};mt.atomic_json(lock,a.local/"IDENTITY_LOCK.json")
 files=("USER_SCOPE.md","HANDOFF.md","PROTOCOL.md","BUDGET.json","RUN_INVENTORY.json","multitask_train.py","prepare_all.py","smoke.py","run_all.py","audit_prepare.py","audit_recovery.py")
 pl={"schema":"round17_protocol_lock_v1","status":"locked_before_formal_results","files":{f:mt.sha(a.local/f) for f in files}};mt.atomic_json(pl,a.local/"PROTOCOL_LOCK.json")
 print(json.dumps({"status":"pass","prepared":len(rows),"support_audit":str(a.local/"SUPPORT_AUDIT.json"),"identity_lock":str(a.local/"IDENTITY_LOCK.json")},indent=2))
if __name__=="__main__":main()
