#!/usr/bin/env python3
"""Build the exact undispatched E1/E2/F1 84-run inventory."""
import argparse,hashlib,json
from pathlib import Path
E1=("V","C","M");E2=("MB",);F1=("K-V","K-F","K-VF");SEEDS=(20260914,20260915,20260916)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--include-e2",action="store_true");a=p.parse_args();here=Path(__file__).resolve().parent;protocol=here/"FROZEN_PROTOCOL_LOCK.json";runs=[];packages=(("E1",E1),("F1",F1))+((("E2",E2),) if a.include_e2 else ())
 for fold in range(1,5):
  for seed in SEEDS:
   current=a.run_root/"round22_921_g1_joint_frozen"/"prepared"/"e3_current_common_gt"/f"p{fold}_s{seed}"/"prepared.pt";future=a.run_root/"round14_htt_future_force_dual"/"prepared"/f"p{fold}_s{seed}"/"prepared.pt"
   for package,groups in packages:
    data=current if package in ("E1","E2") else future;trainer=here/("train_frozen.py" if package in ("E1","E2") else "train_f1.py");support_inventory=here/("PREPARE_CURRENT_SUPPORT.json" if package in ("E1","E2") else "G1_PREPARE.json");dependencies={"support_inventory_sha256":sha(support_inventory)}
    if package in ("E1","E2"):dependencies["r18_source_sha256"]=sha(here.parent/"round18_htt_force_conditioned_film"/"train.py")
    for group in groups:runs.append({"run":f"{package}/{group}/p{fold}_s{seed}","package":package,"group":group,"fold":fold,"seed":seed,"data":str(data),"data_sha256":sha(data),"trainer":str(trainer),"trainer_sha256":sha(trainer),"protocol":str(protocol),"protocol_sha256":sha(protocol),"dependencies":dependencies,"output":str(a.run_root/"round22_921_g1_joint_frozen"/"formal"/"frozen"/package/group/f"p{fold}_s{seed}"),"status":"candidate_not_dispatched","applicability":"budget_pending_optional_E2" if package=="E2" else "authorized_scope_budget_pending"})
 counts={p:len(gs)*12 for p,gs in packages};out={"schema":"round22_frozen_run_inventory_v1","status":"candidate_not_dispatched_budget_and_authorization_false","formal_training_started":False,"packages":[p for p,_ in packages],"run_count":len(runs),"counts":counts,"runs":runs};a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":out["status"],"runs":len(runs),"packages":out["packages"]}))
if __name__=="__main__":main()
