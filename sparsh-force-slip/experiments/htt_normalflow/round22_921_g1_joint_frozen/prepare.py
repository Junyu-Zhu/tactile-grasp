#!/usr/bin/env python3
"""Register the frozen G1 grids and audit actual cached endpoint support."""
from __future__ import annotations
import argparse, hashlib, json, tempfile, os
from pathlib import Path
import torch

ROLES=("fit","selection","calibration","validation");SEEDS=(20260914,20260915,20260916)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:Path(t).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def keys(d,role):return {(e,int(t)) for e,t in zip(d["roles"][role]["episode_id"],d["roles"][role]["t"])}
def counts(d,role):
 r=d["roles"][role];ts={}
 for e,t in zip(r["episode_id"],r["t"]):ts.setdefault(e,[]).append(int(t))
 return {"endpoints":len(r["t"]),"trials":len(ts),"leakage_groups":len(set(r["leakage_group"])),"min_t":min(map(int,r["t"])),"max_t":max(map(int,r["t"])),"noncontiguous_trials":sum(sorted(v)!=list(range(min(v),max(v)+1)) for v in ts.values())}
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-root",type=Path,default=Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow"));p.add_argument("--output",type=Path,required=True);a=p.parse_args();aud=[]
 for fold in range(1,5):
  ref=None
  for seed in SEEDS:
   r17=a.run_root/f"round17_htt_shared_temporal_multitask/prepared/p{fold}_s{seed}/prepared.pt";r14=a.run_root/f"round14_htt_future_force_dual/prepared/p{fold}_s{seed}/prepared.pt";d17=torch.load(r17,map_location="cpu",weights_only=False);d14=torch.load(r14,map_location="cpu",weights_only=False)
   row={"fold":fold,"seed":seed,"current_cache":str(r17),"current_sha256":sha(r17),"future_cache":str(r14),"future_sha256":sha(r14),"roles":{}}
   for role in ROLES:
    kc,kf=keys(d17,role),keys(d14,role);c=counts(d17,role);f=counts(d14,role);row["roles"][role]={"current":c,"future":f,"intersection":len(kc&kf),"current_only":len(kc-kf),"future_only":len(kf-kc),"same_trial_role":set(d17["roles"][role]["episode_id"])==set(d14["roles"][role]["episode_id"]),"current_gt_finite":bool(torch.isfinite(d17["roles"][role]["y_current"]).all())}
   identity={role:keys(d17,role) for role in ROLES}
   if ref is None:ref=identity
   elif identity!=ref:raise ValueError(f"seed endpoint mismatch fold {fold}")
   aud.append(row)
 runs=[]
 specs={"E1":("V","C","M"),"E2":("MB",),"E3":("A","B","C","D"),"F1":("K-V","K-F","K-VF")}
 for package,groups in specs.items():
  for group in groups:
   for fold in range(1,5):
    for seed in SEEDS:runs.append({"package":package,"group":group,"fold":fold,"seed":seed,"status":"registered_not_dispatched"})
 out={"schema":"round22_g1_prepare_v1","status":"prepared_not_authorized","support":aud,"runs":runs,"run_counts":{k:len(v)*12 for k,v in specs.items()},"F2":{"neural_runs":0,"operation":"0.5 times accepted K-VF delta"},"F3":{"max_runs":12,"status":"gate_pending","first_gross_requires_no_prior_gross":True,"negative_requires_complete_known_horizon":True},"test_consumed":False}
 write(a.output/"G1_PREPARE.json",out);print(json.dumps({"status":out["status"],"runs":len(runs),"counts":out["run_counts"]}))
if __name__=="__main__":main()
