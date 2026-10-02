#!/usr/bin/env python3
"""E0 incremental-endpoint and conditional F3 support audit."""
from __future__ import annotations
import argparse,json,hashlib,os,tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
ROLES=("fit","selection","calibration","validation")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:Path(t).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/vla1/zjy/sparsh_runs/force_slip_htt_normalflow"));ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();e0=[];f3=[]
 for fold in range(1,5):
  support_path=a.root/f"round10_htt_force_supervision_adaptation/force_support/fold_p{fold}.json";support=json.loads(support_path.read_text());entries={e["episode_id"]:e for e in support["entries"]}
  manifest_path=a.root/f"round10_htt_force_supervision_adaptation/force_predictions/p{fold}_s20260914/prediction_manifest.json";manifest=json.loads(manifest_path.read_text());pred={e["episode_id"]:e for e in manifest["entries"]}
  old_path=a.root/f"round17_htt_shared_temporal_multitask/prepared/p{fold}_s20260914/prepared.pt";old=torch.load(old_path,map_location="cpu",weights_only=False)
  for role in ROLES:
   oldkeys={(e,int(t)) for e,t in zip(old["roles"][role]["episode_id"],old["roles"][role]["t"])};added=[];excluded=defaultdict(int)
   role_entries=[e for e in entries.values() if e["role"]==role]
   for e in role_entries:
    if e["episode_id"] not in pred:excluded["prediction_episode_missing"]+=1;continue
    y=np.load(e["label_path"]);gt=np.load(e["force_native_n_path"]);pr=np.load(pred[e["episode_id"]]["prediction_path"],mmap_mode="r")
    if len(y)!=e["frames"] or len(gt)!=e["frames"] or len(pr)!=e["frames"]:raise ValueError("frame alignment")
    for t in range(13,e["frames"]):
     if (e["episode_id"],t) in oldkeys:continue
     if int(y[t]) not in (0,1,2):excluded["unknown_label"]+=1;continue
     if not np.isfinite(gt[t]).all():excluded["gt_nonfinite"]+=1;continue
     if not np.isfinite(pr[t-8:t+1]).all():excluded["prediction_nonfinite"]+=1;continue
     added.append((e["episode_id"],t,int(y[t])))
   e0.append({"fold":fold,"role":role,"old_endpoints":len(oldkeys),"label_gt_prediction_aligned_upper_bound":len(oldkeys)+len(added),"added_endpoint_upper_bound":len(added),"trials_with_added_endpoints":len({x[0] for x in added}),"new_independent_trials":0,"added_static":sum(x[2]==0 for x in added),"added_incipient":sum(x[2]==1 for x in added),"added_gross":sum(x[2]==2 for x in added),"excluded":dict(excluded),"actual_minimum_t":13,"minimum_t_basis":"R10 minimum_unpadded_base=5 plus nine base steps t-8..t","limitation":"upper bound verifies labels, GT and R10 prediction arrays; expanded nine-step visual cache materialization remains pending"})
   by_h={}
   for horizon in (1,5,10):
    pos=set();neg=set();unknown=0;left=0
    for e in role_entries:
     y=np.load(e["label_path"])
     for t in range(13,len(y)-horizon):
      future=y[t+1:t+horizon+1]
      if int(y[t])!=0:continue
      if np.any(y[:t]==2):left+=1;continue
      if not all(int(v) in (0,1,2) for v in future):unknown+=1;continue
      if np.any(future==2):pos.add(e["episode_id"])
      else:neg.add(e["episode_id"])
    by_h[str(horizon)]={"positive_trials":len(pos),"negative_complete_trials":len(neg),"positive_trial_ids":sorted(pos),"negative_trial_ids":sorted(neg),"unknown_or_gap_endpoints":unknown,"prior_gross_excluded_endpoints":left}
   f3.append({"fold":fold,"role":role,"windows":by_h})
 gate=[]
 for fold in range(1,5):
  for h in (1,5,10):
   rows={(x["role"]):x["windows"][str(h)] for x in f3 if x["fold"]==fold};ok=rows["fit"]["positive_trials"]>=20 and rows["fit"]["negative_complete_trials"]>=20 and all(rows[r]["positive_trials"]>=5 and rows[r]["negative_complete_trials"]>=5 for r in ("selection","calibration","validation"));gate.append({"fold":fold,"horizon":h,"outer_gate_pass":ok,"counts":rows})
 out={"schema":"round22_e0_f3_support_v2","status":"pass","e0":e0,"f3":f3,"f3_gate":gate,"f3_selected_first_fold_window":next(({"fold":x["fold"],"horizon":x["horizon"]} for x in gate if x["outer_gate_pass"]),None),"f3_gate_conclusion":"not_triggered; every fold/window fails fixed outer-role support even under this label-availability upper bound","test_consumed":False,"support_hashes":{str(a.root/f"round10_htt_force_supervision_adaptation/force_support/fold_p{f}.json"):sha(a.root/f"round10_htt_force_supervision_adaptation/force_support/fold_p{f}.json") for f in range(1,5)}};atomic(a.output,out);print(json.dumps({"e0_added_upper_bound":sum(x["added_endpoint_upper_bound"] for x in e0),"f3_selected":out["f3_selected_first_fold_window"]}))
if __name__=="__main__":main()
