#!/usr/bin/env python3
"""Create an explicit G2 inventory; missing new-support inputs remain blocking."""
import argparse,hashlib,json
from pathlib import Path
GROUPS=("A","B","C","D");SEEDS=(20260914,20260915,20260916)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def memoized_hasher():
 cache={}
 def maybe(p):
  p=Path(p)
  if not p.is_file():return None
  key=(str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns)
  if key not in cache:cache[key]=sha(p)
  return cache[key]
 return maybe
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-root",type=Path,required=True);p.add_argument("--source-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();here=Path(__file__).resolve().parent;large=a.run_root/"round22_921_g1_joint_frozen";prefix=a.run_root/"round19_htt_partial_encoder_finetuning"/"prefix_cache"/"PREFIX_INDEX.json";source=a.source_root/"phase_lambda_decoupled_mae_lam010_20260528_000000"/"mae_decoupled_multitask"/"checkpoints"/"epoch-0030.pth";protocol=here/"E3_PROTOCOL_LOCK.json";maybe=memoized_hasher();train_source_sha=maybe(here/"train_e3.py");runs=[]
 for fold in range(1,5):
  for seed in SEEDS:
   data=large/"prepared"/"e3_current_common_gt"/f"p{fold}_s{seed}"/"prepared.pt";visual=a.run_root/"round3_mae_slip_adaptation"/"runs"/"B"/f"fold_p{fold}"/f"seed_{seed}"/"best.pth"
   for group in GROUPS:runs.append({"run":f"{group}/p{fold}_s{seed}","group":group,"fold":fold,"seed":seed,"data":str(data),"data_sha256":maybe(data),"data_schema":"round22_e3_current_common_gt_v1","support_scope":"round22_current_common_gt_exact","prefix_index":str(prefix),"prefix_index_sha256":maybe(prefix),"source":str(source),"source_sha256":maybe(source),"visual_checkpoint":str(visual),"visual_checkpoint_sha256":maybe(visual),"protocol":str(protocol),"protocol_sha256":maybe(protocol),"train_source_sha256":train_source_sha,"output":str(large/"formal"/group/f"p{fold}_s{seed}")})
 out={"schema":"round22_g2_run_inventory_v1","status":"blocked_missing_new_e3_data_and_protocol" if any(not r["data_sha256"] or not r["protocol_sha256"] for r in runs) else "ready","runs":runs};a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":out["status"],"runs":len(runs),"missing_data":sum(not r["data_sha256"] for r in runs),"missing_protocol":sum(not r["protocol_sha256"] for r in runs)}))
if __name__=="__main__":main()
