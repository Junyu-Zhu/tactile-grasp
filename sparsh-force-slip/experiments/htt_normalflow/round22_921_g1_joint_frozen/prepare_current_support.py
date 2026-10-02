#!/usr/bin/env python3
"""Materialize exact current-detection/E3 support from accepted R10 full endpoints."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np,torch
ROLES=("fit","selection","calibration","validation");SEEDS=(20260914,20260915,20260916)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def atomic_save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:torch.save(x,t);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(dir=p.parent);os.close(fd)
 try:Path(t).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def endpoint_hash(r):
 h=hashlib.sha256()
 for e,t,g in zip(r["episode_id"],r["t"].tolist(),r["leakage_group"]):h.update(f"{e}\0{int(t)}\0{g}\n".encode())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();records=[]
 for fold in range(1,5):
  support_path=a.run_root/"round10_htt_force_supervision_adaptation"/"force_support"/f"fold_p{fold}.json";support_sha=sha(support_path);support=json.loads(support_path.read_text());entries={e["episode_id"]:e for e in support["entries"]}
  for seed in SEEDS:
   parent=a.run_root/"round10_htt_force_supervision_adaptation"/"prepare"/f"p{fold}_s{seed}"/"prepared.pt";manifest_path=a.run_root/"round10_htt_force_supervision_adaptation"/"force_predictions"/f"p{fold}_s{seed}"/"prediction_manifest.json";old_path=a.run_root/"round17_htt_shared_temporal_multitask"/"prepared"/f"p{fold}_s{seed}"/"prepared.pt";out=a.output/f"p{fold}_s{seed}"/"prepared.pt";audit=out.with_name("audit.json")
   parent_sha=sha(parent);manifest_sha=sha(manifest_path);old_sha=sha(old_path);source_sha=sha(__file__)
   if out.exists() and audit.exists():
    oldaudit=json.loads(audit.read_text())
    if (oldaudit.get("parent_sha256"),oldaudit.get("force_prediction_manifest_sha256"),oldaudit.get("support_sha256"),oldaudit.get("old_round17_sha256"),oldaudit.get("source_sha256"))==(parent_sha,manifest_sha,support_sha,old_sha,source_sha) and oldaudit.get("prepared_sha256")==sha(out):records.append(oldaudit);continue
    raise RuntimeError(f"stale committed current support cache: {out}")
   data=torch.load(parent,map_location="cpu",weights_only=False);old=torch.load(old_path,map_location="cpu",weights_only=False);manifest=json.loads(manifest_path.read_text());pred={e["episode_id"]:e for e in manifest["entries"]}
   if manifest.get("minimum_unpadded_base")!=5 or data.get("experiment")!="round10_new_force":raise ValueError("incompatible R10 source")
   roles={};role_stats={}
   for role in ROLES:
    src=data["roles"]["train" if role in ("fit","selection") else role];wanted={e["episode_id"] for e in support["entries"] if e["role"]==role};ids=torch.tensor([i for i,e in enumerate(src["episode_id"]) if e in wanted],dtype=torch.long)
    item={"x":src["x"][ids,:,:195].contiguous(),"t":src["t"][ids].clone(),"stage":src["stage"][ids].clone(),"episode_id":[src["episode_id"][i] for i in ids.tolist()],"leakage_group":[src["leakage_group"][i] for i in ids.tolist()],"probe":[src["probe"][i] for i in ids.tolist()]}
    gt=[];gt_cache={eid:np.load(entries[eid]["force_native_n_path"],allow_pickle=False) for eid in wanted};label_cache={eid:np.load(entries[eid]["label_path"],allow_pickle=False) for eid in wanted}
    for eid,t in zip(item["episode_id"],item["t"].tolist()):gt.append(gt_cache[eid][int(t)])
    item["y_current"]=torch.from_numpy(np.asarray(gt,np.float32));roles[role]=item
    if item["x"].shape[1:]!=(9,195) or not torch.isfinite(item["x"]).all() or not torch.isfinite(item["y_current"]).all():raise ValueError("nonfinite or shape")
    if int(item["t"].min())!=13 or not torch.isin(item["stage"],torch.tensor([0,1,2])).all():raise ValueError("time/stage")
    expected_stage=torch.tensor([int(label_cache[eid][int(t)]) for eid,t in zip(item["episode_id"],item["t"].tolist())],dtype=item["stage"].dtype)
    if not torch.equal(item["stage"],expected_stage):raise ValueError(f"support label_path stage parity failed: {role}")
    for eid in wanted:
     ts=sorted(int(t) for e,t in zip(item["episode_id"],item["t"].tolist()) if e==eid)
     if ts!=list(range(13,entries[eid]["frames"])):raise ValueError(f"noncontiguous current support {eid}")
    oldr=old["roles"][role];index={(e,int(t)):i for i,(e,t) in enumerate(zip(item["episode_id"],item["t"]))};ix=torch.tensor([index[(e,int(t))] for e,t in zip(oldr["episode_id"],oldr["t"])])
    old_episode=[item["episode_id"][i] for i in ix.tolist()];old_leakage=[item["leakage_group"][i] for i in ix.tolist()];old_probe=[item["probe"][i] for i in ix.tolist()]
    if not torch.equal(item["x"][ix],oldr["x"]) or not torch.equal(item["y_current"][ix],oldr["y_current"]) or not torch.equal(item["stage"][ix],oldr["stage"]) or old_episode!=oldr["episode_id"] or old_leakage!=oldr["leakage_group"] or old_probe!=oldr["probe"]:raise ValueError("old intersection x/y/stage/role parity")
    role_stats[role]={"endpoints":len(ids),"trials":len(wanted),"leakage_groups":len(set(item["leakage_group"])),"stage_counts":{str(c):int(item["stage"].eq(c).sum()) for c in (0,1,2)},"endpoint_sha256":endpoint_hash(item),"old_intersection":len(ix),"added_tail_endpoints":len(ids)-len(ix),"old_intersection_exact":True}
   groups=[set(roles[r]["leakage_group"]) for r in ROLES]
   if any(groups[i]&groups[j] for i in range(4) for j in range(i)):raise ValueError("role leakage")
   immutable={"r10_parent":{"path":str(parent),"sha256":parent_sha},"force_prediction_manifest":{"path":str(manifest_path),"sha256":manifest_sha},"support":{"path":str(support_path),"sha256":support_sha},"round17_old_common":{"path":str(old_path),"sha256":old_sha},"source_checkpoint":data["provenance"]["source_checkpoint"],"visual_checkpoint":data["provenance"]["visual_checkpoint"],"force_checkpoint":data["provenance"]["force_checkpoint"]}
   result={"schema":"round22_e3_current_common_gt_v1","fold":fold,"seed":seed,"roles":roles,"provenance":{"immutable_upstream":immutable,"support_scope":"current endpoints t=13..last, exact current label/GT/R10 prediction; no future completeness","test_consumed":False,"raw_image_union":"t-13..t"}}
   atomic_save(out,result);rec={"schema":"round22_current_support_audit_v2","status":"pass","fold":fold,"seed":seed,"prepared":str(out),"prepared_sha256":sha(out),"parent_sha256":parent_sha,"force_prediction_manifest_sha256":manifest_sha,"support_sha256":support_sha,"old_round17_sha256":old_sha,"source_sha256":source_sha,"roles":role_stats,"new_independent_trials":0,"test_consumed":False};atomic_json(audit,rec);records.append(rec);print(json.dumps({"fold":fold,"seed":seed,"roles":role_stats}),flush=True)
 atomic_json(a.output/"PREPARE_CURRENT_SUPPORT.json",{"schema":"round22_prepare_current_support_v1","status":"pass","runs":records,"total_added_tail_endpoints":sum(sum(v["added_tail_endpoints"] for v in r["roles"].values()) for r in records),"note":"tail endpoint counts repeat across seeds; independent trial support is unchanged"})
if __name__=="__main__":main()
