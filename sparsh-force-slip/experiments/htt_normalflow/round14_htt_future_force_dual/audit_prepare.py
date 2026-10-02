#!/usr/bin/env python3
"""Audit all 12 prepared future caches without loading historical test data."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np, torch
import future_train as ft

def main():
 p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--r10",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
 runs=[]; endpoint_ref={}
 for fold in range(1,5):
  support=a.r10/f"force_support/fold_p{fold}.json"; sm=json.load(open(support)); entries={e["episode_id"]:e for e in sm["entries"]}
  for seed in (20260914,20260915,20260916):
   path=a.root/f"prepared/p{fold}_s{seed}/prepared.pt"; d=torch.load(path,map_location="cpu",weights_only=False)
   assert d["schema"]=="round14_future_cache_v1" and tuple(d["horizons"])==ft.HORIZONS; ft.assert_role_isolation(d["roles"])
   role_summary={}; endpoint_identity={}
   for role,r in d["roles"].items():
    assert r["x"].shape[1:]==(9,195) and r["y"].shape[1:]==(3,3) and r["y_current"].shape[1:]==(3,)
    assert torch.isfinite(r["x"]).all() and torch.isfinite(r["y"]).all() and torch.isfinite(r["y_current"]).all() and float(r["y"].abs().max())<=20 and float(r["y_current"].abs().max())<=20
    by={}
    for eid,t in zip(r["episode_id"],r["t"].tolist()): by.setdefault(eid,[]).append(t)
    assert all(sorted(ts)==list(range(13,entries[eid]["frames"]-max(ft.HORIZONS))) for eid,ts in by.items())
    # Deterministic endpoint samples prove current/future binding.
    ix=sorted(set([0,len(r["t"])//2,len(r["t"])-1]))
    for i in ix:
     eid=r["episode_id"][i]; t=int(r["t"][i]); gt=np.load(entries[eid]["force_native_n_path"])
     assert np.array_equal(r["y_current"][i].numpy(),gt[t]) and np.array_equal(r["y"][i].numpy(),gt[[t+h for h in ft.HORIZONS]])
    key=[(e,int(t),g) for e,t,g in zip(r["episode_id"],r["t"],r["leakage_group"])]
    endpoint_identity[role]=hashlib.sha256(repr(key).encode()).hexdigest()
    delta=(r["y"][:,2]-r["y_current"]).abs().amax(1)
    role_summary[role]={"rows":len(r["t"]),"episodes":len(by),"groups":len(set(r["leakage_group"])),"t_min":int(r["t"].min()),"t_max":int(r["t"].max()),"stable_le_0p25":int((delta<=.25).sum()),"change_ge_1p0":int((delta>=1).sum()),"saturated_values":int((r["y"].abs()>=20).sum()),"endpoint_sha256":endpoint_identity[role]}
   if fold not in endpoint_ref: endpoint_ref[fold]=endpoint_identity
   else: assert endpoint_ref[fold]==endpoint_identity
   runs.append({"fold":fold,"seed":seed,"path":str(path),"sha256":ft.sha(path),"roles":role_summary,"provenance":d["provenance"]})
 result={"schema":"round14_prepare_audit_v1","status":"pass","run_count":len(runs),"horizons_frames":list(ft.HORIZONS),"history_base_steps":9,"raw_image_union":"t-13..t, each base uses (s,s-5)","target":"reference-relative xyz clipped [-20,20] N","checks":["all_12_caches","no_test_role","role_group_disjoint","all_endpoints_contiguous_and_complete","same_fold_endpoints_identical_across_seeds","sampled_targets_exact","finite_and_clipped","immutable_upstream_paths_rehashed_during_prepare"],"change_strata_note":"0.25/1.0 N thresholds were fixed before this audit; counts describe support and did not choose thresholds","runs":runs}
 ft.atomic_json(result,a.output); print(json.dumps({"status":"pass","runs":len(runs),"output":str(a.output)}))
if __name__=="__main__": main()

