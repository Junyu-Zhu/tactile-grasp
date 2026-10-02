#!/usr/bin/env python3
"""Bounded direct-vs-deduplicated score diagnostic for a failed prediction replay."""
from __future__ import annotations
import argparse,importlib.util,json,random,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;R22=HERE.parent/"round22_921_g1_joint_frozen"
def load_train():
 s=importlib.util.spec_from_file_location("r23_diag_train",R22/"train_e3.py");m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
tr=load_train()
def main():
 p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--run",required=True);p.add_argument("--deduplicated-npz",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();rr=next(x for x in json.loads(a.inventory.read_text())["runs"] if x["run"]==a.run);run=Path(rr["output"]);cm=json.loads((run/"COMMIT.json").read_text());ck=torch.load(run/cm["best"]["path"],map_location="cpu",weights_only=False);norm=ck["normalizer"]
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True);random.seed(rr["seed"]);np.random.seed(rr["seed"]);torch.manual_seed(rr["seed"]);torch.cuda.manual_seed_all(rr["seed"])
 data=tr.Data(rr["data"],rr["prefix_index"]);model=tr.Model(rr["group"],rr["seed"],rr["source"],rr["visual_checkpoint"]);model.load_state_dict(ck["model"],strict=True);model.to(a.device).eval();direct=tr.infer(model,data,"selection",norm,a.device).numpy().astype(np.float64);dedup=np.load(a.deduplicated_npz,allow_pickle=False)["selection_score"].astype(np.float64);d=data.d["roles"]["selection"];mask=(d["stage"].eq(0)|d["stage"].eq(2));best=float(json.loads((run/"summary.json").read_text())["best_metric"]);pd=tr.low_fpr_auc(d["stage"][mask],torch.from_numpy(direct)[mask]);pc=tr.low_fpr_auc(d["stage"][mask],torch.from_numpy(dedup)[mask]);delta=np.abs(direct-dedup);order_d=np.argsort(-direct,kind="stable");order_c=np.argsort(-dedup,kind="stable")
 out={"schema":"round23_e3_prediction_parity_diagnostic_v1","status":"complete","run":a.run,"best_metric":best,"direct_pAUC":pd,"direct_replay_abs_error":abs(pd-best),"deduplicated_pAUC":pc,"deduplicated_replay_abs_error":abs(pc-best),"score_max_abs":float(delta.max()),"score_mean_abs":float(delta.mean()),"score_changed":int((delta>0).sum()),"score_gt_1e-6":int((delta>1e-6).sum()),"score_gt_1e-5":int((delta>1e-5).sum()),"exact_rank_positions_changed":int((order_d!=order_c).sum()),"direct_required_for_this_run":abs(pd-best)<=2e-6 and abs(pc-best)>2e-6,"best_commit_sha256":cm["best"]["sha256"],"test_consumed":False};tr.atomic_json(a.output,out);print(json.dumps(out));raise SystemExit(0 if out["direct_required_for_this_run"] else 1)
if __name__=="__main__":main()
