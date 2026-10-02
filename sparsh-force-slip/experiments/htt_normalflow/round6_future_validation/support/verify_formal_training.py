#!/usr/bin/env python3
"""Independent read-only acceptance audit for the nine formal Round-6 GRUs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import torch


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path,value) -> None:
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));os.replace(tmp,path)


def load_module(path: Path):
    spec=importlib.util.spec_from_file_location("round6_training_audit_module",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser();p.add_argument("--inventory",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    inv=json.loads(a.inventory.read_text());trainer_path=Path(inv["trainer"]);trainer=load_module(trainer_path)
    expected={(g,s) for g in trainer.GROUPS for s in trainer.SEEDS}
    if len(inv["runs"])!=9 or {(r["group"],r["seed"]) for r in inv["runs"]}!=expected or len({r["output"] for r in inv["runs"]})!=9:raise ValueError("formal inventory identity set mismatch")
    for path,digest in inv["frozen_inputs"].items():
        if sha(Path(path))!=digest:raise ValueError(f"frozen input drift: {path}")
    data=torch.load(inv["prepared_data"],map_location="cpu",weights_only=False);queue_root=Path(inv["output_root"]);queue=json.loads((queue_root/"QUEUE_STATE.json").read_text())
    if queue.get("status")!="complete" or set(queue["jobs"])!={r["id"] for r in inv["runs"]} or any(v!="complete" for v in queue["jobs"].values()):raise ValueError("queue is not exactly complete")
    source_sha=trainer.code_bundle_sha();role_counts={role:len(data["roles"][role]["y"]) for role in data["roles"]};runs=[];initial_by_seed={}
    for run in inv["runs"]:
        folder=Path(run["output"]);summary_path=folder/"summary.json";summary=json.loads(summary_path.read_text());rid=run["id"]
        if summary.get("status")!="complete" or summary.get("formal") is not True or summary.get("smoke") is not False:raise ValueError(f"nonformal summary: {rid}")
        if summary["group"]!=run["group"] or summary["seed"]!=run["seed"]:raise ValueError(f"summary identity mismatch: {rid}")
        cfg=summary["run_config"]
        required={"group":run["group"],"seed":run["seed"],"mode":"formal","horizon":1,"effective_epoch_budget":100,"code_bundle_sha256":source_sha,"data_sha256":sha(Path(inv["prepared_data"]))}
        if any(cfg.get(k)!=v for k,v in required.items()):raise ValueError(f"run config mismatch: {rid}")
        checks=summary["checks"]
        for key in ("finite_loss_and_all_trainable_gradients","all_parameter_tensors_updated","checkpoint_roundtrip_exact","calibration_and_outer_not_used_for_training_or_selection"):
            if checks.get(key) is not True:raise ValueError(f"failed check {key}: {rid}")
        if not checks["parameter_updates_by_tensor"] or not all(v is True for v in checks["parameter_updates_by_tensor"].values()):raise ValueError(f"partial parameter update: {rid}")
        for artifact in ("best","latest"):
            row=summary["artifacts"][artifact]
            if Path(row["path"]).resolve()!=folder.joinpath(f"{artifact}.pth").resolve() or sha(Path(row["path"]))!=row["sha256"]:raise ValueError(f"artifact drift: {rid}/{artifact}")
        for role in ("calibration","outer"):
            row=summary["artifacts"]["predictions"][role]
            if sha(Path(row["path"]))!=row["sha256"]:raise ValueError(f"prediction artifact drift: {rid}/{role}")
            with Path(row["path"]).open(newline="") as stream:
                records=list(csv.DictReader(stream))
            if len(records)!=role_counts[role] or any(not math.isfinite(float(x["p_future_raw"])) for x in records):raise ValueError(f"prediction rows invalid: {rid}/{role}")
        best=torch.load(folder/"best.pth",map_location="cpu",weights_only=False);latest=torch.load(folder/"latest.pth",map_location="cpu",weights_only=False)
        if best["run_identity_sha256"]!=summary["run_identity_sha256"] or best["run_config"]!=cfg or latest["run_config"]!=cfg:raise ValueError(f"checkpoint identity mismatch: {rid}")
        expected_keys={"gru.weight_ih_l0","gru.weight_hh_l0","gru.bias_ih_l0","gru.bias_hh_l0","risk.weight","risk.bias"}
        if set(best["model_state"])!=expected_keys or set(latest["model_state"])!=expected_keys:raise ValueError(f"unexpected trainable module: {rid}")
        history=summary["history"];losses=[float(x["selection_unweighted_bce"]) for x in history]
        if not history or not all(math.isfinite(x) for x in losses) or len(history)>100:raise ValueError(f"history invalid: {rid}")
        earliest=min(range(len(losses)),key=losses.__getitem__)+1
        if summary["best_epoch"]!=earliest or abs(summary["best_selection_loss"]-losses[earliest-1])>1e-12:raise ValueError(f"checkpoint selection drift: {rid}")
        if not torch.equal(best["normalization"]["mean"],data["normalization_C"]["mean"][:cfg["input_dim"]]) or not torch.equal(best["normalization"]["std"],data["normalization_C"]["std"][:cfg["input_dim"]]):raise ValueError(f"normalizer drift: {rid}")
        canonical,evidence=trainer.canonical_initial_state(run["group"],run["seed"],data["group_input_dims"],128)
        if evidence!=cfg["initialization"]:raise ValueError(f"fresh initialization identity mismatch: {rid}")
        initial_by_seed.setdefault(run["seed"],{})[run["group"]]=canonical
        attempt=queue["attempts"][rid];receipt_path=queue_root/f"{rid}.receipt.attempt{attempt}.json";receipt=json.loads(receipt_path.read_text())
        if receipt.get("status")!="complete" or receipt.get("inventory_sha256")!=sha(a.inventory) or receipt.get("id")!=rid:raise ValueError(f"receipt mismatch: {rid}")
        expected_argv=[str(Path(inv.get("python",receipt["argv"][0]))),str(trainer_path),"--data",str(Path(inv["prepared_data"])),"--group",run["group"],"--seed",str(run["seed"]),"--output",run["output"],"--device","cuda:0","--execute-formal"]
        if receipt["argv"]!=expected_argv:raise ValueError(f"unexpected formal argv: {rid}")
        runs.append({"id":rid,"summary_sha256":sha(summary_path),"best_sha256":sha(folder/"best.pth"),"latest_sha256":sha(folder/"latest.pth"),"receipt_sha256":sha(receipt_path),"epochs":len(history),"best_epoch":summary["best_epoch"],"best_selection_loss":summary["best_selection_loss"]})
    for seed,states in initial_by_seed.items():
        c=states["C_force_delta"]
        for group in ("A_visual","B_force"):
            state=states[group];dim=data["group_input_dims"][group]
            if not torch.equal(state["gru.weight_ih_l0"],c["gru.weight_ih_l0"][:,:dim]):raise ValueError(f"input prefix initialization mismatch: {seed}/{group}")
            for key in ("gru.weight_hh_l0","gru.bias_ih_l0","gru.bias_hh_l0","risk.weight","risk.bias"):
                if not torch.equal(state[key],c[key]):raise ValueError(f"shared initialization mismatch: {seed}/{group}/{key}")
    result={"format":"round6_formal_training_independent_audit_v1","status":"pass","inventory":{"path":str(a.inventory.resolve()),"sha256":sha(a.inventory)},"queue_state_sha256":sha(queue_root/"QUEUE_STATE.json"),"trainer_sha256":sha(trainer_path),"trainer_code_bundle_sha256":source_sha,"prepared_data_sha256":sha(Path(inv["prepared_data"])),"checks":{"exact_nine_identities":True,"all_frozen_inputs_match":True,"all_receipts_match":True,"all_runs_formal_complete":True,"only_gru_and_risk_parameters":True,"all_gradients_finite_and_all_tensors_updated":True,"fit_only_normalization_exact":True,"earliest_strict_selection_minimum":True,"shared_seed_initialization_exact":True,"calibration_outer_prediction_rows_complete":True},"runs":runs}
    atomic_json(a.output.resolve(),result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
