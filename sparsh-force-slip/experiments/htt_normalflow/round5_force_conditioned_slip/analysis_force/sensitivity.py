#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score

HERE=Path(__file__).resolve().parent;CURRENT=HERE.parent/"current";sys.path.insert(0,str(CURRENT))
from common import atomic_json, sha256_file  # noqa: E402
from data import SlipFrames, load_cache, require_cache_audit, role_entries  # noqa: E402
from metrics import partial_tpr_auc_0_0p1  # noqa: E402
from models import ForceConditionedSlip  # noqa: E402
from sources import load_decoupled_decoder, load_r3_slip_branch  # noqa: E402
from train_slip import fit_condition_data, load_force_predictions  # noqa: E402
sys.path.pop(0)
from analysis_common import atomic_csv, circular_mismatch, protocol, sha256, verify_receipt_inventory  # noqa: E402


def classification(labels: np.ndarray, probability: np.ndarray) -> dict:
    pred=probability>=.5
    return {"partial_tpr_auc_0_0p1":partial_tpr_auc_0_0p1(labels,probability),
            "average_precision":float(average_precision_score(labels,probability)),
            "balanced_accuracy_at_0p5":float(balanced_accuracy_score(labels,pred))}


def predict(model, tokens: np.ndarray, condition: np.ndarray | None, device: str, batch_size: int) -> np.ndarray:
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(tokens),batch_size):
            token=torch.from_numpy(tokens[start:start+batch_size]).to(device).float()
            cond=None if condition is None else torch.from_numpy(condition[start:start+batch_size]).to(device).float()
            chunks.append(torch.softmax(model(token,cond),dim=1)[:,1].cpu().numpy())
    result=np.concatenate(chunks)
    if not np.isfinite(result).all():raise ValueError("non-finite sensitivity probability")
    return result


def sort_examples(tokens, conditions, labels, episodes, frames):
    order=sorted(range(len(labels)),key=lambda i:(episodes[i],frames[i]))
    return (np.asarray(tokens,np.float32)[order],[conditions[i] for i in order],np.asarray(labels,np.int64)[order],
            [episodes[i] for i in order],[frames[i] for i in order])


def run(args: argparse.Namespace) -> dict:
    cfg=protocol();cache=load_cache(args.contract.resolve());audit=require_cache_audit(args.cache_audit.resolve(),cache,False)
    selected=torch.load(args.checkpoint.resolve(),map_location="cpu",weights_only=False);config=selected.get("config",{})
    if selected.get("format")!="round5_force_conditioned_slip_v1" or config.get("variant")!=args.variant or config.get("fold")!=args.fold or int(config.get("seed",-1))!=args.seed or config.get("smoke"):
        raise ValueError("sensitivity requires the matching formal slip checkpoint")
    summary=json.loads(args.training_summary.read_text());receipt=args.training_summary.with_name(args.training_summary.name+".scheduler_receipt.json")
    identity_checks={"summary":summary.get("status")=="complete" and summary.get("smoke") is False and summary.get("variant")==args.variant and summary.get("fold")==args.fold and int(summary.get("seed",-1))==args.seed,
                     "best_path":Path(summary.get("best_checkpoint","")).resolve()==args.checkpoint.resolve(),"best_sha":summary.get("best_checkpoint_sha256")==sha256_file(args.checkpoint.resolve()),
                     "config":all(summary.get(k)==v for k,v in config.items()),"contract":config.get("cache_manifest_sha256")==cache["manifest_sha256"],
                     "audit":config.get("cache_audit_sha256")==sha256_file(args.cache_audit.resolve()),"source":config.get("source_checkpoint_sha256")==sha256_file(args.source_checkpoint.resolve()),
                     "base":config.get("base_slip_checkpoint_sha256")==sha256_file(args.base_slip_checkpoint.resolve())}
    receipt_proof=verify_receipt_inventory(args.training_summary,args.inventory)
    if not all(identity_checks.values()):raise ValueError(f"formal slip provenance mismatch: {[k for k,v in identity_checks.items() if not v]}")
    decoder,_=load_decoupled_decoder(args.source_checkpoint.resolve());base=load_r3_slip_branch(decoder,args.base_slip_checkpoint.resolve())
    model=ForceConditionedSlip(base,args.variant,hidden_dim=64);model.load_state_dict(selected["model_state"],strict=True);model.eval().requires_grad_(False).to(args.device)
    entries=role_entries(cache,args.fold,args.role,"slip");condition_map=None;normalization=None;force_manifest=None
    if args.variant!="V":
        if args.force_predictions is None:raise ValueError("force variant requires --force-predictions")
        forces,force_manifest=load_force_predictions(args.force_predictions.resolve(),cache,args.variant,args.fold,args.seed,False)
        condition_map,normalization=fit_condition_data(forces,role_entries(cache,args.fold,"train","slip"))
        saved=selected.get("condition_normalization")
        if saved is None or any(not np.allclose(np.asarray(saved[k]),np.asarray(v),rtol=0,atol=0) for k,v in normalization.items()):
            raise ValueError("reconstructed train-only condition normalization differs from checkpoint")
    dataset=SlipFrames(entries,condition_map,strict_start=cfg["slip_min_frame"],primary_only=True)
    tokens=[];conditions=[];labels=[];episodes=[];frames=[]
    for i in range(len(dataset)):
        token,condition,label,episode,frame,_=dataset[i];tokens.append(token.numpy());conditions.append(condition.numpy());labels.append(label);episodes.append(episode);frames.append(frame)
    tokens,conditions,labels,episodes,frames=sort_examples(tokens,conditions,labels,episodes,frames)
    original_condition=None if args.variant=="V" else np.asarray(conditions,np.float32)
    original=predict(model,tokens,original_condition,args.device,args.batch_size)
    predictions={"original":original};applicable=args.variant!="V"
    if applicable:
        predictions["masked_zero_train_standardized"]=predict(model,tokens,np.zeros_like(original_condition),args.device,args.batch_size)
        predictions["mismatched_half_cycle"]=predict(model,tokens,circular_mismatch(original_condition),args.device,args.batch_size)
    rows=[]
    for i in range(len(labels)):
        row={"episode_id":episodes[i],"frame":frames[i],"label":int(labels[i]),"p_original":float(original[i])}
        for key,value in predictions.items():
            if key!="original":row["p_"+key]=float(value[i])
        rows.append(row)
    output=args.output.resolve();atomic_csv(output/"predictions.csv",rows)
    result={"status":"complete","format":"round5_force_input_sensitivity_v1","variant":args.variant,"fold":args.fold,"seed":args.seed,"role":args.role,"count":len(labels),
            "protocol_sha256":sha256(HERE/"protocol.json"),"checkpoint_sha256":sha256(args.checkpoint.resolve()),"contract_sha256":sha256(args.contract.resolve()),"cache_audit_sha256":sha256(args.cache_audit.resolve()),
            "training_summary_sha256":sha256(args.training_summary.resolve()),**receipt_proof,
            "force_intervention_applicable":applicable,"interpretation":"descriptive model input reliance; not physical causality","metrics":{k:classification(labels,p) for k,p in predictions.items()},
            "probability_change":({k:{"mean_absolute":float(np.abs(p-original).mean()),"class_flip_at_0p5":float(((p>=.5)!=(original>=.5)).mean())} for k,p in predictions.items() if k!="original"} if applicable else {}),
            "mismatch_rule":cfg["sensitivity"]["mismatch"],"mask_rule":cfg["sensitivity"]["mask"],
            "condition_normalization":({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in saved.items()} if applicable else None),
            "force_prediction_manifest_sha256":force_manifest.get("manifest_sha256") if force_manifest else None,
            "predictions_csv":str((output/"predictions.csv").resolve()),"predictions_csv_sha256":sha256(output/"predictions.csv")}
    atomic_json(output/"metrics.json",result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--contract",type=Path,required=True);p.add_argument("--cache-audit",type=Path,required=True);p.add_argument("--inventory",type=Path,required=True);p.add_argument("--source-checkpoint",type=Path,required=True);p.add_argument("--base-slip-checkpoint",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--training-summary",type=Path,required=True);p.add_argument("--force-predictions",type=Path);p.add_argument("--variant",choices=("V","F-old","F-adapt"),required=True);p.add_argument("--fold",required=True);p.add_argument("--seed",type=int,required=True);p.add_argument("--role",choices=("calibration","validation"),required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=128);args=p.parse_args();print(json.dumps(run(args),indent=2))


if __name__=="__main__":main()
