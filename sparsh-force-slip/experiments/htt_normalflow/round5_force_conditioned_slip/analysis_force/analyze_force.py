#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CURRENT = HERE.parent / "current"
sys.path.insert(0, str(CURRENT))

from common import atomic_json as current_atomic_json  # noqa: E402
from data import ForceFrames, load_cache, require_cache_audit, role_entries  # noqa: E402
from models import ForceAdapter, FrozenOldForce  # noqa: E402
from sources import load_decoupled_decoder  # noqa: E402

sys.path.pop(0)
from analysis_common import atomic_csv, cluster_force_bootstrap, invariant_force, lagged_change_metrics, protocol, regression_metrics, sha256, verify_receipt_inventory  # noqa: E402


def validate_formal_force_parent(checkpoint: Path, summary_path: Path, contract: dict,
                                 cache_audit_path: Path, source_checkpoint: Path,
                                 fold: str, seed: int, inventory_path: Path) -> dict:
    checkpoint=checkpoint.resolve();summary_path=summary_path.resolve();source_checkpoint=source_checkpoint.resolve()
    parent=torch.load(checkpoint,map_location="cpu",weights_only=False);summary=json.loads(summary_path.read_text());config=parent.get("config",{})
    checks={
        "format":parent.get("format")=="round5_htt_native_force_adapter_v1",
        "fold_seed":config.get("fold")==fold and int(config.get("seed",-1))==seed,
        "formal":config.get("smoke") is False and summary.get("status")=="complete" and summary.get("smoke") is False,
        "summary_best_path":Path(summary.get("best_checkpoint","")).resolve()==checkpoint,
        "summary_best_sha":summary.get("best_checkpoint_sha256")==sha256(checkpoint),
        "summary_config":all(summary.get(key)==value for key,value in config.items()),
        "contract":config.get("cache_manifest_sha256")==contract["manifest_sha256"],
        "audit":config.get("cache_audit_sha256")==sha256(cache_audit_path.resolve()),
        "source":config.get("source_checkpoint_sha256")==sha256(source_checkpoint),
        "target":config.get("target")=="clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal",
        "head_reset":parent.get("provenance",{}).get("force_output_head_reinitialized") is True,
    }
    normalization=parent.get("normalization",{})
    checks["normalization"]=all(np.asarray(normalization.get(k)).shape==(3,) and np.isfinite(normalization[k]).all() for k in ("mean","std")) and np.all(np.asarray(normalization["std"])>=1e-6)
    verify_receipt_inventory(summary_path,inventory_path)
    if not all(checks.values()):raise ValueError(f"formal force parent provenance mismatch: {[k for k,v in checks.items() if not v]}")
    return parent


def evaluate_htt(args: argparse.Namespace) -> dict:
    cfg = protocol(); cache = load_cache(args.contract.resolve())
    if sha256(args.source_checkpoint.resolve())!=cfg["canonical_source_checkpoint_sha256"]:raise ValueError("historical source checkpoint identity mismatch")
    require_cache_audit(args.cache_audit.resolve(), cache, False)
    entries = role_entries(cache, args.fold, args.role, "force")
    decoder, _ = load_decoupled_decoder(args.source_checkpoint.resolve())
    parent = None
    if args.variant == "adapt":
        if args.training_summary is None:raise ValueError("adapt requires --training-summary")
        parent=validate_formal_force_parent(args.checkpoint,args.training_summary,cache,args.cache_audit,args.source_checkpoint,args.fold,args.seed,args.inventory)
        model = ForceAdapter(decoder); model.load_state_dict(parent["model_state"], strict=True)
        normalization = parent["normalization"]
    else:
        model = FrozenOldForce(decoder); normalization = None
    model.eval().requires_grad_(False).to(args.device)
    dataset = ForceFrames(entries, min_frame=cfg["force_min_frame"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    predictions=[]; targets=[]; episodes=[]; frames=[]
    with torch.inference_mode():
        for token, target, episode, frame in loader:
            pred=model(token.to(args.device).float())
            if normalization is not None:
                pred=pred*torch.as_tensor(normalization["std"],device=args.device)+torch.as_tensor(normalization["mean"],device=args.device)
            predictions.append(pred.cpu().numpy());targets.append(target.numpy());episodes.extend(episode);frames.extend(frame.numpy().tolist())
    pred=np.concatenate(predictions);target=np.concatenate(targets);ep=np.asarray(episodes);fr=np.asarray(frames)
    rows=[]
    for i in range(len(pred)):
        rows.append({"episode_id":ep[i],"frame":int(fr[i]),**{f"target_{j}":float(target[i,j]) for j in range(3)},**{f"prediction_{j}":float(pred[i,j]) for j in range(3)}})
    atomic_csv(args.output/"predictions.csv",rows)
    result={"status":"complete","format":"round5_htt_force_analysis_v1","variant":args.variant,"fold":args.fold,"seed":args.seed,"role":args.role,
            "protocol_sha256":sha256(HERE/"protocol.json"),"contract_sha256":sha256(args.contract.resolve()),"cache_audit_sha256":sha256(args.cache_audit.resolve()),
            "source_checkpoint_sha256":sha256(args.source_checkpoint.resolve()),"checkpoint_sha256":sha256(args.checkpoint.resolve()) if args.checkpoint else None,
            "coordinate_statement":("HTT native reference-relative clipped target coordinates" if args.variant=="adapt" else "historical source axes/scales; HTT per-axis supervised comparison is not scientifically valid"),
            "interpretation_boundary":"Fn, Ft, Fmag and force ratios are derived from a reference-relative clipped target; they are not a calibrated friction coefficient or physical stability margin.",
            "prediction_axis_std_n":pred.std(0).tolist(),
            "invariant_prediction_distribution":{k:{"mean_n":float(v.mean()),"std_n":float(v.std())} for k,v in invariant_force(pred).items()},
            "predictions_csv":str((args.output/"predictions.csv").resolve()),"predictions_csv_sha256":sha256(args.output/"predictions.csv")}
    if args.variant=="adapt":
        result["supervised_regression"]=regression_metrics(target,pred)
        target_std=target.std(0);prediction_std=pred.std(0);threshold=np.maximum(cfg["collapse"]["absolute_std_floor_n"],cfg["collapse"]["relative_to_target_std"]*target_std)
        invariant_target,invariant_prediction=invariant_force(target),invariant_force(pred)
        result["output_variance"]={"target_axis_std_n":target_std.tolist(),"prediction_axis_std_n":prediction_std.tolist(),
                                   "target_invariant_std_n":{k:float(v.std()) for k,v in invariant_target.items()},"prediction_invariant_std_n":{k:float(v.std()) for k,v in invariant_prediction.items()}}
        collapsed=prediction_std<threshold
        result["collapse_check"]={"axis_threshold_std_n":threshold.tolist(),"axis_near_constant":collapsed.tolist(),"all_axes_near_constant":bool(collapsed.all()),"pass":not bool(collapsed.all())}
        train_mean=np.asarray(normalization["mean"],dtype=np.float32)
        result["no_training_baselines"]={
            "train_mean_constant":regression_metrics(target,np.broadcast_to(train_mean,target.shape)),
            "zero_force":regression_metrics(target,np.zeros_like(target)),
        }
        result["improvement_over_train_mean"]={
            "axis_rmse_n":[float(a-b) for a,b in zip(result["no_training_baselines"]["train_mean_constant"]["axis"]["rmse_n"],result["supervised_regression"]["axis"]["rmse_n"])],
            "invariant_rmse_n":{key:float(result["no_training_baselines"]["train_mean_constant"]["invariant"][key]["rmse_n"]-result["supervised_regression"]["invariant"][key]["rmse_n"]) for key in ("Fn","Ft","Fmag")},
        }
        result["lag5_change_regression"]=lagged_change_metrics(ep,fr,target,pred,cfg["force_delta_lag"])
        saturated=np.isclose(np.abs(target),cfg["saturation_absolute_n"],atol=1e-6)
        result["target_saturation"]={"per_axis_fraction":saturated.mean(0).tolist(),"any_axis_fraction":float(saturated.any(1).mean()),
                                     "saturated_axis_mae_n":[float(np.abs(pred[:,j]-target[:,j])[saturated[:,j]].mean()) if saturated[:,j].any() else None for j in range(3)]}
        split=json.loads(Path(cache["split_manifest"]).read_text());group_by_episode={row["id"]:row["leakage_group"] for row in split["episodes"]}
        result["leakage_group_bootstrap_95ci"]=cluster_force_bootstrap(np.asarray([group_by_episode[x] for x in ep]),target,pred,train_mean)
        result["per_episode"]={episode:regression_metrics(target[ep==episode],pred[ep==episode]) for episode in sorted(set(ep.tolist()))}
    result["status"]="complete" if result.get("collapse_check",{}).get("pass",True) else "needs_attention_output_collapse"
    current_atomic_json(args.output/"metrics.json",result);return result


def evaluate_source(args: argparse.Namespace) -> dict:
    if sha256(args.source_checkpoint.resolve())!=protocol()["canonical_source_checkpoint_sha256"]:raise ValueError("historical source checkpoint identity mismatch")
    payload=torch.load(args.features.resolve(),map_location="cpu",weights_only=False)
    required={"force_gt_n","force_pred_n","metadata","checkpoint","encoder_checkpoint_payload_epoch","split"}
    if required-payload.keys():raise ValueError(f"source cache missing {sorted(required-payload.keys())}")
    target=payload["force_gt_n"].float().numpy();pred=payload["force_pred_n"].float().numpy()
    if len(payload["metadata"])!=len(target):raise ValueError("source metadata length mismatch")
    expected_checkpoint=Path(payload["checkpoint"]).resolve()
    if expected_checkpoint!=args.source_checkpoint.resolve() or not expected_checkpoint.is_file():raise ValueError("source cache checkpoint mismatch")
    result={"status":"complete","format":"round5_source_force_regression_v1","split":payload["split"],"protocol_sha256":sha256(HERE/"protocol.json"),"features_path":str(args.features.resolve()),"features_sha256":sha256(args.features.resolve()),
            "source_checkpoint":str(expected_checkpoint),"source_checkpoint_sha256":sha256(expected_checkpoint),"source_epoch":int(payload["encoder_checkpoint_payload_epoch"]),
            "regression":regression_metrics(target,pred),"coordinate_statement":"source-domain GT and prediction share the source cache convention; do not merge axis metrics with HTT",
            "adapted_htt_model_evaluation":"not_in_this_cache: pooled z cannot drive the force pooler; use source-adapt on the preserved complete source validation loader",
            "provenance_limit":"historical cache records source checkpoint path and epoch but not its generation-time content SHA; the current canonical checkpoint identity is checked, so these metrics are retained as historical reference rather than strict regeneration proof"}
    current_atomic_json(args.output.resolve(),result);return result


def evaluate_source_adapt(args: argparse.Namespace) -> dict:
    """Run the HTT-adapted force branch on the complete historical source val split.

    The output axes are intentionally not scored against source axes. Invariant
    magnitudes are descriptive because reference subtraction may still differ.
    """
    scripts = HERE.parents[3] / "scripts"
    sys.path.insert(0, str(scripts))
    import phase2_b_multitask as p2
    if sha256(args.source_checkpoint.resolve())!=protocol()["canonical_source_checkpoint_sha256"]:raise ValueError("historical source checkpoint identity mismatch")
    decoder,_=load_decoupled_decoder(args.source_checkpoint.resolve())
    checkpoints=[]
    if args.all_adapt_root:
        for fold in range(1,5):
            for seed in (20260914,20260915,20260916):checkpoints.append((f"p{fold}_s{seed}",args.all_adapt_root/f"p{fold}_s{seed}"/"best.pth"))
    else:checkpoints=[("single",args.checkpoint)]
    models={};parents={};parent_proofs={}
    for key,path in checkpoints:
        if path is None or not path.is_file():raise FileNotFoundError(f"missing adapted checkpoint: {path}")
        parent=validate_formal_force_parent(path,path.parent/"training_summary.json",args.contract_payload,args.cache_audit,args.source_checkpoint,
                                            f"htt_leave_p{key[1]}" if key!="single" else args.fold,
                                            int(key.split("_s")[-1]) if key!="single" else args.seed,args.inventory);pc=parent["config"]
        adapted=ForceAdapter(decoder);adapted.load_state_dict(parent["model_state"],strict=True);adapted.eval().requires_grad_(False).to(args.device)
        summary_path=path.parent/"training_summary.json";models[key]=(adapted,torch.as_tensor(parent["normalization"]["mean"],device=args.device),torch.as_tensor(parent["normalization"]["std"],device=args.device),path.resolve());parents[key]=parent
        parent_proofs[key]={"training_summary":str(summary_path.resolve()),"training_summary_sha256":sha256(summary_path.resolve()),**verify_receipt_inventory(summary_path,args.inventory)}
    torch.manual_seed(42)
    encoder_bundle=p2.FrozenEncoderSharedForceSlip("mae",decoder_variant="decoupled").to(args.device)
    encoder_bundle.encoder.eval().requires_grad_(False)
    old_model=FrozenOldForce(decoder).eval().requires_grad_(False).to(args.device)
    all_target=[];all_old=[];all_prediction={key:[] for key in models};per_dataset={key:{} for key in models};old_per_dataset={};source_inputs={}
    for name in p2.VAL_DATASETS:
        for input_path in (p2.DERIVED_ROOT/name/"dataset_slip_forces.pkl",p2.DERIVED_ROOT/name/"source_manifest.json"):
            if not input_path.is_file():raise FileNotFoundError(f"missing source validation input: {input_path}")
            source_inputs[str(input_path.resolve())]=sha256(input_path.resolve())
        loader=p2.make_loader([name],0,args.batch_size,args.workers,False,False,"mae")
        targets=[];old_predictions=[];predictions={key:[] for key in models}
        with torch.inference_mode():
            for batch in loader:
                image=batch["image"].to(args.device).float();tokens=encoder_bundle.encoder(image)
                target=batch["force"].to(args.device).float()*batch["force_scale"].to(args.device).float();targets.append(target.cpu().numpy())
                old_predictions.append(old_model(tokens).cpu().numpy())
                for key,(adapted,mean,std,_) in models.items():predictions[key].append((adapted(tokens)*std+mean).cpu().numpy())
        target=np.concatenate(targets);old_prediction=np.concatenate(old_predictions);all_target.append(target);all_old.append(old_prediction);ti=invariant_force(target)
        old_per_dataset[name]=regression_metrics(target,old_prediction)
        for key in models:
            prediction=np.concatenate(predictions[key]);all_prediction[key].append(prediction);pi=invariant_force(prediction)
            per_dataset[key][name]={"count":len(target),"target_invariant_mean_n":{k:float(v.mean()) for k,v in ti.items()},
                                    "prediction_invariant_mean_n":{k:float(v.mean()) for k,v in pi.items()},
                                    "descriptive_invariant_difference":{k:{"bias_n":float((pi[k]-ti[k]).mean()),"mae_n":float(np.abs(pi[k]-ti[k]).mean())} for k in ti}}
    target=np.concatenate(all_target);old_prediction=np.concatenate(all_old);ti=invariant_force(target);model_results={}
    for key,(_,_,_,path) in models.items():
        prediction=np.concatenate(all_prediction[key]);pi=invariant_force(prediction);pc=parents[key]["config"]
        model_results[key]={"fold":pc["fold"],"seed":pc["seed"],"checkpoint":str(path),"checkpoint_sha256":sha256(path),"formal_parent_proof":parent_proofs[key],
                            "aggregate":{"target_invariant_mean_n":{k:float(v.mean()) for k,v in ti.items()},"prediction_invariant_mean_n":{k:float(v.mean()) for k,v in pi.items()},
                                         "descriptive_invariant_difference":{k:{"bias_n":float((pi[k]-ti[k]).mean()),"mae_n":float(np.abs(pi[k]-ti[k]).mean())} for k in ti}},"per_dataset":per_dataset[key]}
    result={"status":"complete","format":"round5_source_val_adapt_distribution_v1","split":"complete original validation datasets",
            "datasets":list(p2.VAL_DATASETS),"count":len(target),"frame_population":{"total":len(target),"per_dataset":{name:old_per_dataset[name]["count"] for name in p2.VAL_DATASETS}},"models":model_results,"historical_old_head_fresh_regression":{"aggregate":regression_metrics(target,old_prediction),"per_dataset":old_per_dataset},"encoder_passes_per_batch":1,"adapted_heads_per_encoder_output":len(models),
            "source_checkpoint_sha256":sha256(args.source_checkpoint.resolve()),"derived_root":str(p2.DERIVED_ROOT),
            "coordinate_statement":"HTT-adapted output and source GT do not have a verified signed-axis/reference map. No per-axis regression metric is reported; even invariant differences are descriptive domain-shift evidence, not calibrated physical error.",
            "input_provenance":{"loader":"phase2_b_multitask.make_loader VAL_DATASETS encoder=mae slip_horizon=0","dataset_count":len(p2.VAL_DATASETS),"complete_dataset_inputs_sha256":source_inputs}}
    current_atomic_json(args.output.resolve(),result);return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("htt");a.add_argument("--contract",type=Path,required=True);a.add_argument("--cache-audit",type=Path,required=True);a.add_argument("--inventory",type=Path,required=True);a.add_argument("--source-checkpoint",type=Path,required=True);a.add_argument("--variant",choices=("old","adapt"),required=True);a.add_argument("--checkpoint",type=Path);a.add_argument("--training-summary",type=Path);a.add_argument("--fold",required=True);a.add_argument("--seed",type=int,required=True);a.add_argument("--role",choices=("calibration","validation"),required=True);a.add_argument("--output",type=Path,required=True);a.add_argument("--device",default="cuda:0");a.add_argument("--batch-size",type=int,default=128);a.add_argument("--workers",type=int,default=2)
    a=sub.add_parser("source");a.add_argument("--features",type=Path,required=True);a.add_argument("--source-checkpoint",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("source-adapt");group=a.add_mutually_exclusive_group(required=True);group.add_argument("--checkpoint",type=Path);group.add_argument("--all-adapt-root",type=Path);a.add_argument("--contract",type=Path,required=True);a.add_argument("--cache-audit",type=Path,required=True);a.add_argument("--inventory",type=Path,required=True);a.add_argument("--fold");a.add_argument("--seed",type=int);a.add_argument("--source-checkpoint",type=Path,required=True);a.add_argument("--output",type=Path,required=True);a.add_argument("--device",default="cuda:0");a.add_argument("--batch-size",type=int,default=128);a.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    if args.command=="htt" and args.variant=="adapt" and (args.checkpoint is None or args.training_summary is None):p.error("adapt requires --checkpoint and --training-summary")
    if args.command=="source-adapt":
        args.contract_payload=load_cache(args.contract.resolve());require_cache_audit(args.cache_audit.resolve(),args.contract_payload,False)
        if args.checkpoint and (args.fold is None or args.seed is None):p.error("single source-adapt checkpoint requires --fold and --seed")
    result=evaluate_htt(args) if args.command=="htt" else (evaluate_source(args) if args.command=="source" else evaluate_source_adapt(args));print(json.dumps(result,indent=2))


if __name__=="__main__":main()
