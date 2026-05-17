#!/usr/bin/env python3
"""Finalize Phase4-1 MAE multi-seed report.

The script reuses existing seed0 artifacts and evaluates/trains only missing
seed1/seed2 derived artifacts. It never mutates raw datasets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_b_multitask as p2  # noqa: E402
import phase4_paper_experiments as p4  # noqa: E402

REPORT_ROOT = Path("/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4")
PHASE3_1_REPORT = Path("/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/phase3_1_decoupled_multitask_report.json")
PHASE3_1_SEPARATE = Path("/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/eval_cache/a_mae_allsource_val.json")
PHASE3_1_DECOUPLED = Path("/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/eval_cache/decoupled_mae_phase3_1_decoupled_gsmini_20260516_154730_allsource_val.json")
PHASE3_2_WORLD = Path("/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_2_20260517_014013/phase3_world_model_summary.json")
PHASE3_2_WORLD_CHECKPOINT = Path("/vla1/zjy/sparsh_runs/force_slip_phase3/phase3_2_world_model_mae_20260517_014013/multihorizon/checkpoints/best.pth")
PHASE3_1_DECOUPLED_CKPT = Path("/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=p2.json_default), encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def aggregate_metrics(values: list[float | None]) -> dict[str, float | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    import statistics
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals), "n": len(vals)}


def get_metric(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def resolve_phase4_exp(kind: str, seed_label: str, stamp: str) -> Path:
    pattern = f"latest:*phase4_1_mae_{kind}_{seed_label}_{stamp}"
    return p2.resolve_experiment_dir(pattern)


def seed0_row() -> dict[str, Any]:
    sep = read_json(PHASE3_1_SEPARATE)
    dec = read_json(PHASE3_1_DECOUPLED)
    world = read_json(PHASE3_2_WORLD)
    multih = world["stage_reports"]["multihorizon"]
    return {
        "seed_label": "seed0",
        "seed": 42,
        "separate_eval": sep,
        "decoupled_eval": dec,
        "world_report": multih,
        "force_exp": sep.get("force_experiment"),
        "slip_exp": sep.get("slip_experiment"),
        "decoupled_checkpoint": str(PHASE3_1_DECOUPLED_CKPT),
        "world_checkpoint": str(PHASE3_2_WORLD_CHECKPOINT),
        "reused": True,
    }


def phase4_seed_row(seed_label: str, seed: int, stamp: str, args: argparse.Namespace, report_dir: Path) -> dict[str, Any]:
    eval_dir = report_dir / "eval_cache"
    eval_dir.mkdir(parents=True, exist_ok=True)
    force_exp = resolve_phase4_exp("force", seed_label, stamp)
    slip_exp = resolve_phase4_exp("slip", seed_label, stamp)
    sep_json = eval_dir / f"separate_{seed_label}.json"
    if sep_json.exists():
        sep_eval = read_json(sep_json)
    else:
        sep_eval = p4.evaluate_separate_pair("mae", force_exp, slip_exp, sep_json, args.eval_batch_size, args.num_workers)
    run_id = f"phase4_1_mae_decoupled_{seed_label}_{stamp}"
    selection_json = eval_dir / f"checkpoint_selection_decoupled_{seed_label}.json"
    if selection_json.exists():
        selection = read_json(selection_json)
    else:
        selection = p2.select_b_checkpoint_for_gate(run_id, "mae", "decoupled", sep_eval)
        write_json(selection_json, selection)
    checkpoint = Path(selection["checkpoint"])
    if not checkpoint.exists():
        # Fallback to the final checkpoint if the selector chose an absent best-F1 alias.
        summary_path = Path("/vla1/zjy/sparsh_runs/force_slip_phase2") / run_id / "mae_decoupled_multitask" / "training_summary.json"
        summary = read_json(summary_path)
        checkpoint = Path(summary["final_checkpoint"])
    dec_json = eval_dir / f"decoupled_{seed_label}.json"
    if dec_json.exists():
        dec_eval = read_json(dec_json)
    else:
        dec_eval = p4.evaluate_decoupled_checkpoint(checkpoint, dec_json, args.eval_batch_size, args.num_workers)
    world_json = report_dir / f"phase4_1_world_{seed_label}_{stamp}_report.json"
    if world_json.exists():
        world_report = read_json(world_json)
    else:
        feature_run_id = f"phase4_1_world_features_{seed_label}_{stamp}"
        p4.ensure_decoupled_features(feature_run_id, checkpoint, tuple(args.horizons), args.precompute_batch_size, args.num_workers)
        world_report = p4.train_future_head(
            feature_run_id=feature_run_id,
            experiment_name=f"phase4_1_world_{seed_label}_{stamp}",
            input_mode="full",
            horizons=tuple(args.horizons),
            max_epochs=args.world_epochs,
            batch_size=args.train_batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seed=seed,
            wandb_mode=args.wandb_mode,
            report_dir=report_dir,
        )
        # train_future_head writes experiment-specific JSON; write a stable alias too.
        write_json(world_json, world_report)
    return {
        "seed_label": seed_label,
        "seed": seed,
        "separate_eval": sep_eval,
        "decoupled_eval": dec_eval,
        "world_report": world_report,
        "force_exp": str(force_exp),
        "slip_exp": str(slip_exp),
        "checkpoint_selection": selection,
        "decoupled_checkpoint": str(checkpoint),
        "world_checkpoint": world_report.get("checkpoint"),
        "reused": False,
    }


def build_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "separate_force_rmse_mean_N": "separate_eval.aggregate.force_rmse_mean_N",
        "separate_slip_f1": "separate_eval.aggregate.slip_f1",
        "separate_slip_accuracy": "separate_eval.aggregate.slip_accuracy",
        "decoupled_force_rmse_mean_N": "decoupled_eval.aggregate.force_rmse_mean_N",
        "decoupled_slip_f1": "decoupled_eval.aggregate.slip_f1",
        "decoupled_slip_accuracy": "decoupled_eval.aggregate.slip_accuracy",
        "world_current_force_rmse_mean_N": "world_report.base_current_metrics_val.force_rmse_mean_N",
        "world_current_slip_f1": "world_report.base_current_metrics_val.slip_f1",
        "world_current_slip_accuracy": "world_report.base_current_metrics_val.slip_accuracy",
        "world_H1_future_slip_f1": "world_report.best_val.H1.future_slip_f1",
        "world_H1_future_slip_auroc": "world_report.best_val.H1.future_slip_auroc",
        "world_H1_future_slip_auprc": "world_report.best_val.H1.future_slip_auprc",
        "world_H1_future_stability_auroc": "world_report.best_val.H1.future_stability_auroc",
        "world_H1_future_stability_auprc": "world_report.best_val.H1.future_stability_auprc",
        "world_H1_stability_ece": "world_report.best_val.H1.stability_calibration_error",
        "world_H3_future_slip_f1": "world_report.best_val.H3.future_slip_f1",
        "world_H3_future_slip_auroc": "world_report.best_val.H3.future_slip_auroc",
        "world_H3_future_slip_auprc": "world_report.best_val.H3.future_slip_auprc",
        "world_H5_future_slip_f1": "world_report.best_val.H5.future_slip_f1",
        "world_H5_future_slip_auroc": "world_report.best_val.H5.future_slip_auroc",
        "world_H5_future_slip_auprc": "world_report.best_val.H5.future_slip_auprc",
    }
    out: dict[str, Any] = {}
    for key, path in metric_paths.items():
        vals = [get_metric(row, path) for row in rows]
        out[key] = aggregate_metrics(vals)
    return out


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase4-1 MAE Multi-seed Report",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- stamp: `{payload['stamp']}`",
        "- seeds: seed0=42 reused from Phase3, seed1=43, seed2=44",
        "- raw_data_modified: `False`",
        "- split: Phase1 derived all-source train/val split",
        "",
        "## Per-seed metrics",
        "",
        "| seed | separate force RMSE | separate slip F1 | separate slip acc | decoupled force RMSE | decoupled slip F1 | world H1 F1 | world H1 AUROC | world H1 AUPRC | world H1 ECE | H3 F1 | H5 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['seed_label']} | {fmt(get_metric(row,'separate_eval.aggregate.force_rmse_mean_N'))} | {fmt(get_metric(row,'separate_eval.aggregate.slip_f1'))} | {fmt(get_metric(row,'separate_eval.aggregate.slip_accuracy'))} | "
            f"{fmt(get_metric(row,'decoupled_eval.aggregate.force_rmse_mean_N'))} | {fmt(get_metric(row,'decoupled_eval.aggregate.slip_f1'))} | {fmt(get_metric(row,'world_report.best_val.H1.future_slip_f1'))} | "
            f"{fmt(get_metric(row,'world_report.best_val.H1.future_slip_auroc'))} | {fmt(get_metric(row,'world_report.best_val.H1.future_slip_auprc'))} | {fmt(get_metric(row,'world_report.best_val.H1.stability_calibration_error'))} | "
            f"{fmt(get_metric(row,'world_report.best_val.H3.future_slip_f1'))} | {fmt(get_metric(row,'world_report.best_val.H5.future_slip_f1'))} |"
        )
    lines += ["", "## Mean ± std", "", "| metric | mean | std | n |", "|---|---:|---:|---:|"]
    for key, stat in payload["metric_summary"].items():
        lines.append(f"| {key} | {fmt(stat['mean'])} | {fmt(stat['std'])} | {stat['n']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- This phase checks whether the MAE paper-facing route is stable across three seeds rather than being a single-run artifact.",
        "- Separate force/slip remains the SPARSH-style current-task baseline; decoupled multitask is judged by force protection and its ability to feed the future-stability head.",
        "- The world-model row uses only derived feature caches and predicts future slip/instability, not grasp success.",
        "",
        "## Artifacts",
        "",
    ]
    for row in payload["rows"]:
        lines.append(f"- {row['seed_label']} force_exp: `{row.get('force_exp')}`")
        lines.append(f"- {row['seed_label']} slip_exp: `{row.get('slip_exp')}`")
        lines.append(f"- {row['seed_label']} decoupled_checkpoint: `{row.get('decoupled_checkpoint')}`")
        lines.append(f"- {row['seed_label']} world_checkpoint: `{row.get('world_checkpoint')}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--report-stamp", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--precompute-batch-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--world-epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    report_stamp = args.report_stamp or args.stamp
    report_dir = REPORT_ROOT / f"phase4_1_{report_stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [seed0_row()]
    rows.append(phase4_seed_row("seed1", 43, args.stamp, args, report_dir))
    rows.append(phase4_seed_row("seed2", 44, args.stamp, args, report_dir))
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase4_1_mae_multiseed",
        "stamp": args.stamp,
        "report_stamp": report_stamp,
        "rows": rows,
        "metric_summary": build_metric_summary(rows),
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase4_1_mae_multiseed_report.json"
    md_path = report_dir / "phase4_1_mae_multiseed_report.md"
    payload["json_path"] = str(json_path)
    payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render(payload), encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
