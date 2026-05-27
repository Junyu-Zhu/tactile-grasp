#!/usr/bin/env python3
"""Run best-lambda future input ablation and consolidate reports.

This script depends on the Phase-A lambda sweep report. It writes only derived
run/report artifacts and never mutates raw datasets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase4_paper_experiments as p4  # noqa: E402
import phase5_friction_stability as p5  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
PHASE_A_ROOT = WORKSPACE / "reports/phase_lambda_decoupled"
REPORT_ROOT = WORKSPACE / "reports/phase_lambda_decoupled_future_ablation"
FINAL_SUMMARY = WORKSPACE / "reports/phase_lambda_decoupled_summary.md"
OLD_PHASE4_JSON = WORKSPACE / "reports/phase4/phase4_3_20260517_171500/phase4_3_world_model_input_ablation_report.json"
OLD_PHASE5_JSON = WORKSPACE / "reports/phase5/phase5_2_friction_future_head/phase5_2_friction_future_head_report.json"
WANDB_BASE = "https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=p2.json_default), encoding="utf-8")


def fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if math.isnan(f) or math.isinf(f):
        return "n/a"
    return f"{f:.{digits}f}"


def get_path(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def condition_row(s: dict[str, Any]) -> dict[str, Any]:
    best = s.get("best_val") or {}
    hrows = {f"H{h}": best.get(f"H{h}", {}) for h in best.get("horizons", [1, 3, 5])}
    return {
        "condition": s.get("condition") or s.get("input_mode"),
        "feature_kind": s.get("feature_kind") or s.get("manifest_feature_kind"),
        "input_mode": s.get("input_mode") or s.get("condition"),
        "checkpoint": s.get("checkpoint"),
        "json_path": s.get("json_path"),
        "md_path": s.get("md_path"),
        "wandb_name": s.get("wandb_name"),
        "wandb_url": s.get("wandb_url"),
        "best_epoch": s.get("best_epoch"),
        "H1_F1": get_path(hrows, "H1", "future_slip_f1"),
        "H1_AUPRC": get_path(hrows, "H1", "future_slip_auprc"),
        "H1_ECE": get_path(hrows, "H1", "stability_calibration_error"),
        "H3_F1": get_path(hrows, "H3", "future_slip_f1"),
        "H3_AUPRC": get_path(hrows, "H3", "future_slip_auprc"),
        "H3_ECE": get_path(hrows, "H3", "stability_calibration_error"),
        "H5_F1": get_path(hrows, "H5", "future_slip_f1"),
        "H5_AUPRC": get_path(hrows, "H5", "future_slip_auprc"),
        "H5_ECE": get_path(hrows, "H5", "stability_calibration_error"),
        "H1_lead_mean": get_path(hrows, "H1", "lead_time_to_slip_onset_steps_mean"),
        "H1_high_risk_recall": get_path(hrows, "H1", "high_risk_model_recall_on_future_slip"),
        "H3_high_risk_recall": get_path(hrows, "H3", "high_risk_model_recall_on_future_slip"),
        "H5_high_risk_recall": get_path(hrows, "H5", "high_risk_model_recall_on_future_slip"),
    }


def best_condition(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    def score(r: dict[str, Any]) -> float:
        vals = [r.get("H1_AUPRC"), r.get("H3_AUPRC"), r.get("H5_AUPRC")]
        nums = [float(v) for v in vals if v is not None]
        return float(np.mean(nums)) if nums else -1.0
    return max(rows, key=score) if rows else None


def mean_auprc(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    vals = [row.get("H1_AUPRC"), row.get("H3_AUPRC"), row.get("H5_AUPRC")]
    nums = [float(v) for v in vals if v is not None]
    return float(np.mean(nums)) if nums else None


def build_comparison_rows(rows: list[dict[str, Any]], old: dict[str, Any]) -> list[dict[str, Any]]:
    old_phase4 = {r.get("condition"): r for r in old.get("phase4_conditions", [])}
    old_phase5 = {r.get("condition"): r for r in old.get("phase5_conditions", [])}
    out = []
    for r in rows:
        cond = r.get("condition")
        old_row = old_phase5.get(cond) if cond == "full_plus_q" else old_phase4.get(cond)
        new_m = mean_auprc(r)
        old_m = mean_auprc(old_row)
        delta = (new_m - old_m) if new_m is not None and old_m is not None else None
        out.append({"condition": cond, "new_mean_auprc": new_m, "old_mean_auprc": old_m, "delta_mean_auprc": delta, "old_row_available": old_row is not None})
    return out


def load_old_rows() -> dict[str, Any]:
    old: dict[str, Any] = {"phase4_json": str(OLD_PHASE4_JSON), "phase5_json": str(OLD_PHASE5_JSON), "phase4_conditions": [], "phase5_conditions": []}
    if OLD_PHASE4_JSON.exists():
        d = read_json(OLD_PHASE4_JSON)
        old["phase4_conditions"] = [condition_row(c) for c in d.get("conditions", [])]
    if OLD_PHASE5_JSON.exists():
        d = read_json(OLD_PHASE5_JSON)
        old["phase5_conditions"] = [condition_row(c) for c in d.get("conditions", [])]
    return old


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Best-λ Future Causal-input Ablation",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- stamp: `{payload['stamp']}`",
        f"- best_lambda: `{payload['best_lambda']}`",
        f"- stage_i_checkpoint: `{payload['stage_i_checkpoint']}`",
        f"- feature_run_id: `{payload['feature_run_id']}`",
        "- raw_data_modified: `False`",
        "",
        "## Best-λ Stage-II results",
        "",
        "| condition | H1 F1 | H1 AUPRC | H1 ECE | H1 lead | H1 HR recall | H3 F1 | H3 AUPRC | H3 ECE | H5 F1 | H5 AUPRC | H5 ECE | ckpt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["rows"]:
        lines.append(
            f"| {r['condition']} | {fmt(r.get('H1_F1'))} | {fmt(r.get('H1_AUPRC'))} | {fmt(r.get('H1_ECE'))} | {fmt(r.get('H1_lead_mean'))} | {fmt(r.get('H1_high_risk_recall'))} | "
            f"{fmt(r.get('H3_F1'))} | {fmt(r.get('H3_AUPRC'))} | {fmt(r.get('H3_ECE'))} | "
            f"{fmt(r.get('H5_F1'))} | {fmt(r.get('H5_AUPRC'))} | {fmt(r.get('H5_ECE'))} | `{r.get('checkpoint')}` |"
        )
    b = payload.get("best_condition") or {}
    lines += [
        "",
        "## Selection and comparison",
        "",
        f"- Best Stage-II condition by mean H1/H3/H5 AUPRC: `{b.get('condition', 'n/a')}`.",
        f"- Old λ=1.0 Phase4 report: `{payload['old_lambda1'].get('phase4_json')}`.",
        f"- Old λ=1.0 Phase5 report: `{payload['old_lambda1'].get('phase5_json')}`.",
        "- Interpret replacement conservatively: use best-λ as the main result only if it improves the paper-facing future metrics without violating current force/slip gates; otherwise keep it as an ablation.",
        "",
        "## Direct comparison to old λ=1.0 future ablation",
        "",
        "| condition | new mean AUPRC | old mean AUPRC | Δ mean AUPRC | old row found |",
        "|---|---:|---:|---:|---|",
    ]
    for c in payload.get('comparison_rows', []):
        lines.append(f"| {c['condition']} | {fmt(c.get('new_mean_auprc'))} | {fmt(c.get('old_mean_auprc'))} | {fmt(c.get('delta_mean_auprc'))} | {c.get('old_row_available')} |")
    lines += ["", "## W&B runs", ""]
    for r in payload["rows"]:
        if r.get("wandb_url"):
            lines.append(f"- {r['condition']}: [{r.get('wandb_name')}]({r.get('wandb_url')})")
    lines += ["", "## Commands", ""]
    for c in payload.get("commands", []):
        lines.append(f"```bash\n{c}\n```")
    return "\n".join(lines) + "\n"


def render_final_summary(phase_a: dict[str, Any], phase_b: dict[str, Any]) -> str:
    lines = [
        "# Phase Lambda Decoupled Summary",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Phase A report: `{phase_a.get('md_path')}`",
        f"- Phase B report: `{phase_b.get('md_path')}`",
        "",
        "## Phase A current force-slip λ sweep",
        "",
        "| λ | F RMSE | ΔF | SF1 | ΔSF1 drop | SA | ΔSA drop | gate | checkpoint |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in phase_a.get("rows", []):
        lines.append(
            f"| {float(r['lambda_slip']):.2f} | {fmt(r.get('force_rmse_mean_N'))} | {fmt(r.get('deltaF_pct'))}% | "
            f"{fmt(r.get('slip_f1'))} | {fmt(r.get('deltaSF1_pct_drop'))}% | {fmt(r.get('slip_accuracy'))} | {fmt(r.get('deltaSA_pct_drop'))}% | "
            f"{(r.get('gate') or {}).get('status')} | `{r.get('checkpoint')}` |"
        )
    best = (phase_a.get("best_lambda") or {}).get("selected") or {}
    lines += [
        "",
        f"Selected best λ: `{best.get('lambda_slip')}` using non-hard-fail -> lowest Force RMSE -> higher SF1/SA.",
        "",
        "## Phase B best-λ future ablation",
        "",
        "| condition | H1 F1 | H1 AUPRC | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in phase_b.get("rows", []):
        lines.append(f"| {r['condition']} | {fmt(r.get('H1_F1'))} | {fmt(r.get('H1_AUPRC'))} | {fmt(r.get('H3_F1'))} | {fmt(r.get('H3_AUPRC'))} | {fmt(r.get('H5_F1'))} | {fmt(r.get('H5_AUPRC'))} |")
    bc = phase_b.get("best_condition") or {}
    lines += [
        "",
        f"Best future condition by mean AUPRC: `{bc.get('condition', 'n/a')}`.",
        "",
        "## Paper-use recommendation",
        "",
        "Replace the current λ=1.0 main result only if the best-λ row is not a hard fail in Phase A and improves or matches the old future-ablation metrics. If the improvement is mixed, keep λ sweep and best-λ future prediction as ablation evidence rather than changing Table2 main claims.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    stamp = args.stamp
    phase_a_json = Path(args.phase_a_json) if args.phase_a_json else PHASE_A_ROOT / stamp / "phase_lambda_decoupled_report.json"
    if not phase_a_json.exists():
        raise FileNotFoundError(phase_a_json)
    phase_a = read_json(phase_a_json)
    selected = (phase_a.get("best_lambda") or {}).get("selected") or {}
    checkpoint = Path(selected.get("checkpoint", ""))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    best_lambda = selected.get("lambda_slip")
    report_dir = REPORT_ROOT / stamp
    report_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(int(h) for h in args.horizons)
    feature_run_id = args.feature_run_id or f"phase_lambda_best_future_features_{stamp}"
    commands = []

    p4.ensure_decoupled_features(feature_run_id, checkpoint, horizons, args.precompute_batch_size, args.num_workers)
    conditions = []
    for mode in ["z_only", "z_p_slip", "z_force", "z_force_slip", "full"]:
        exp = f"phase_lambda_best_{mode}_{stamp}"
        os.environ["WANDB_NAME"] = exp
        cmd = (
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','0')} WANDB_NAME={exp} WANDB_MODE={args.wandb_mode} "
            f"python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head "
            f"--feature-kind decoupled --feature-run-id {feature_run_id} --checkpoint {checkpoint} "
            f"--experiment-name {exp} --input-mode {mode} --report-dir {report_dir} "
            f"--horizons {' '.join(map(str, horizons))} --precompute-batch-size {args.precompute_batch_size} "
            f"--train-batch-size {args.train_batch_size} --num-workers {args.num_workers} --max-epochs {args.max_epochs} "
            f"--lr {args.lr} --hidden-dim {args.hidden_dim} --dropout {args.dropout} --seed {args.seed} --wandb-mode {args.wandb_mode}"
        )
        commands.append(cmd)
        exp_json = report_dir / f"{exp}_report.json"
        if exp_json.exists():
            s = read_json(exp_json)
        else:
            s = p4.train_future_head(feature_run_id, exp, mode, horizons, args.max_epochs, args.train_batch_size, args.lr, args.hidden_dim, args.dropout, args.seed, args.wandb_mode, report_dir)
        s["condition"] = mode
        s["wandb_name"] = exp
        s["wandb_url"] = f"{WANDB_BASE}/{exp}"
        write_json(Path(s["json_path"]), s)
        conditions.append(s)

    q_summary = None
    if args.include_q:
        train = p5.load_feature(p4.PHASE4_RUN_ROOT, feature_run_id, "train")
        thresholds = p5.threshold_from_train(train, "pred")
        q_feature_run_id = args.q_feature_run_id or f"phase_lambda_best_friction_features_{stamp}"
        if not p5.manifest_path(p5.PHASE5_RUN_ROOT, q_feature_run_id).exists():
            p5.save_augmented_features(feature_run_id, q_feature_run_id, thresholds)
        train_aug = p5.load_feature(p5.PHASE5_RUN_ROOT, q_feature_run_id, "train")
        val_aug = p5.load_feature(p5.PHASE5_RUN_ROOT, q_feature_run_id, "val")
        exp = f"phase_lambda_best_full_plus_q_{stamp}"
        os.environ["WANDB_NAME"] = exp
        q_json = report_dir / f"{exp}_report.json"
        if q_json.exists():
            q_summary = read_json(q_json)
        else:
            q_summary = p5.train_future_head(
                p5.PHASE5_RUN_ROOT,
                q_feature_run_id,
                exp,
                p5.FRICTION_MODES["full_plus_q"],
                p5.all_indices(train_aug),
                p5.all_indices(val_aug),
                horizons,
                args.max_epochs,
                args.train_batch_size,
                args.lr,
                args.hidden_dim,
                args.dropout,
                args.seed,
                args.wandb_mode,
                report_dir,
                thresholds,
                "full_plus_q",
            )
        q_wandb_name = f"{exp}_full_plus_q"
        q_summary["wandb_name"] = q_wandb_name
        q_summary["wandb_url"] = f"{WANDB_BASE}/{q_wandb_name}"
        q_summary["json_path"] = str(q_json)
        q_summary["md_path"] = str(report_dir / f"{exp}_report.md")
        write_json(Path(q_summary["json_path"]), q_summary)
        conditions.append(q_summary)
        commands.append(
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','0')} WANDB_NAME={exp} WANDB_MODE={args.wandb_mode} "
            f"python sparsh-force-slip/scripts/phase_lambda_future_ablation.py --stamp {stamp} --feature-run-id {feature_run_id} "
            f"--q-feature-run-id {q_feature_run_id} --horizons {' '.join(map(str, horizons))} --max-epochs {args.max_epochs} --wandb-mode {args.wandb_mode} # internal full_plus_q head"
        )

    rows = [condition_row(c) for c in conditions]
    old_lambda1 = load_old_rows()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "phase": "phase_lambda_decoupled_future_ablation",
        "stamp": stamp,
        "best_lambda": best_lambda,
        "stage_i_checkpoint": str(checkpoint),
        "feature_run_id": feature_run_id,
        "q_feature_run_id": (q_summary or {}).get("feature_run_id"),
        "conditions": conditions,
        "rows": rows,
        "best_condition": best_condition(rows),
        "old_lambda1": old_lambda1,
        "comparison_rows": build_comparison_rows(rows, old_lambda1),
        "commands": commands,
        "raw_data_modified": False,
    }
    json_path = report_dir / "phase_lambda_decoupled_future_ablation_report.json"
    md_path = report_dir / "phase_lambda_decoupled_future_ablation_report.md"
    payload["json_path"] = str(json_path)
    payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_report(payload), encoding="utf-8")
    FINAL_SUMMARY.write_text(render_final_summary(phase_a, payload), encoding="utf-8")
    print(json.dumps({"report": str(md_path), "summary": str(FINAL_SUMMARY), "best_condition": (payload["best_condition"] or {}).get("condition")}, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stamp", required=True)
    p.add_argument("--phase-a-json", default=None)
    p.add_argument("--feature-run-id", default=None)
    p.add_argument("--q-feature-run-id", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--precompute-batch-size", type=int, default=128)
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "online"))
    p.add_argument("--include-q", action="store_true", default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    if branch != "sparsh-force-slip":
        raise SystemExit(f"Expected branch sparsh-force-slip, got {branch}")
    run(args)


if __name__ == "__main__":
    main()
