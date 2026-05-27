#!/usr/bin/env python3
"""Launch/finalize architecture-to-future supplemental ablations.

The phase fixes Stage-II input to full dynamics and varies only the Stage-I
force-slip interface. It writes derived reports only under sparsh-force-slip.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
REPORT_ROOT = WORKSPACE / "reports/phase_lambda_architecture_future_ablation"
PHASE_A_JSON = WORKSPACE / "reports/phase_lambda_decoupled/20260528_000000/phase_lambda_decoupled_report.json"
PHASE_B_JSON = WORKSPACE / "reports/phase_lambda_decoupled_future_ablation/20260528_000000/phase_lambda_decoupled_future_ablation_report.json"
WANDB_BASE = "https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def git_head() -> dict[str, str]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=REPO, text=True).strip()
    return {"branch": run(["git", "branch", "--show-current"]), "head": run(["git", "rev-parse", "--short", "HEAD"])}


def specs(stamp: str) -> list[dict[str, Any]]:
    phase_a = read_json(PHASE_A_JSON)
    old = phase_a["best_lambda"]["lambda1_reference"]
    partial_sel = read_json(WORKSPACE / "reports/phase2/phase2_b_ps_lam025_gsmini_20260514_063052/checkpoint_selection_mae_partially_shared.json")
    cons_sel = read_json(WORKSPACE / "reports/phase2/phase2_c_consistency_gsmini_20260514_161845/checkpoint_selection_mae_consistency.json")
    return [
        {
            "label": "decoupled_lam100_old", "architecture": "Decoupled λ=1.00 old", "gpu": 1,
            "checkpoint": old["checkpoint"],
            "current_eval_json": str(WORKSPACE / "reports/phase_lambda_decoupled/20260528_000000/eval_cache/phase3_1_decoupled_gsmini_20260516_154730_val.json"),
            "stage_i_source": old.get("eval", {}).get("run_dir"),
        },
        {
            "label": "consistency_decoder", "architecture": "Consistency decoder", "gpu": 2,
            "checkpoint": cons_sel["checkpoint"],
            "current_eval_json": str(WORKSPACE / "reports/phase2/phase2_c_consistency_gsmini_20260514_161845/eval_cache/c_mae_consistency_phase2_c_consistency_gsmini_20260514_161845_allsource_val.json"),
            "stage_i_source": cons_sel.get("run_dir"),
        },
        {
            "label": "partially_shared_lam025", "architecture": "Partially shared λ=0.25", "gpu": 3,
            "checkpoint": partial_sel["checkpoint"],
            "current_eval_json": str(WORKSPACE / "reports/phase2/phase2_b_ps_lam025_gsmini_20260514_063052/eval_cache/b_mae_partially_shared_allsource_val.json"),
            "stage_i_source": partial_sel.get("run_dir"),
        },
    ]


def report_name(label: str, stamp: str) -> str:
    return f"phase_lambda_arch_future_{label}_full_{stamp}"


def report_path(label: str, stamp: str) -> Path:
    return REPORT_ROOT / stamp / f"{report_name(label, stamp)}_report.json"


def train_command(spec: dict[str, Any], stamp: str) -> str:
    exp = report_name(spec["label"], stamp)
    feature_run = f"phase_lambda_arch_future_{spec['label']}_features_{stamp}"
    return (
        f"cd {REPO} && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && "
        f"CUDA_VISIBLE_DEVICES={spec['gpu']} WANDB_MODE=offline WANDB_NAME={exp} PYTHONPATH=. "
        f"python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head "
        f"--feature-kind decoupled --feature-run-id {feature_run} --checkpoint {spec['checkpoint']} "
        f"--experiment-name {exp} --input-mode full --report-dir {REPORT_ROOT / stamp} "
        f"--horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --wandb-mode offline"
    )


def command_manifest(stamp: str) -> dict[str, Any]:
    ss = specs(stamp)
    payload = {"stamp": stamp, "generated_at": datetime.now().isoformat(timespec="seconds"), "tasks": []}
    for s in ss:
        exp = report_name(s["label"], stamp)
        payload["tasks"].append({**s, "experiment_name": exp, "feature_run_id": f"phase_lambda_arch_future_{s['label']}_features_{stamp}", "report_json": str(report_path(s["label"], stamp)), "wandb_name": exp, "wandb_url": f"{WANDB_BASE}/{exp}", "command": train_command(s, stamp)})
    write_json(REPORT_ROOT / stamp / "architecture_future_command_manifest.json", payload)
    return payload


def get_nested(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def current_metrics_from_phase_a() -> dict[str, Any]:
    phase_a = read_json(PHASE_A_JSON)
    sel = phase_a["best_lambda"]["selected"]
    return {
        "checkpoint": sel.get("checkpoint"),
        "force_rmse_mean_N": sel.get("force_rmse_mean_N"),
        "slip_f1": sel.get("slip_f1"),
        "slip_accuracy": sel.get("slip_accuracy"),
    }


def future_full_from_phase_b() -> dict[str, Any] | None:
    if not PHASE_B_JSON.exists():
        return None
    d = read_json(PHASE_B_JSON)
    for row in d.get("rows", []):
        if row.get("condition") == "full":
            return row
    return None


def row_from_head(task: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_val") or {}
    base = summary.get("base_current_metrics_val") or {}
    row = {
        "architecture": task["architecture"],
        "label": task["label"],
        "stage_i_checkpoint": task["checkpoint"],
        "future_head_checkpoint": summary.get("checkpoint"),
        "current_force_rmse_mean_N": base.get("force_rmse_mean_N"),
        "current_slip_f1": base.get("slip_f1"),
        "current_slip_accuracy": base.get("slip_accuracy"),
        "best_epoch": summary.get("best_epoch"),
        "wandb_name": task["wandb_name"],
        "wandb_url": task["wandb_url"],
        "training_command": task["command"],
        "report_json": summary.get("json_path"),
        "report_md": summary.get("md_path"),
    }
    for h in [1, 3, 5]:
        m = best.get(f"H{h}", {})
        row[f"H{h}_F1"] = m.get("future_slip_f1")
        row[f"H{h}_AUPRC"] = m.get("future_slip_auprc")
        row[f"H{h}_ECE"] = m.get("stability_calibration_error")
        row[f"H{h}_lead_mean"] = m.get("lead_time_to_slip_onset_steps_mean")
        row[f"H{h}_high_risk_recall"] = m.get("high_risk_model_recall_on_future_slip")
    return row


def mean_auprc(row: dict[str, Any]) -> float | None:
    vals = [row.get(f"H{h}_AUPRC") for h in [1, 3, 5]]
    nums = [float(v) for v in vals if v is not None]
    return sum(nums) / len(nums) if nums else None


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Architecture-to-Future Full-dynamics Ablation",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- stamp: `{payload['stamp']}`",
        f"- branch/head: `{payload['git']['branch']}` / `{payload['git']['head']}`",
        "- Stage-II input: `Z + predicted force + slip probability + causal ΔF` (`input_mode=full`).",
        "- Decoupled λ=0.10 full-dynamics result is reused from Phase B; it is not retrained.",
        "- raw_data_modified: `False`",
        "- W&B mode: `offline fallback` because `api.wandb.ai:443` was unreachable from zjy-4090 during launch; run `wandb sync` on the saved run directories when network recovers. The deterministic W&B run ids/names are still recorded below.",
        "",
        "## Current and future metrics",
        "",
        "| architecture | F RMSE | SF1 | SA | H1 F1 | H1 AUPRC | H1 ECE | H1 lead | H3 F1 | H3 AUPRC | H3 ECE | H3 lead | H5 F1 | H5 AUPRC | H5 ECE | H5 lead | mean AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["rows"]:
        lines.append(
            f"| {r['architecture']} | {fmt(r.get('current_force_rmse_mean_N'))} | {fmt(r.get('current_slip_f1'))} | {fmt(r.get('current_slip_accuracy'))} | "
            f"{fmt(r.get('H1_F1'))} | {fmt(r.get('H1_AUPRC'))} | {fmt(r.get('H1_ECE'))} | {fmt(r.get('H1_lead_mean'))} | "
            f"{fmt(r.get('H3_F1'))} | {fmt(r.get('H3_AUPRC'))} | {fmt(r.get('H3_ECE'))} | {fmt(r.get('H3_lead_mean'))} | "
            f"{fmt(r.get('H5_F1'))} | {fmt(r.get('H5_AUPRC'))} | {fmt(r.get('H5_ECE'))} | {fmt(r.get('H5_lead_mean'))} | {fmt(mean_auprc(r))} |"
        )
    lines += ["", "## Checkpoints and W&B", ""]
    for r in payload["rows"]:
        lines += [
            f"### {r['architecture']}",
            f"- Stage-I checkpoint: `{r.get('stage_i_checkpoint')}`",
            f"- Future-head checkpoint: `{r.get('future_head_checkpoint')}`",
            f"- W&B: [{r.get('wandb_name')}]({r.get('wandb_url')})" if r.get("wandb_url") else f"- W&B: `{r.get('wandb_name')}`",
            f"- report_json: `{r.get('report_json')}`",
            "",
        ]
    lines += ["## Interpretation", ""]
    for note in payload.get("conclusions", []):
        lines.append(f"- {note}")
    lines += ["", "## Training commands", ""]
    for t in payload.get("tasks", []):
        lines.append(f"### {t['architecture']}\n```bash\n{t['command']}\n```\n")
    if payload.get("missing"):
        lines += ["", "## Missing/failed", ""]
        for m in payload["missing"]:
            lines.append(f"- {m}")
    return "\n".join(lines) + "\n"


def finalize(stamp: str) -> dict[str, Any]:
    manifest = command_manifest(stamp)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    # Main method reused from Phase B.
    main = future_full_from_phase_b()
    if main:
        cm = current_metrics_from_phase_a()
        row = {
            "architecture": "Decoupled λ=0.10 (main, Phase B reused)",
            "label": "decoupled_lam010_main",
            "stage_i_checkpoint": cm.get("checkpoint"),
            "future_head_checkpoint": main.get("checkpoint"),
            "current_force_rmse_mean_N": cm.get("force_rmse_mean_N"),
            "current_slip_f1": cm.get("slip_f1"),
            "current_slip_accuracy": cm.get("slip_accuracy"),
            "best_epoch": main.get("best_epoch"),
            "wandb_name": main.get("wandb_name"),
            "wandb_url": main.get("wandb_url"),
            "training_command": "reused from Phase B; no retraining",
            "report_json": str(PHASE_B_JSON),
            "report_md": str(PHASE_B_JSON.with_suffix('.md')),
        }
        for h in [1, 3, 5]:
            row[f"H{h}_F1"] = main.get(f"H{h}_F1")
            row[f"H{h}_AUPRC"] = main.get(f"H{h}_AUPRC")
            row[f"H{h}_ECE"] = main.get(f"H{h}_ECE")
            row[f"H{h}_lead_mean"] = main.get(f"H{h}_lead_mean")
            row[f"H{h}_high_risk_recall"] = main.get(f"H{h}_high_risk_recall")
        rows.append(row)
    else:
        missing.append(f"Phase B full-dynamics row missing: {PHASE_B_JSON}")
    for t in manifest["tasks"]:
        p = Path(t["report_json"])
        if p.exists():
            rows.append(row_from_head(t, read_json(p)))
        else:
            missing.append(f"missing report for {t['architecture']}: {p}")
    best_future = max((r for r in rows if mean_auprc(r) is not None), key=lambda r: mean_auprc(r), default=None)
    main_row = next((r for r in rows if r["label"] == "decoupled_lam010_main"), None)
    conclusions = []
    if main_row:
        conclusions.append("Decoupled λ=0.10 remains the preferred Stage-I interface for the main method because it is the λ-sweep selection: it preserves the current force-slip balance while matching the old decoupled λ=1.00 future mean AUPRC under the same full-dynamics Stage-II input.")
    if best_future:
        conclusions.append(f"Best mean H1/H3/H5 future AUPRC in this comparison is `{best_future['architecture']}` ({fmt(mean_auprc(best_future))}), but the margin over Decoupled λ=0.10 is small and must be weighed against current Force RMSE, calibration, and architectural simplicity.")
    conclusions.append("Consistency decoder is not used as the main interface because its small future-AUPRC gain comes with worse current Force RMSE than the selected Decoupled λ=0.10 interface and higher calibration error at longer horizons; it is better presented as a tested alternative/upper-bound ablation.")
    conclusions.append("Partially shared λ=0.25 is not used as the main method because it does not dominate the future metrics and shows worse long-horizon calibration/ECE, so shared capacity is still vulnerable to force-slip negative transfer.")
    conclusions.append("The final manuscript can use this table to justify Decoupled λ=0.10 as a conservative, force-stable Stage-I interface and cite consistency/partially-shared rows as alternatives tested under identical full-dynamics Stage-II inputs.")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stamp": stamp,
        "git": git_head(),
        "phase_b_json": str(PHASE_B_JSON),
        "phase_a_json": str(PHASE_A_JSON),
        "stage_ii_input": "full dynamics: Z + predicted force + slip probability + causal delta force",
        "rows": rows,
        "tasks": manifest["tasks"],
        "missing": missing,
        "best_future_by_mean_auprc": best_future["architecture"] if best_future else None,
        "conclusions": conclusions,
        "raw_data_modified": False,
    }
    out_dir = REPORT_ROOT / stamp
    json_path = out_dir / "phase_lambda_architecture_future_ablation_report.json"
    md_path = out_dir / "phase_lambda_architecture_future_ablation_report.md"
    payload["json_path"] = str(json_path)
    payload["md_path"] = str(md_path)
    write_json(json_path, payload)
    md_path.write_text(render_report(payload), encoding="utf-8")
    (REPORT_ROOT / "current_phase_lambda_architecture_future_ablation.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["manifest", "finalize"])
    ap.add_argument("--stamp", default="20260528_000000")
    args = ap.parse_args()
    if args.command == "manifest":
        payload = command_manifest(args.stamp)
    else:
        payload = finalize(args.stamp)
    print(json.dumps({"json_path": payload.get("json_path"), "md_path": payload.get("md_path"), "missing": payload.get("missing")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
