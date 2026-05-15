#!/usr/bin/env python3
"""Finalize Phase3-0 sanity + DINO ABC reports.

This script is intentionally read/evaluate oriented. It does not modify original
raw data; it only reads Phase1 derived data, Phase2/Phase3 training artifacts, and
writes reports/runbooks under sparsh-force-slip.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import phase2_b_multitask as p2  # noqa: E402

REPO = Path("/home/zjy/document/tactile-grasp")
WORKSPACE = REPO / "sparsh-force-slip"
PHASE3_REPORT_ROOT = WORKSPACE / "reports/phase3"
PHASE3_LOG_ROOT = WORKSPACE / "logs/phase3"
PHASE3_RUNBOOK_ROOT = WORKSPACE / "runbooks"
PHASE2_REPORT_ROOT = WORKSPACE / "reports/phase2"
ENCODERS = ["mae", "dino", "dinov2", "ijepa", "vjepa"]
FORCEONLY_VARIANT = "partially_shared"
DINO_B_VARIANT = "partially_shared"
DINO_C_VARIANT = "consistency"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=p2.json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(stamp: str) -> dict[str, Any]:
    path = PHASE3_REPORT_ROOT / f"phase3_0_{stamp}" / "launch_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def run_dir(run_id: str, encoder: str, variant: str) -> Path:
    return p2.PHASE2_ROOT / run_id / p2.decoder_run_suffix(encoder, variant)


def training_summary_path(run_id: str, encoder: str, variant: str) -> Path:
    return run_dir(run_id, encoder, variant) / "training_summary.json"


def select_forceonly_checkpoint(run_id: str, encoder: str) -> dict[str, Any]:
    root = run_dir(run_id, encoder, FORCEONLY_VARIANT)
    history_path = root / "history.json"
    summary_path = root / "training_summary.json"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    history = read_json(history_path)
    candidates: list[dict[str, Any]] = []
    for record in history:
        val = record.get("val")
        if not val:
            continue
        epoch = int(record["epoch"])
        ckpt = root / "checkpoints" / f"epoch-{epoch:04d}.pth"
        if not ckpt.exists():
            continue
        candidates.append(
            {
                "epoch": epoch,
                "global_step": record.get("global_step"),
                "checkpoint": str(ckpt),
                "force_rmse_mean_N": val.get("force_rmse_mean_N"),
                "force_mae_mean_N": val.get("force_mae_mean_N"),
                "slip_f1": val.get("slip_f1"),
                "slip_accuracy": val.get("slip_accuracy"),
                "val": val,
            }
        )
    if not candidates:
        raise RuntimeError(f"No validation candidates found for {encoder} in {history_path}")
    selected = sorted(candidates, key=lambda c: (float(c["force_rmse_mean_N"]), -float(c.get("slip_f1") or 0.0)))[0]
    summary = read_json(summary_path) if summary_path.exists() else {}
    selected.update(
        {
            "selection_policy": "lowest validation force_rmse_mean_N among saved validation checkpoints",
            "run_dir": str(root),
            "training_summary": str(summary_path) if summary_path.exists() else None,
            "wandb_url": summary.get("wandb_url"),
            "wandb_name": summary.get("wandb_name"),
            "candidate_count": len(candidates),
        }
    )
    return selected


def forceonly_gate(a_eval: dict[str, Any], forceonly: dict[str, Any]) -> dict[str, Any]:
    a_force = float(a_eval["aggregate"]["force_rmse_mean_N"])
    s_force = float(forceonly["force_rmse_mean_N"])
    inc = (s_force / a_force - 1.0) * 100.0 if a_force else float("inf")
    if inc > 10.0:
        status = "hard_fail"
    elif inc >= 5.0:
        status = "warning_band"
    else:
        status = "pass"
    return {
        "status": status,
        "A_force_rmse_mean_N": a_force,
        "forceonly_rmse_mean_N": s_force,
        "force_rmse_increase_pct_vs_A": inc,
        "thresholds": {"pass": "<5% increase", "warning_band": "5-10% increase", "hard_fail": ">10% increase"},
    }


def load_or_eval_a(encoder: str, reference: dict[str, Any], cache_dir: Path, batch_size: int, num_workers: int, refresh: bool) -> dict[str, Any]:
    cache = cache_dir / f"a_{encoder}_allsource_val.json"
    if cache.exists() and not refresh:
        return read_json(cache)
    out = p2.evaluate_a_models(encoder, reference, batch_size=batch_size, num_workers=num_workers)
    write_json(cache, out)
    return out


def eval_b_candidate(run_id: str, encoder: str, a_eval: dict[str, Any], reference: dict[str, Any], cache_dir: Path, batch_size: int, num_workers: int, refresh: bool) -> dict[str, Any]:
    cache = cache_dir / f"b_{encoder}_{run_id}_{DINO_B_VARIANT}_allsource_val.json"
    if cache.exists() and not refresh:
        return read_json(cache)
    selection = p2.select_b_checkpoint_for_gate(run_id, encoder, DINO_B_VARIANT, a_eval)
    out = p2.evaluate_b_checkpoint_for_report(
        run_id,
        encoder,
        reference,
        batch_size=batch_size,
        num_workers=num_workers,
        decoder_variant=DINO_B_VARIANT,
        checkpoint_path=selection.get("checkpoint"),
    )
    out["checkpoint_selection"] = selection
    out["gate_vs_A"] = p2.gate_status(a_eval, out)
    write_json(cache, out)
    return out


def choose_best_b(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for lam, eval_payload in candidates.items():
        gate = eval_payload["gate_vs_A"]
        ag = eval_payload["aggregate"]
        rows.append(
            {
                "lambda_slip": lam,
                "run_id": eval_payload.get("run_id"),
                "gate_status": gate["status"],
                "force_rmse_mean_N": ag["force_rmse_mean_N"],
                "slip_f1": ag["slip_f1"],
                "force_rmse_increase_pct_vs_A": gate["force_rmse_increase_pct"],
                "slip_f1_drop_pp_vs_A": gate["slip_f1_drop_pp"],
                "checkpoint": eval_payload.get("checkpoint"),
                "wandb_url": (eval_payload.get("checkpoint_selection") or {}).get("wandb_url"),
                "payload": eval_payload,
            }
        )
    feasible = [r for r in rows if r["gate_status"] != "hard_fail"]
    if feasible:
        selected = sorted(feasible, key=lambda r: (float(r["slip_f1"]), -float(r["force_rmse_increase_pct_vs_A"])), reverse=True)[0]
        policy = "non_hard_fail_then_best_slip_f1"
    else:
        selected = sorted(rows, key=lambda r: (float(r["force_rmse_increase_pct_vs_A"]), -float(r["slip_f1"]))) [0]
        policy = "all_hard_fail_choose_lowest_force_regression_damage"
    return {"policy": policy, "selected": {k: v for k, v in selected.items() if k != "payload"}, "rows": [{k: v for k, v in r.items() if k != "payload"} for r in rows]}


def prior_abc_rows() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    c_report = PHASE2_REPORT_ROOT / "phase2_c_consistency_gsmini_20260514_161845/phase2_c_consistency_report.json"
    if c_report.exists():
        d = read_json(c_report)
        for enc in d.get("encoders", []):
            rows[enc] = {
                "source": str(c_report),
                "A_force_rmse_mean_N": d["a_evaluations"][enc]["aggregate"].get("force_rmse_mean_N"),
                "A_slip_f1": d["a_evaluations"][enc]["aggregate"].get("slip_f1"),
                "B_force_rmse_mean_N": d["b_evaluations"][enc]["aggregate"].get("force_rmse_mean_N"),
                "B_slip_f1": d["b_evaluations"][enc]["aggregate"].get("slip_f1"),
                "B_gate_status": d["gate_b_vs_a"][enc].get("status"),
                "C_force_rmse_mean_N": d["c_evaluations"][enc]["aggregate"].get("force_rmse_mean_N"),
                "C_slip_f1": d["c_evaluations"][enc]["aggregate"].get("slip_f1"),
                "C_gate_vs_A": d["gate_c_vs_a"][enc].get("status"),
                "C_gate_vs_B": d["gate_c_vs_b"][enc].get("status"),
                "C_training_launched": True,
            }
    jepa_summary = PHASE2_REPORT_ROOT / "phase2_jepa_abc_summary_20260515.json"
    if jepa_summary.exists():
        d = read_json(jepa_summary)
        for row in d.get("rows", []):
            enc = row["encoder"]
            b = row.get("B_selected") or {}
            rows[enc] = {
                "source": str(jepa_summary),
                "A_force_rmse_mean_N": row.get("A_force_rmse_mean_N"),
                "A_slip_f1": row.get("A_slip_f1"),
                "B_force_rmse_mean_N": b.get("force_rmse_B"),
                "B_slip_f1": b.get("slip_f1_B"),
                "B_gate_status": b.get("gate_status"),
                "C_force_rmse_mean_N": row.get("C_force_rmse_mean_N"),
                "C_slip_f1": row.get("C_slip_f1"),
                "C_gate_vs_A": row.get("C_gate_vs_A"),
                "C_gate_vs_B": row.get("C_gate_vs_B"),
                "C_training_launched": bool(row.get("C_training_launched")),
            }
    return rows


def missing_prereqs(manifest: dict[str, Any]) -> list[str]:
    missing = []
    force_run = manifest["run_id_forceonly"]
    for enc in ENCODERS:
        sp = training_summary_path(force_run, enc, FORCEONLY_VARIANT)
        if not sp.exists():
            missing.append(f"force-only summary missing: {enc} {sp}")
    for lam, run_id in manifest["dino_b_run_ids"].items():
        sp = training_summary_path(run_id, "dino", DINO_B_VARIANT)
        if not sp.exists():
            missing.append(f"DINO B lambda={lam} summary missing: {sp}")
    for spec_name, spec in [("DINO A force", p2.A_FORCE_EXPS["dino"]), ("DINO A slip", p2.A_SLIP_EXPS["dino"] )]:
        try:
            exp = p2.resolve_experiment_dir(spec)
            ckpt = exp / "checkpoints/epoch-0051.pth"
            if not ckpt.exists():
                missing.append(f"{spec_name} checkpoint missing: {ckpt}")
        except Exception as exc:
            missing.append(f"{spec_name} experiment missing: {exc}")
    return missing


def launch_c_if_needed(stamp: str, selected_b: dict[str, Any], gpu: int) -> dict[str, Any]:
    c_run_id = f"phase3_0_dino_c_consistency_gsmini_{stamp}"
    summary = training_summary_path(c_run_id, "dino", DINO_C_VARIANT)
    if summary.exists():
        return {"status": "already_completed", "run_id": c_run_id, "summary": str(summary)}
    runbook_dir = PHASE3_RUNBOOK_ROOT / f"phase3_0_{stamp}"
    log_dir = PHASE3_LOG_ROOT / f"phase3_0_{stamp}"
    runbook_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    lam = selected_b["lambda_slip"]
    script = runbook_dir / f"run_dino_c_consistency_gpu{gpu}.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/zjy/document/tactile-grasp
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES={gpu}
export WANDB_MODE=online
export WANDB_NAME=\"{c_run_id}_dino_c_consistency_lam{lam}_beta0.05\"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \\
  --encoder dino \\
  --run-id \"{c_run_id}\" \\
  --decoder-variant consistency \\
  --lambda-slip {lam} \\
  --beta-consistency 0.05 \\
  --consistency-alpha 10 \\
  --consistency-tau-source p65 \\
  --max-epochs 51 \\
  --batch-size 100 \\
  --num-workers 2 \\
  --validation-frequency 5 \\
  --wandb-mode online \\
  +trainer.devices=1
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    session = f"p30_{stamp}_dino_c_g{gpu}"
    if subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return {"status": "session_exists", "session": session, "run_id": c_run_id, "runbook": str(script)}
    log = log_dir / f"{session}.log"
    cmd = f"bash '{script}' 2>&1 | tee '{log}'"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd], check=True)
    return {"status": "launched", "session": session, "run_id": c_run_id, "runbook": str(script), "log": str(log)}


def render_report(report: dict[str, Any]) -> str:
    def fmt(v: Any, digits: int = 4) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, str):
            return v
        return f"{float(v):.{digits}f}"

    lines = [
        "# Phase3-0 Sanity + DINO ABC Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- stamp: `{report['stamp']}`",
        f"- derived_dataset: `{report['derived_dataset']}`",
        f"- manual_push_pull: `{str(report['manual_push_pull']).lower()}`",
        "",
        "## Force-only multitask sanity",
        "",
        "| encoder | A force RMSE N | force-only RMSE N | Δ vs A | gate | checkpoint |",
        "|---|---:|---:|---:|---|---|",
    ]
    for enc in ENCODERS:
        row = report["forceonly_sanity"][enc]
        gate = row["gate"]
        lines.append(
            f"| {enc} | {fmt(gate['A_force_rmse_mean_N'])} | {fmt(gate['forceonly_rmse_mean_N'])} | "
            f"{fmt(gate['force_rmse_increase_pct_vs_A'], 2)}% | {gate['status']} | `{Path(row['checkpoint']).name}` |"
        )
    lines += [
        "",
        "## DINO ABC",
        "",
        "| stage | force RMSE N | slip F1 | gate/status | run/checkpoint |",
        "|---|---:|---:|---|---|",
    ]
    dino = report["dino_abc"]
    lines.append(f"| A | {fmt(dino['A']['force_rmse_mean_N'])} | {fmt(dino['A']['slip_f1'])} | baseline | `{dino['A']['force_experiment']}` |")
    for row in dino["B_candidates"]:
        lines.append(f"| B λ={row['lambda_slip']} | {fmt(row['force_rmse_mean_N'])} | {fmt(row['slip_f1'])} | {row['gate_status']} | `{Path(row['checkpoint']).name}` |")
    c = dino["C"]
    lines.append(f"| C | {fmt(c.get('force_rmse_mean_N'))} | {fmt(c.get('slip_f1'))} | {c.get('status')} | `{c.get('run_id')}` |")
    lines += [
        "",
        "## A/B/C summary across backbones",
        "",
        "| encoder | A force | A slip F1 | B force | B slip F1 | B gate | C force | C slip F1 | C status |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for enc in ENCODERS:
        row = report["abc_rows"].get(enc, {})
        lines.append(
            f"| {enc} | {fmt(row.get('A_force_rmse_mean_N'))} | {fmt(row.get('A_slip_f1'))} | "
            f"{fmt(row.get('B_force_rmse_mean_N'))} | {fmt(row.get('B_slip_f1'))} | {row.get('B_gate_status', 'n/a')} | "
            f"{fmt(row.get('C_force_rmse_mean_N'))} | {fmt(row.get('C_slip_f1'))} | {row.get('C_status', row.get('C_gate_vs_A', 'n/a'))} |"
        )
    lines += ["", "## Gate conclusion", "", report["conclusion"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--refresh-eval", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--launch-c-if-needed", action="store_true")
    parser.add_argument("--c-gpu", type=int, default=0)
    args = parser.parse_args()

    if args.stamp is None:
        stamp_path = PHASE3_REPORT_ROOT / "current_phase3_0_stamp.txt"
        args.stamp = stamp_path.read_text(encoding="utf-8").strip()
    manifest = load_manifest(args.stamp)
    report_dir = PHASE3_REPORT_ROOT / f"phase3_0_{args.stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_prereqs(manifest)
    if missing and not args.allow_partial:
        status = {"generated_at": datetime.now().isoformat(timespec="seconds"), "stamp": args.stamp, "status": "pending", "missing": missing}
        write_json(report_dir / "phase3_0_pending_status.json", status)
        print(json.dumps(status, indent=2))
        return 2

    reference_path = report_dir / "phase3_0_force_ratio_reference.json"
    if reference_path.exists() and not args.refresh_eval:
        reference = read_json(reference_path)
    else:
        reference = p2.collect_force_ratio_reference(slip_horizon=0)
        write_json(reference_path, reference)
    cache_dir = report_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    a_evals = {enc: load_or_eval_a(enc, reference, cache_dir, args.batch_size, args.num_workers, args.refresh_eval) for enc in ENCODERS}
    force_run = manifest["run_id_forceonly"]
    sanity = {}
    for enc in ENCODERS:
        sel = select_forceonly_checkpoint(force_run, enc)
        gate = forceonly_gate(a_evals[enc], sel)
        sanity[enc] = {**sel, "gate": gate}

    dino_b_evals = {}
    for lam, run_id in manifest["dino_b_run_ids"].items():
        ev = eval_b_candidate(run_id, "dino", a_evals["dino"], reference, cache_dir, args.batch_size, args.num_workers, args.refresh_eval)
        ev["run_id"] = run_id
        dino_b_evals[lam] = ev
    best_b = choose_best_b(dino_b_evals)
    selected_gate = best_b["selected"]["gate_status"]
    c_info: dict[str, Any]
    if selected_gate == "hard_fail":
        c_info = {"status": "diagnostic_only_skipped_b_hard_fail", "run_id": None, "force_rmse_mean_N": None, "slip_f1": None}
    else:
        c_run_id = f"phase3_0_dino_c_consistency_gsmini_{args.stamp}"
        c_summary = training_summary_path(c_run_id, "dino", DINO_C_VARIANT)
        if c_summary.exists():
            c_eval = p2.evaluate_b_checkpoint_for_report(c_run_id, "dino", reference, args.batch_size, args.num_workers, decoder_variant=DINO_C_VARIANT)
            gate_a = p2.gate_status(a_evals["dino"], c_eval)
            gate_b = p2.gate_status(dino_b_evals[best_b["selected"]["lambda_slip"]], c_eval)
            c_info = {
                "status": "completed",
                "run_id": c_run_id,
                "force_rmse_mean_N": c_eval["aggregate"].get("force_rmse_mean_N"),
                "slip_f1": c_eval["aggregate"].get("slip_f1"),
                "gate_vs_A": gate_a,
                "gate_vs_B": gate_b,
                "checkpoint": c_eval.get("checkpoint"),
                "summary": str(c_summary),
            }
        elif args.launch_c_if_needed:
            launch = launch_c_if_needed(args.stamp, best_b["selected"], args.c_gpu)
            write_json(report_dir / "phase3_0_dino_c_launch.json", launch)
            print(json.dumps(launch, indent=2))
            return 3
        else:
            c_info = {"status": "needed_not_launched", "run_id": c_run_id, "force_rmse_mean_N": None, "slip_f1": None}

    abc_rows = prior_abc_rows()
    abc_rows["dino"] = {
        "source": str(report_dir / "phase3_0_sanity_dino_abc_report.json"),
        "A_force_rmse_mean_N": a_evals["dino"]["aggregate"].get("force_rmse_mean_N"),
        "A_slip_f1": a_evals["dino"]["aggregate"].get("slip_f1"),
        "B_force_rmse_mean_N": best_b["selected"].get("force_rmse_mean_N"),
        "B_slip_f1": best_b["selected"].get("slip_f1"),
        "B_gate_status": best_b["selected"].get("gate_status"),
        "C_force_rmse_mean_N": c_info.get("force_rmse_mean_N"),
        "C_slip_f1": c_info.get("slip_f1"),
        "C_status": c_info.get("status"),
        "C_training_launched": c_info.get("status") == "completed",
    }

    hard_sanity = [enc for enc, row in sanity.items() if row["gate"]["status"] == "hard_fail"]
    if hard_sanity:
        conclusion = "Sanity hard-failed for " + ", ".join(hard_sanity) + "; pause P3-1/P3-2 and diagnose dataloader/normalization/frame-label alignment before structural experiments."
    elif c_info["status"] == "needed_not_launched":
        conclusion = "Force-only sanity has no hard fail, but DINO B passed/warned and DINO C still needs to be launched before final P3-0 completion."
    else:
        conclusion = "P3-0 completed. Use force-only sanity gates plus DINO ABC rows to decide whether P3-1 decoupled multitask may proceed."

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stamp": args.stamp,
        "phase": "P3-0 sanity + DINO ABC",
        "derived_dataset": manifest.get("derived_dataset"),
        "manual_push_pull": False,
        "manifest": manifest,
        "missing_prereqs": missing,
        "A_evaluations": a_evals,
        "forceonly_sanity": sanity,
        "dino_abc": {
            "A": {
                "force_rmse_mean_N": a_evals["dino"]["aggregate"].get("force_rmse_mean_N"),
                "slip_f1": a_evals["dino"]["aggregate"].get("slip_f1"),
                "force_experiment": a_evals["dino"].get("force_experiment"),
                "slip_experiment": a_evals["dino"].get("slip_experiment"),
            },
            "B_candidates": best_b["rows"],
            "B_selection": best_b,
            "C": c_info,
        },
        "abc_rows": abc_rows,
        "conclusion": conclusion,
    }
    json_path = report_dir / "phase3_0_sanity_dino_abc_report.json"
    md_path = report_dir / "phase3_0_sanity_dino_abc_report.md"
    write_json(json_path, report)
    md_path.write_text(render_report(report), encoding="utf-8")
    (PHASE3_REPORT_ROOT / "current_phase3_0_report.md").write_text(f"# Current Phase3-0 Report\n\n- report: `{md_path}`\n- json: `{json_path}`\n- generated_at: `{report['generated_at']}`\n", encoding="utf-8")
    print(json.dumps({"status": "reported", "report": str(md_path), "json": str(json_path), "conclusion": conclusion}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
