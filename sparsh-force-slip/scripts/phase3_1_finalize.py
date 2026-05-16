#!/usr/bin/env python3
"""Finalize Phase3-1 decoupled multitask reports.

Reads completed decoupled multitask checkpoints, evaluates them against the
existing A single-task force/slip baselines on the same Phase1 derived train/val
policy, and writes Phase3-1 JSON/Markdown reports under sparsh-force-slip.
"""
from __future__ import annotations

import argparse
import json
import math
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
ENCODERS = ["mae", "dino", "dinov2", "ijepa", "vjepa"]
DECODER_VARIANT = "decoupled"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=p2.json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def domain_name(dataset_name: str) -> str:
    return dataset_name.split("_", 1)[0]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def weighted_mean(items: list[tuple[float | None, int]]) -> float | None:
    num = 0.0
    den = 0
    for value, weight in items:
        if value is None:
            continue
        num += float(value) * int(weight)
        den += int(weight)
    return num / den if den else None


def domain_summary(eval_payload: dict[str, Any]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for name, metrics in eval_payload.get("per_dataset", {}).items():
        buckets.setdefault(domain_name(name), []).append(metrics)
    out: dict[str, Any] = {}
    for domain, rows in sorted(buckets.items()):
        n = sum(int(row.get("n_samples", 0)) for row in rows)
        out[domain] = {
            "n_samples": n,
            "force_rmse_mean_N": weighted_mean([(row.get("force_rmse_mean_N"), row.get("n_samples", 0)) for row in rows]),
            "slip_f1_weighted_dataset_mean": weighted_mean([(row.get("slip_f1"), row.get("n_samples", 0)) for row in rows]),
            "slip_accuracy": weighted_mean([(row.get("slip_accuracy"), row.get("n_samples", 0)) for row in rows]),
            "contradiction_rate": weighted_mean([((row.get("consistency") or {}).get("contradiction_rate"), row.get("n_samples", 0)) for row in rows]),
            "monotonic_calibration_error": weighted_mean([((row.get("consistency") or {}).get("monotonic_calibration_error"), row.get("n_samples", 0)) for row in rows]),
        }
    return out


def domain_deltas(a_domains: dict[str, Any], b_domains: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for domain in sorted(set(a_domains) | set(b_domains)):
        a = a_domains.get(domain, {})
        b = b_domains.get(domain, {})
        a_force = a.get("force_rmse_mean_N")
        b_force = b.get("force_rmse_mean_N")
        a_f1 = a.get("slip_f1_weighted_dataset_mean")
        b_f1 = b.get("slip_f1_weighted_dataset_mean")
        force_inc = None
        if a_force not in (None, 0) and b_force is not None:
            force_inc = (float(b_force) / float(a_force) - 1.0) * 100.0
        f1_drop = None
        if a_f1 is not None and b_f1 is not None:
            f1_drop = (float(a_f1) - float(b_f1)) * 100.0
        out[domain] = {
            "A": a,
            "decoupled": b,
            "force_rmse_increase_pct_vs_A": force_inc,
            "slip_f1_drop_pp_vs_A": f1_drop,
        }
    return out


def consistency_value(eval_payload: dict[str, Any], key: str) -> float | None:
    return (eval_payload.get("aggregate", {}).get("consistency") or {}).get(key)


def row_from_eval(encoder: str, a_eval: dict[str, Any], b_eval: dict[str, Any]) -> dict[str, Any]:
    gate = p2.gate_status(a_eval, b_eval)
    a_domains = domain_summary(a_eval)
    b_domains = domain_summary(b_eval)
    dd = domain_deltas(a_domains, b_domains)
    domain_force_incs = [v.get("force_rmse_increase_pct_vs_A") for v in dd.values() if v.get("force_rmse_increase_pct_vs_A") is not None]
    domain_f1_drops = [v.get("slip_f1_drop_pp_vs_A") for v in dd.values() if v.get("slip_f1_drop_pp_vs_A") is not None]
    a_con = consistency_value(a_eval, "contradiction_rate")
    b_con = consistency_value(b_eval, "contradiction_rate")
    a_mono = consistency_value(a_eval, "monotonic_calibration_error")
    b_mono = consistency_value(b_eval, "monotonic_calibration_error")
    contradiction_delta = None if a_con is None or b_con is None else float(b_con) - float(a_con)
    monotonic_delta = None if a_mono is None or b_mono is None else float(b_mono) - float(a_mono)
    return {
        "encoder": encoder,
        "checkpoint": b_eval.get("checkpoint"),
        "run_dir": b_eval.get("run_dir"),
        "selection": b_eval.get("checkpoint_selection"),
        "gate_vs_A": gate,
        "A_force_rmse_mean_N": a_eval["aggregate"].get("force_rmse_mean_N"),
        "decoupled_force_rmse_mean_N": b_eval["aggregate"].get("force_rmse_mean_N"),
        "A_slip_f1": a_eval["aggregate"].get("slip_f1"),
        "decoupled_slip_f1": b_eval["aggregate"].get("slip_f1"),
        "decoupled_slip_accuracy": b_eval["aggregate"].get("slip_accuracy"),
        "A_contradiction_rate": a_con,
        "decoupled_contradiction_rate": b_con,
        "contradiction_delta_vs_A": contradiction_delta,
        "A_monotonic_calibration_error": a_mono,
        "decoupled_monotonic_calibration_error": b_mono,
        "monotonic_calibration_delta_vs_A": monotonic_delta,
        "domain_summary": dd,
        "worst_domain_force_rmse_increase_pct_vs_A": max(domain_force_incs) if domain_force_incs else None,
        "worst_domain_slip_f1_drop_pp_vs_A": max(domain_f1_drops) if domain_f1_drops else None,
    }


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    gate = row["gate_vs_A"]
    force_inc = float(gate.get("force_rmse_increase_pct", 1e9))
    slip_drop = float(gate.get("slip_f1_drop_pp", 1e9))
    contradiction_delta = row.get("contradiction_delta_vs_A")
    monotonic_delta = row.get("monotonic_calibration_delta_vs_A")
    worst_force = row.get("worst_domain_force_rmse_increase_pct_vs_A")
    worst_slip = row.get("worst_domain_slip_f1_drop_pp_vs_A")
    force_tier = 0 if force_inc <= 5.0 else 1 if force_inc <= 10.0 else 2
    slip_tier = 0 if slip_drop <= 0.0 else 1 if slip_drop <= 1.0 else 2 if slip_drop <= 2.0 else 3
    consistency_tier = 0 if (contradiction_delta is not None and contradiction_delta <= 0.0 and (monotonic_delta is None or monotonic_delta <= 0.0)) else 1
    domain_tier = 0 if ((worst_force is None or worst_force <= 10.0) and (worst_slip is None or worst_slip <= 2.0)) else 1
    return (
        force_tier,
        max(force_inc, -100.0),
        slip_tier,
        -float(row.get("decoupled_slip_f1") or 0.0),
        consistency_tier,
        domain_tier,
        float(row.get("decoupled_force_rmse_mean_N") or 1e9),
    )


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase3-1 Decoupled Multitask Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- run_id: `{report['run_id']}`",
        f"- decoder_variant: `{report['decoder_variant']}`",
        f"- derived_dataset: `{report['derived_root']}`",
        "- selection_priority: force RMSE non-degradation > slip F1 retained/improved > consistency improvement > per-domain stability.",
        "",
        "## A vs decoupled summary",
        "",
        "| encoder | A force RMSE | decoupled force RMSE | Δ force | A slip F1 | decoupled slip F1 | F1 drop pp | slip acc | contradiction Δ | monotonic Δ | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["ranking_rows"]:
        gate = row["gate_vs_A"]
        lines.append(
            f"| {row['encoder']} | {fmt(row['A_force_rmse_mean_N'])} | {fmt(row['decoupled_force_rmse_mean_N'])} | "
            f"{fmt(gate['force_rmse_increase_pct'], 2)}% | {fmt(row['A_slip_f1'])} | {fmt(row['decoupled_slip_f1'])} | "
            f"{fmt(gate['slip_f1_drop_pp'], 2)} | {fmt(row['decoupled_slip_accuracy'])} | "
            f"{fmt(row['contradiction_delta_vs_A'])} | {fmt(row['monotonic_calibration_delta_vs_A'])} | {gate['status']} |"
        )
    best = report["best_backbone"]
    lines.extend(
        [
            "",
            "## Best backbone decision",
            "",
            f"- best_backbone: `{best['encoder']}`",
            f"- best_checkpoint: `{best.get('checkpoint')}`",
            f"- reason: `{best['selection_reason']}`",
            "",
            "## Per-domain stability snapshot",
            "",
            "| encoder | domain | A force | decoupled force | Δ force | A slip F1 | decoupled slip F1 | F1 drop pp | contradiction | monotonic error |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["ranking_rows"]:
        for domain, drow in sorted(row["domain_summary"].items()):
            a = drow.get("A", {})
            b = drow.get("decoupled", {})
            lines.append(
                f"| {row['encoder']} | {domain} | {fmt(a.get('force_rmse_mean_N'))} | {fmt(b.get('force_rmse_mean_N'))} | "
                f"{fmt(drow.get('force_rmse_increase_pct_vs_A'), 2)}% | {fmt(a.get('slip_f1_weighted_dataset_mean'))} | "
                f"{fmt(b.get('slip_f1_weighted_dataset_mean'))} | {fmt(drow.get('slip_f1_drop_pp_vs_A'), 2)} | "
                f"{fmt(b.get('contradiction_rate'))} | {fmt(b.get('monotonic_calibration_error'))} |"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON report: `{report['json_path']}`",
            f"- Markdown report: `{report['md_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stamp", default=None)
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=ENCODERS)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--slip-horizon", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-eval", action="store_true")
    parser.add_argument("--refresh-reference", action="store_true")
    args = parser.parse_args()

    p2.set_seed(args.seed)
    stamp = args.stamp or args.run_id.replace("phase3_1_decoupled_gsmini_", "")
    report_dir = PHASE3_REPORT_ROOT / f"phase3_1_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    reference_path = report_dir / "phase3_1_force_ratio_reference.json"
    if reference_path.exists() and not args.refresh_reference:
        reference = read_json(reference_path)
    else:
        reference = p2.collect_force_ratio_reference(slip_horizon=args.slip_horizon)
        write_json(reference_path, reference)

    cache_dir = report_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    a_evals: dict[str, Any] = {}
    decoupled_evals: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for enc in args.encoders:
        a_cache = cache_dir / f"a_{enc}_allsource_val.json"
        b_cache = cache_dir / f"decoupled_{enc}_{args.run_id}_allsource_val.json"
        if a_cache.exists() and not args.refresh_eval:
            a_eval = read_json(a_cache)
        else:
            a_eval = p2.evaluate_a_models(enc, reference, args.batch_size, args.num_workers)
            write_json(a_cache, a_eval)
        if b_cache.exists() and not args.refresh_eval:
            b_eval = read_json(b_cache)
        else:
            selection = p2.select_b_checkpoint_for_gate(args.run_id, enc, DECODER_VARIANT, a_eval)
            b_eval = p2.evaluate_b_checkpoint_for_report(
                args.run_id,
                enc,
                reference,
                args.batch_size,
                args.num_workers,
                decoder_variant=DECODER_VARIANT,
                checkpoint_path=selection.get("checkpoint"),
            )
            b_eval["checkpoint_selection"] = selection
            write_json(b_cache, b_eval)
        a_evals[enc] = a_eval
        decoupled_evals[enc] = b_eval
        rows.append(row_from_eval(enc, a_eval, b_eval))

    ranking_rows = sorted(rows, key=rank_key)
    best = dict(ranking_rows[0])
    best["selection_reason"] = (
        "lowest lexicographic rank by force non-degradation, slip F1 retention, "
        "consistency deltas, and per-domain stability"
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "stamp": stamp,
        "phase1_run_id": p2.PHASE1_RUN_ID,
        "derived_root": str(p2.DERIVED_ROOT),
        "decoder_variant": DECODER_VARIANT,
        "encoders": args.encoders,
        "a_evaluations": a_evals,
        "decoupled_evaluations": decoupled_evals,
        "ranking_rows": ranking_rows,
        "best_backbone": best,
        "selection_priority": [
            "force RMSE non-degradation",
            "slip F1 retained/improved",
            "consistency improvement",
            "per-domain stability",
        ],
        "json_path": str(report_dir / "phase3_1_decoupled_multitask_report.json"),
        "md_path": str(report_dir / "phase3_1_decoupled_multitask_report.md"),
    }
    json_path = report_dir / "phase3_1_decoupled_multitask_report.json"
    md_path = report_dir / "phase3_1_decoupled_multitask_report.md"
    write_json(json_path, report)
    md_path.write_text(render_report(report), encoding="utf-8")
    current = PHASE3_REPORT_ROOT / "current_phase3_1_report.md"
    current.write_text(
        "\n".join(
            [
                "# Current Phase3-1",
                "",
                f"- run_id: `{args.run_id}`",
                f"- report: `{md_path}`",
                f"- best_backbone: `{best['encoder']}`",
                f"- generated_at: `{report['generated_at']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(md_path)


if __name__ == "__main__":
    main()
