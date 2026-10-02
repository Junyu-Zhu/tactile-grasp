#!/usr/bin/env python3
"""Build the Round24/G3 evidence synthesis from frozen Round22/23 tables.

This script performs no model loading, inference, threshold fitting, bootstrap
resampling, or test-role access.  It only filters, checks, aggregates, and
copies already accepted validation statistics and confidence intervals.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE.parent
REPO = OUT.parents[2]
G1 = REPO / "experiments/htt_normalflow/round22_921_g1_joint_frozen"
G2 = REPO / "experiments/htt_normalflow/round23_921_g2_force_aux_finetune"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

G1_WP = G1 / "formal_evaluation/detection_core/WORKPOINT_METRICS.csv"
G1_DET_CI = G1 / "formal_evaluation/detection_core/PAIRED_CI.csv"
G1_FORCE = G1 / "formal_evaluation/force_core/METRICS.csv"
G1_FORCE_CI = G1 / "formal_evaluation/force_core/PAIRED_CI.csv"
G2_WP = G2 / "formal_evaluation/evaluation/metrics/WORKPOINT_METRICS.csv"
G2_CI = G2 / "formal_evaluation/evaluation/bootstrap/PAIRED_CI.csv"

RQ1_METRICS = [
    "pAUC", "AP", "frame_static_FPR", "gross_recall",
    "balanced_accuracy", "macro_f1", "event_recall",
    "false_starts_per_trial",
]
RQ2_METRICS = ["delta_mae_n", "absolute_future_mae_n"]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, records: list[dict], fields: list[str] | None = None) -> None:
    if not records:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = fields or list(records[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def fmean(values) -> float:
    values = [float(x) for x in values]
    return sum(values) / len(values)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_design(records: list[dict], groups: set[str], expected: int) -> None:
    keys = {(r["group"], int(r["fold"]), int(r["seed"])) for r in records}
    expected_keys = {
        (group, fold, seed)
        for group in groups
        for fold in range(1, 5)
        for seed in (20260914, 20260915, 20260916)
    }
    assert keys == expected_keys, (len(keys), sorted(expected_keys - keys)[:3])
    assert len(records) == expected


def aggregate_rq1(stage: str, source: Path) -> tuple[list[dict], list[dict]]:
    selected = [
        r for r in rows(source)
        if r["role"] == "validation" and r["policy"] == "FPR5|raw"
    ]
    groups = {"V", "C", "M", "MB"} if stage == "G1_frozen" else {"A", "B", "C", "D"}
    assert_design(selected, groups, 48)
    run_rows = []
    for r in selected:
        out = {"stage": stage, "group": r["group"], "fold": r["fold"], "seed": r["seed"]}
        out.update({m: f"{float(r[m]):.12g}" for m in RQ1_METRICS})
        run_rows.append(out)
    summary = []
    for group in sorted(groups):
        subset = [r for r in run_rows if r["group"] == group]
        out = {"stage": stage, "group": group, "runs": len(subset)}
        out.update({m: f"{fmean(r[m] for r in subset):.12g}" for m in RQ1_METRICS})
        summary.append(out)
    return run_rows, summary


def build_rq1_contrasts(run_rows: list[dict]) -> list[dict]:
    values = {
        (r["stage"], r["group"], r["fold"], r["seed"]): r
        for r in run_rows
    }
    specs = {
        "G1_frozen": [("M-V", "M", "V"), ("C-V", "C", "V"), ("MB-M", "MB", "M")],
        "G2_finetuned": [("B-A", "B", "A"), ("D-C", "D", "C"), ("C-A", "C", "A"), ("D-B", "D", "B")],
    }
    out = []
    for stage, contrasts in specs.items():
        for name, left, right in contrasts:
            for fold in range(1, 5):
                paired = []
                for seed in (20260914, 20260915, 20260916):
                    paired.append((values[(stage, left, str(fold), str(seed))], values[(stage, right, str(fold), str(seed))]))
                row = {"stage": stage, "contrast": name, "fold": fold, "seeds": 3}
                for m in RQ1_METRICS:
                    row[f"delta_{m}"] = f"{fmean(float(a[m]) - float(b[m]) for a, b in paired):.12g}"
                out.append(row)
    return out


def normalize_ci() -> list[dict]:
    out = []
    keep_metrics = {"pAUC", "frame_static_FPR", "gross_recall", "balanced_accuracy", "macro_f1"}
    keep_g1 = {"M-V", "C-V", "MB-M"}
    for r in rows(G1_DET_CI):
        contrast = f'{r["left"]}-{r["right"]}'
        if contrast in keep_g1 and r["metric"] in keep_metrics:
            out.append({
                "stage": "G1_frozen", "contrast": contrast, "fold": r["fold"],
                "metric": r["metric"], "draws": r["draws"], "valid": r["valid"],
                "mean": r["mean"], "q025": r["q025"], "q975": r["q975"],
            })
    keep_g2 = {"B-A", "D-C", "C-A", "D-B", "interaction_(D-B)-(C-A)"}
    for r in rows(G2_CI):
        if r["contrast"] in keep_g2 and r["metric"] in keep_metrics:
            out.append({
                "stage": "G2_finetuned", "contrast": r["contrast"], "fold": r["fold"],
                "metric": r["metric"], "draws": r["attempted_draws"], "valid": r["valid_draws"],
                "mean": r["mean"], "q025": r["q025"], "q975": r["q975"],
            })
    assert len(out) == (3 * 4 * 5) + (5 * 4 * 5)
    return out


def aggregate_rq2() -> tuple[list[dict], list[dict]]:
    selected = [
        r for r in rows(G1_FORCE)
        if r["role"] == "validation" and r["stratum"] == "all" and r["axis"] == "all"
    ]
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in selected:
        key = (r["variant"], int(r["fold"]), int(r["seed"]), int(r["horizon"]))
        buckets[key].append(r)
    wanted = {"hold", "K-V", "K-F", "K-VF", "F2-half", "ridge-K-VF"}
    run_rows = []
    for (variant, fold, seed, horizon), duplicate_rows in sorted(buckets.items()):
        if variant not in wanted:
            continue
        for metric in RQ2_METRICS:
            vals = {round(float(r[metric]), 12) for r in duplicate_rows}
            assert len(vals) == 1, (variant, fold, seed, horizon, metric, vals)
        base = duplicate_rows[0]
        run_rows.append({
            "variant": variant, "fold": fold, "seed": seed, "horizon": horizon,
            "source_duplicate_rows": len(duplicate_rows),
            **{m: f"{float(base[m]):.12g}" for m in RQ2_METRICS},
        })
    assert len(run_rows) == 6 * 4 * 3 * 3
    summary = []
    for variant in sorted(wanted):
        for horizon in (1, 5, 10):
            subset = [r for r in run_rows if r["variant"] == variant and r["horizon"] == horizon]
            summary.append({
                "variant": variant, "horizon": horizon, "runs": len(subset),
                **{m: f"{fmean(r[m] for r in subset):.12g}" for m in RQ2_METRICS},
            })
    return run_rows, summary


def normalize_rq2_ci() -> list[dict]:
    contrasts = {"K-VF-K-V", "K-VF-hold", "K-V-hold", "K-F-hold", "F2-half-K-VF"}
    out = []
    for r in rows(G1_FORCE_CI):
        contrast = f'{r["left"]}-{r["right"]}'
        if contrast not in contrasts:
            continue
        out.append({
            "contrast": contrast, "fold": r["fold"], "horizon": r["horizon"],
            "stratum": r["stratum"], "metric": r["metric"], "draws": r["draws"],
            "valid": r["valid"], "mean": r["mean"], "q025": r["q025"], "q975": r["q975"],
        })
    return out


def svg_bars(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    width, height = 820, 460
    left, top, bottom = 80, 55, 80
    plot_h = height - top - bottom
    vmax = max(values) * 1.15
    bw = (width - left - 30) / len(values)
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-30}" y2="{height-bottom}" stroke="black"/>',
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#ff9da6", "#9d755d"]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * bw + bw * 0.16
        bar_w = bw * 0.68
        bar_h = value / vmax * plot_h
        y = height - bottom - bar_h
        pieces += [
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[i % len(colors)]}"/>',
            f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.4f}</text>',
            f'<text x="{x+bar_w/2:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>',
        ]
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def svg_rq1_two_panels(path: Path, summaries: list[dict]) -> None:
    """Render G1/G2 as separate descriptive panels with an explicit no-ranking guard."""
    width, height = 980, 520
    top, bottom = 92, 88
    panel_w, gap = 410, 70
    panel_lefts = [80, 80 + panel_w + gap]
    plot_h = height - top - bottom
    ymax = 0.5
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="27" text-anchor="middle" font-family="sans-serif" font-size="18">RQ1 validation pAUC descriptive means</text>',
        f'<text x="{width/2}" y="51" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#8b1a1a">Descriptive only; no cross-pipeline ranking or fine-tuning attribution</text>',
        f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14">pAUC(FPR≤0.1)/0.1</text>',
    ]
    for pi, (stage, title) in enumerate((("G1_frozen", "G1: frozen encoder"), ("G2_finetuned", "G2: blocks10/11 fine-tuned"))):
        left = panel_lefts[pi]
        base_y = top + plot_h
        subset = [r for r in summaries if r["stage"] == stage]
        bw = panel_w / len(subset)
        pieces += [
            f'<text x="{left+panel_w/2}" y="{top-13}" text-anchor="middle" font-family="sans-serif" font-size="15">{title}</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{base_y}" stroke="black"/>',
            f'<line x1="{left}" y1="{base_y}" x2="{left+panel_w}" y2="{base_y}" stroke="black"/>',
        ]
        for tick in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            y = base_y - tick / ymax * plot_h
            pieces += [
                f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="black"/>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_w}" y2="{y:.1f}" stroke="#dddddd"/>',
                f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:.1f}</text>',
            ]
        for i, row in enumerate(subset):
            value = float(row["pAUC"])
            x = left + i * bw + bw * 0.19
            bar_w = bw * 0.62
            bar_h = value / ymax * plot_h
            y = base_y - bar_h
            pieces += [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[i]}"/>',
                f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.4f}</text>',
                f'<text x="{x+bar_w/2:.1f}" y="{base_y+23}" text-anchor="middle" font-family="sans-serif" font-size="12">{row["group"]}</text>',
            ]
    pieces += [
        f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="sans-serif" font-size="12">Each bar is the arithmetic mean of 12 validation runs (4 folds × 3 seeds).</text>',
        '</svg>',
    ]
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    g1_runs, g1_summary = aggregate_rq1("G1_frozen", G1_WP)
    g2_runs, g2_summary = aggregate_rq1("G2_finetuned", G2_WP)
    rq1_runs = g1_runs + g2_runs
    rq1_summary = g1_summary + g2_summary
    write_csv(TABLES / "RQ1_RUN_RESULTS.csv", rq1_runs)
    write_csv(TABLES / "RQ1_GROUP_SUMMARY.csv", rq1_summary)
    write_csv(TABLES / "RQ1_FOLD_CONTRASTS.csv", build_rq1_contrasts(rq1_runs))
    write_csv(TABLES / "RQ1_PAIRED_CI.csv", normalize_ci())

    rq2_runs, rq2_summary = aggregate_rq2()
    write_csv(TABLES / "RQ2_RUN_RESULTS.csv", rq2_runs)
    write_csv(TABLES / "RQ2_GROUP_SUMMARY.csv", rq2_summary)
    write_csv(TABLES / "RQ2_PAIRED_CI.csv", normalize_rq2_ci())

    svg_rq1_two_panels(FIGURES / "RQ1_PAUC_MEANS.svg", rq1_summary)
    h10 = [r for r in rq2_summary if r["horizon"] == 10]
    svg_bars(
        FIGURES / "RQ2_H10_DELTA_MAE.svg",
        "RQ2 horizon-10 force-change MAE in N (12 fold/seed runs; descriptive mean)",
        [r["variant"] for r in h10],
        [float(r["delta_mae_n"]) for r in h10],
        "MAE (N; lower is better)",
    )

    inputs = [G1_WP, G1_DET_CI, G1_FORCE, G1_FORCE_CI, G2_WP, G2_CI]
    outputs = sorted(TABLES.glob("*.csv")) + sorted(FIGURES.glob("*.svg"))
    manifest = {
        "schema": "round24_g3_cpu_summary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operation": "read-only aggregation of accepted validation outputs",
        "forbidden_operations_performed": [],
        "test_consumed": False,
        "inputs": [{"path": str(p.relative_to(REPO)), "sha256": sha256(p)} for p in inputs],
        "outputs": [{"path": str(p.relative_to(OUT)), "sha256": sha256(p)} for p in outputs],
        "counts": {
            "rq1_run_rows": len(rq1_runs), "rq1_summary_rows": len(rq1_summary),
            "rq1_ci_rows": len(normalize_ci()), "rq2_run_rows": len(rq2_runs),
            "rq2_summary_rows": len(rq2_summary), "rq2_ci_rows": len(normalize_rq2_ci()),
        },
        "comparability_guard": "G1 and G2 rows retain stage labels and are never ranked as one common pipeline.",
    }
    (OUT / "SUMMARY_BUILD.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
