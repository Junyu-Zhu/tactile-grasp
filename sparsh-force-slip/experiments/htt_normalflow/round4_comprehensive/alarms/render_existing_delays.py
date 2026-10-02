#!/usr/bin/env python3
"""Add event-delay columns by rendering already-computed alarm metrics only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.root / "summary.json"
    summary = json.loads(summary_path.read_text())
    metric_paths = sorted((args.root / "runs").glob("*/*/seed_*/metrics.json"))
    if summary.get("status") != "complete" or len(metric_paths) != 48:
        raise RuntimeError("delay rendering requires the accepted complete 48-run metrics")

    rows = []
    for path in metric_paths:
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError(f"incomplete metrics: {path}")
        for operating_point, item in result["operating_points"].items():
            for mode in ("raw", "sequential"):
                if mode not in item:
                    continue
                metrics = item[mode]["validation"]
                rows.append({
                    "model": result["model"], "fold": result["fold"], "seed": result["seed"],
                    "operating_point": operating_point, "mode": mode,
                    **{key: metrics[key] for key in (
                        "balanced_accuracy", "macro_f1", "average_precision", "positive_prevalence",
                        "static_fpr", "gross_recall", "gross_event_recall", "false_alarm_starts_per_trial",
                        "mean_detection_delay_frames", "median_detection_delay_frames", "tn", "fp", "fn", "tp",
                    )},
                })
    if len(rows) != 384:
        raise RuntimeError(f"expected 384 rendered rows, got {len(rows)}")

    target = args.root / "run_metrics.csv"
    temp = target.with_name(target.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temp, target)

    indexed = {(row["model"], row["operating_point"], row["mode"]): [] for row in rows}
    for row in rows:
        indexed[(row["model"], row["operating_point"], row["mode"])].append(row)
    descriptive = []
    displayed = ("balanced_accuracy", "macro_f1", "average_precision", "positive_prevalence", "static_fpr",
                 "gross_recall", "gross_event_recall", "false_alarm_starts_per_trial",
                 "mean_detection_delay_frames", "median_detection_delay_frames")
    for key in sorted(indexed):
        item = {"model": key[0], "operating_point": key[1], "mode": key[2], "runs": len(indexed[key])}
        for metric in displayed:
            values = np.asarray([row[metric] for row in indexed[key]], dtype=float)
            item[f"{metric}_mean"] = float(np.nanmean(values))
            item[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        descriptive.append(item)
    summary["descriptive_mean_std_across_fold_seed_runs"] = descriptive
    summary["rendering_provenance"] = {
        "mode": "existing metrics only; thresholds, rules and per-run metrics unchanged",
        "renderer": str(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "input_metrics_count": len(metric_paths),
        "input_metrics_sha256": {str(path.relative_to(args.root)): sha256(path) for path in metric_paths},
    }
    atomic_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    lines = ["# Round 4 低误报告警评估", "", "已评估 48 个模型运行；状态：`complete`。",
             "阈值及连续告警规则只在 calibration 上确定，然后原样应用到 validation。", "",
             "结果为四折重叠开发评估，不是独立盲测；检测延迟只以帧报告，且只对已检出 gross 事件求均值/中位数。", "",
             "## 覆盖", "", "- dino: 12 runs", "- ijepa: 12 runs", "- mae: 12 runs", "- mae_letterbox: 12 runs", "",
             "## Validation 描述统计", "",
             "| 模型 | 工作点 | 模式 | BA | macro-F1 | AP | 正类占比 | FPR | gross召回 | 事件召回 | 误报启动/试次 | 平均延迟(帧) | 中位延迟(帧) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in descriptive:
        value = lambda name: f"{row[name + '_mean']:.4f}±{row[name + '_std']:.4f}"
        lines.append("| " + " | ".join((row["model"], row["operating_point"], row["mode"],
            value("balanced_accuracy"), value("macro_f1"), value("average_precision"), value("positive_prevalence"),
            value("static_fpr"), value("gross_recall"), value("gross_event_recall"),
            value("false_alarm_starts_per_trial"), value("mean_detection_delay_frames"),
            value("median_detection_delay_frames"))) + " |")
    lines += ["", "均值和标准差只描述重叠的折与三个种子，不将其当作独立重复。每个运行的 `metrics.json` 保留固定 0.5、最大 BA、FPR 1%/5%/10% 工作点、实际 validation 指标、事件检测和 200 次完整 leakage-group bootstrap；`trials.csv` 保留逐试次计数。本次渲染只读取这些既有指标，没有重新选择阈值或规则。", ""]
    atomic_text(args.root / "SUMMARY_ZH.md", "\n".join(lines))
    print(json.dumps({"status": "complete", "metrics": len(metric_paths), "rows": len(rows),
                      "summary_sha256": sha256(summary_path), "run_metrics_sha256": sha256(target)}, indent=2))


if __name__ == "__main__":
    main()
