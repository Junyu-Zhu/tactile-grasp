#!/usr/bin/env python3
"""Create the compact Round-6 legacy-head comparison and Chinese interpretation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def aggregate(payload: dict) -> list[dict]:
    groups = defaultdict(list)
    for domain in ("source", "htt"):
        for row in payload[domain]["results"]:
            metric = row["validation"]
            if metric.get("status") != "available":
                continue
            key = (domain, row["horizon"], row["variant"], row["method"], row["operating_point"])
            groups[key].append(metric)
    output = []
    names = ("average_precision", "balanced_accuracy", "macro_f1", "brier", "fpr", "recall", "prevalence")
    for key, rows in sorted(groups.items()):
        item = dict(zip(("domain", "horizon", "variant", "method", "operating_point"), key))
        item["runs"] = len(rows)
        for name in names:
            values = [r[name] for r in rows]
            item[f"{name}_mean"] = float(np.mean(values))
            item[f"{name}_std_across_seeds"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        output.append(item)
    return output


def lookup(rows, domain, horizon, variant, method, op="fixed_0.5"):
    matches = [r for r in rows if (r["domain"], r["horizon"], r["variant"], r["method"], r["operating_point"]) == (domain, horizon, variant, method, op)]
    if len(matches) != 1:
        raise ValueError(f"headline aggregate missing or duplicated: {(domain,horizon,variant,method,op)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("reevaluation is incomplete")
    rows = aggregate(payload)
    csv_path = args.output_dir / "aggregate_metrics.csv"
    keys = sorted(set().union(*(r.keys() for r in rows)))
    tmp = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, csv_path)

    source = {}
    for horizon in (1, 3, 5):
        best = lookup(rows, "source", horizon, "z_p_slip_force_pred_delta", "raw_future")
        current = lookup(rows, "source", horizon, "shared_baseline", "current_slip")
        source[str(horizon)] = {
            "best_variant": "z_p_slip_force_pred_delta",
            "best_ap": best["average_precision_mean"],
            "current_slip_ap": current["average_precision_mean"],
            "ap_gain": best["average_precision_mean"] - current["average_precision_mean"],
            "fixed_0p5_fpr": best["fpr_mean"],
            "fixed_0p5_recall": best["recall_mean"],
        }
    htt_raw = lookup(rows, "htt", 8, "base", "raw_future")
    htt_gate = lookup(rows, "htt", 8, "base", "raw_mul_current_slip")
    headline = {
        "source": source,
        "htt": {
            "outer_frames": payload["htt"]["counts"]["outer_validation"]["frames"],
            "outer_leakage_groups": payload["htt"]["counts"]["outer_validation"]["groups"],
            "positive_frames": payload["htt"]["counts"]["outer_validation"]["positive_frames"],
            "base_raw_future_ap": htt_raw["average_precision_mean"],
            "base_raw_future_fixed_0p5_ba": htt_raw["balanced_accuracy_mean"],
            "base_raw_future_fixed_0p5_fpr": htt_raw["fpr_mean"],
            "base_raw_future_fixed_0p5_recall": htt_raw["recall_mean"],
            "multiplicative_gate_fixed_0p5_recall": htt_gate["recall_mean"],
        },
    }
    report = f"""# 第六轮旧 future head 与简单基线重评

## 结果范围

- Source 使用 H1/H3/H5，只评价 support manifest 列出的当前 static、首次数据集 slip 标签之前端点，并与第五轮要求 `t>=10`、精确 `t-5/t-10` 的缓存取交集。原 validation 是 outer；阈值 calibration 来自旧模型已经见过的原 train，只是受污染的开发诊断。
- HTT 仅评价第五轮固定 H8。outer validation 只有 {headline['htt']['outer_frames']} 帧、{headline['htt']['outer_leakage_groups']} 个 leakage groups、{headline['htt']['positive_frames']} 个正帧；first-gross 是受力规则影响的标签代理，不是独立物理滑移真值。
- 全部结果复用冻结的第五轮 3 个种子，不读取 test，不重新训练。

## Source 结果

`z+p_slip+force+predicted-force-delta` 在三个窗口的平均 AP 都高于仅用当前 slip 概率：H1 {source['1']['best_ap']:.4f} 对 {source['1']['current_slip_ap']:.4f}（+{source['1']['ap_gain']:.4f}），H3 {source['3']['best_ap']:.4f} 对 {source['3']['current_slip_ap']:.4f}（+{source['3']['ap_gain']:.4f}），H5 {source['5']['best_ap']:.4f} 对 {source['5']['current_slip_ap']:.4f}（+{source['5']['ap_gain']:.4f}）。固定 0.5 下，该模型 H1/H3/H5 的 FPR 分别为 {source['1']['fixed_0p5_fpr']:.4f}/{source['3']['fixed_0p5_fpr']:.4f}/{source['5']['fixed_0p5_fpr']:.4f}，召回为 {source['1']['fixed_0p5_recall']:.4f}/{source['3']['fixed_0p5_recall']:.4f}/{source['5']['fixed_0p5_recall']:.4f}。

单独的预测力差分规则 AP 明显较弱，说明收益来自已训练的联合表征；这些结果不能证明力变化具有物理因果作用。原乘法门控通常压低固定阈值召回，不能作为默认部署门控。

## HTT 结果

旧 raw future head 的排序性能高于 current/history/position/简单力差分基线。base 的三种子平均 AP 为 {headline['htt']['base_raw_future_ap']:.4f}，固定 0.5 的 BA/FPR/recall 为 {headline['htt']['base_raw_future_fixed_0p5_ba']:.4f}/{headline['htt']['base_raw_future_fixed_0p5_fpr']:.4f}/{headline['htt']['base_raw_future_fixed_0p5_recall']:.4f}。risk、base、full_state 的 AP 在这个小 cohort 中相同，无法证明 force/state 辅助带来额外收益。

原乘法门控在固定 0.5 下把 recall 降到 {headline['htt']['multiplicative_gate_fixed_0p5_recall']:.4f}，属于 never-alarm 式退化。calibration 只有 3 个 leakage groups、outer 只有 2 个 groups，阈值和事件召回对种子非常不稳定；不能据此宣称可靠预警。

## 结论边界

旧 future head 在两个离线域均表现出高于简单历史规则的排序信号，其中 Source 的预测力差分联合模型证据较明确。HTT 样本极小、标签是代理、且第五轮 checkpoint 已用同一 validation 做模型选择，所以这里只能说明“存在开发集预测信号”，不能证明轻量级世界模型已经可靠，也不能给出真实物体提前量或成功率。Source 的连续 current-slip 标签可定义数据集内 onset，但没有独立物理真值，本报告不把它解释为真实物理预警。

Brier 对手工规则仅表示 score error，不表示已校准概率质量。告警 run 按原始 `t` 连续性计算；HTT late 仅在完整因果输出轨迹上作诊断，严格 eligible 人口中的缺失尾窗仍记为 censored。
"""
    report_path = args.output_dir / "SUMMARY_ZH.md"
    atomic_text(report_path, report)
    payload["headline"] = headline
    payload["delivery_source_hashes"] = {str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    payload["output_hashes"][csv_path.name] = sha(csv_path)
    payload["output_hashes"][report_path.name] = sha(report_path)
    atomic_text(args.input, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "summary": str(args.input), "report": str(report_path), "aggregate": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
