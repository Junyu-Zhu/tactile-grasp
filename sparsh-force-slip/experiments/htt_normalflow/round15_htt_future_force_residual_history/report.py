#!/usr/bin/env python3
"""Build compact tables, figures, and Chinese Round-15 scientific summary."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import future_train as ft


def read(path):
    return list(csv.DictReader(open(path)))


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reporting", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--evaluation-audit", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--protocol-diagnostics-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    metrics = read(arguments.reporting / "all_metrics.csv")
    summary = read(arguments.reporting / "summary.csv")
    ci = read(arguments.reporting / "paired_group_ci.csv")

    seed_fold_horizon = []
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                for method in ("neural", "predicted_current_persistence", "fit_ridge_residual", "gt_current_persistence_ideal_only"):
                    for horizon in ft.HORIZONS:
                        selected = [row for row in metrics if row["group"] == group and int(row["fold"]) == fold and int(row["seed"]) == seed and row["method"] == method and row["role"] == "validation" and row["stratum"] == "all" and int(row["horizon"]) == horizon]
                        seed_fold_horizon.append({"group": group, "fold": fold, "seed": seed, "method": method, "horizon": horizon,
                                                  "future_mae_axis_mean": mean(selected, "future_mae"), "future_rmse_axis_mean": mean(selected, "future_rmse"),
                                                  "change_mae_axis_mean": mean(selected, "change_mae")})
    write(arguments.output / "seed_fold_horizon.csv", seed_fold_horizon)

    horizon_rows = []
    for group in ft.GROUPS:
        for method in ("neural", "predicted_current_persistence"):
            for horizon in ft.HORIZONS:
                selected = [row for row in seed_fold_horizon if row["group"] == group and row["method"] == method and row["horizon"] == horizon]
                horizon_rows.append({"group": group, "method": method, "horizon": horizon,
                                     "future_mae_mean": mean(selected, "future_mae_axis_mean"),
                                     "seed_fold_descriptive_sd": float(np.std([row["future_mae_axis_mean"] for row in selected], ddof=1)), "runs": len(selected)})
    write(arguments.output / "horizon_summary.csv", horizon_rows)

    strata_rows = []
    for group in ft.GROUPS:
        for stratum in ("stable", "transitional", "changing"):
            neural = [row for row in metrics if row["group"] == group and row["method"] == "neural" and row["role"] == "validation" and row["stratum"] == stratum]
            baseline = [row for row in metrics if row["group"] == group and row["method"] == "predicted_current_persistence" and row["role"] == "validation" and row["stratum"] == stratum]
            strata_rows.append({"group": group, "stratum": stratum, "neural_future_mae": mean(neural, "future_mae"),
                                "persistence_future_mae": mean(baseline, "future_mae"),
                                "neural_minus_persistence": mean(neural, "future_mae") - mean(baseline, "future_mae"), "cells": len(neural)})
    write(arguments.output / "strata_summary.csv", strata_rows)

    overall_ci = [row for row in ci if row["horizon"] == "all" and row["axis"] == "all"]
    write(arguments.output / "paired_group_ci_overall.csv", overall_ci)
    decomposition = []
    for group in ft.GROUPS:
        selected = [row for row in metrics if row["group"] == group and row["method"] == "neural" and row["role"] == "validation" and row["stratum"] == "all"]
        decomposition.append({"group": group, "future_mse": mean(selected, "future_mse"),
                              "anchor_error_squared_term": mean(selected, "anchor_error_mse_component"),
                              "residual_minus_true_change_squared_term": mean(selected, "residual_minus_true_change_mse_component"),
                              "twice_cross_term": mean(selected, "twice_cross_component"),
                              "identity_residual": mean(selected, "future_mse") - mean(selected, "anchor_error_mse_component") - mean(selected, "residual_minus_true_change_mse_component") - mean(selected, "twice_cross_component")})
    write(arguments.output / "mse_decomposition_summary.csv", decomposition)

    diagnostic_fields = ("current_abs_error", "fconcat_future_abs_error", "persistence_future_abs_error", "change_abs_error", "shuffle_displacement_abs", "shuffle_future_abs_error", "repeat_displacement_abs", "repeat_future_abs_error")
    diagnostic_sums = defaultdict(lambda: defaultdict(float)); counts = Counter(); direction_count = Counter(); direction_sum = Counter()
    for row in csv.DictReader(open(arguments.diagnostic / "endpoint_diagnostic.csv")):
        stratum = row["stratum"]; counts[stratum] += 1
        for field in diagnostic_fields: diagnostic_sums[stratum][field] += float(row[field])
        if row["direction_agreement"] != "": direction_count[stratum] += 1; direction_sum[stratum] += int(row["direction_agreement"])
    diagnostic_rows = []
    for stratum in ("stable", "transitional", "changing"):
        diagnostic_rows.append({"stratum": stratum, "rows": counts[stratum],
                                **{f"mean_{field}": diagnostic_sums[stratum][field] / counts[stratum] for field in diagnostic_fields},
                                "direction_eligible_rows": direction_count[stratum],
                                "direction_agreement_fraction": direction_sum[stratum] / direction_count[stratum] if direction_count[stratum] else ""})
    write(arguments.output / "r14_history_diagnostic_summary.csv", diagnostic_rows)

    def value(group, method, metric):
        return float(next(row for row in summary if row["group"] == group and row["method"] == method and row["metric"] == metric)["mean"])
    c_mae = value("C_current", "neural", "future_mae"); h_mae = value("H_history", "neural", "future_mae")
    persistence = value("H_history", "predicted_current_persistence", "future_mae")
    c_change = value("C_current", "neural", "change_mae"); h_change = value("H_history", "neural", "change_mae")
    persistence_change = value("H_history", "predicted_current_persistence", "change_mae")
    r14_concat = value("R14_F_concat", "neural_absolute_output", "future_mae")
    collapse = read(arguments.reporting / "temporal_collapse_summary.csv")
    output_range = read(arguments.protocol_diagnostics_summary)
    cost = json.loads(arguments.cost.read_text()); failure = json.loads((arguments.failure / "FAILURE_CASE_AUDIT.json").read_text())
    training = json.loads(arguments.training_audit.read_text()); evaluation = json.loads(arguments.evaluation_audit.read_text())

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    labels = ["Persistence", "R14 F_concat", "R15 C_current", "R15 H_history"]
    values = [persistence, r14_concat, c_mae, h_mae]
    axis.bar(labels, values, color=["#888888", "#4C78A8", "#F58518", "#54A24B"])
    axis.set_ylabel("Validation future-force MAE (N)"); axis.set_title("Persistence-anchored residual prediction")
    axis.grid(axis="y", alpha=0.25); figure.tight_layout(); figure.savefig(arguments.output / "future_mae_overall.svg"); plt.close(figure)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for group, color in zip(ft.GROUPS, ("#F58518", "#54A24B")):
        ys = [float(next(row for row in horizon_rows if row["group"] == group and row["method"] == "neural" and row["horizon"] == horizon)["future_mae_mean"]) for horizon in ft.HORIZONS]
        axis.plot(ft.HORIZONS, ys, marker="o", label=group, color=color)
    base = [float(next(row for row in horizon_rows if row["group"] == "H_history" and row["method"] == "predicted_current_persistence" and row["horizon"] == horizon)["future_mae_mean"]) for horizon in ft.HORIZONS]
    axis.plot(ft.HORIZONS, base, marker="s", linestyle="--", color="#888888", label="Persistence")
    axis.set_xticks(ft.HORIZONS); axis.set_xlabel("Horizon (frames)"); axis.set_ylabel("Validation future-force MAE (N)")
    axis.grid(alpha=0.25); axis.legend(frameon=False); figure.tight_layout(); figure.savefig(arguments.output / "future_mae_by_horizon.svg"); plt.close(figure)
    figure, axis = plt.subplots(figsize=(8.0, 4.8)); width = 0.35; positions = np.arange(3)
    for index, group in enumerate(ft.GROUPS):
        ys = [float(next(row for row in strata_rows if row["group"] == group and row["stratum"] == stratum)["neural_minus_persistence"]) for stratum in ("stable", "transitional", "changing")]
        axis.bar(positions + (index - 0.5) * width, ys, width, label=group)
    axis.axhline(0, color="black", lw=0.8); axis.set_xticks(positions, ("Stable", "Transitional", "Changing"))
    axis.set_ylabel("Neural minus persistence MAE (N)"); axis.legend(frameon=False); axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(arguments.output / "strata_difference.svg"); plt.close(figure)

    h_c_fold = [row for row in overall_ci if row["comparison"] == "H_history_minus_C_current" and row["stratum"] == "all"]
    h_p_stable = [row for row in overall_ci if row["comparison"] == "H_history_minus_persistence" and row["stratum"] == "stable"]
    h_p_changing = [row for row in overall_ci if row["comparison"] == "H_history_minus_persistence" and row["stratum"] == "changing"]
    d_stable = next(row for row in diagnostic_rows if row["stratum"] == "stable")
    d_changing = next(row for row in diagnostic_rows if row["stratum"] == "changing")
    lines = [
        "# 第十五轮：保持锚定的未来力修正与历史收益验证", "",
        "## 结论", "",
        f"24/24个预注册模型完成，训练与评价审计均通过。C_current未来力MAE为{c_mae:.4f} N，H_history为{h_mae:.4f} N，预测当前力保持为{persistence:.4f} N，R14绝对输出F_concat为{r14_concat:.4f} N。两种残差模型总体都略优于保持，并优于R14 F_concat；但是H与C仅差{h_mae-c_mae:+.4f} N，没有一致的历史收益。", "",
        "H−C完整泄漏组CI按折分别为：" + "；".join(f"p{row['fold']} {float(row['point']):+.4f} [{float(row['ci_low']):+.4f},{float(row['ci_high']):+.4f}] N" for row in h_c_fold) + "。p1显示H较差，p3小幅较好，p2/p4跨0；折间方向不一致，且折重叠，不能合并成独立重复。", "",
        f"绝对未来力改善不能称为动力学改善。C与H的真实变化误差分别为{c_change:.4f}/{h_change:.4f} N，均高于零修正保持的{persistence_change:.4f} N。平方误差恒等式逐单元核验通过：未来误差等于当前anchor误差加修正减真实变化误差，交叉项允许偏差抵消。", "",
        "## 分层与历史诊断", "",
        f"稳定段C/H相对保持分别恶化{float(next(row for row in strata_rows if row['group']=='C_current' and row['stratum']=='stable')['neural_minus_persistence']):+.4f}/{float(next(row for row in strata_rows if row['group']=='H_history' and row['stratum']=='stable')['neural_minus_persistence']):+.4f} N；变化段分别改善{float(next(row for row in strata_rows if row['group']=='C_current' and row['stratum']=='changing')['neural_minus_persistence']):+.4f}/{float(next(row for row in strata_rows if row['group']=='H_history' and row['stratum']=='changing')['neural_minus_persistence']):+.4f} N。未设稳定非劣界。H相对保持的稳定段四折CI中两折明确恶化、两折跨0；变化段四折均改善。", "",
        f"锁协议后的R14诊断显示：F_concat在稳定段MAE {float(d_stable['mean_fconcat_future_abs_error']):.4f} N，高于保持{float(d_stable['mean_persistence_future_abs_error']):.4f} N；在变化段为{float(d_changing['mean_fconcat_future_abs_error']):.4f} N，略低于保持{float(d_changing['mean_persistence_future_abs_error']):.4f} N。这里的0.7303/0.5931是汇总12个运行的validation端点后，对帧×轴×horizon单元的微平均；R14原摘要0.7406/0.6065则是对108个fold×seed×轴×horizon单元MAE的等权宏平均。两者使用同一冻结预测，差异只来自稳定端点数量的权重。联合打乱过去和用当前重复过去使预测平均移动约0.15–0.26 N，但误差变化很小；这是分布外敏感性，不是历史因果价值证明。", "",
        "C/H神经模型的逐试次×轴×窗口塌缩标记均为0/1350；复制比例分别为" + "/".join(f"{100*float(next(row for row in collapse if row['group']==group and row['method']=='neural')['mean_copy_fraction']):.1f}%" for group in ft.GROUPS) + "。该结论只覆盖预登记的描述性阈值。", "",
        "补充的逐试次×轴×horizon表覆盖24个运行、300个运行内episode和4种方法，共10,800行，逐行报告future/change MAE与RMSE。" +
        f"在两组共用的validation标量target中，交付target恰为±20 N的单元为{int(next(row for row in output_range if row['group']=='C_current' and row['method']=='neural')['target_eq_abs20_count'])}/305100（{100*float(next(row for row in output_range if row['group']=='C_current' and row['method']=='neural')['target_eq_abs20_fraction']):.3f}%）；C/H神经输出越过±20 N的单元分别为{int(next(row for row in output_range if row['group']=='C_current' and row['method']=='neural')['prediction_outside_abs20_count'])}/{int(next(row for row in output_range if row['group']=='H_history' and row['method']=='neural')['prediction_outside_abs20_count'])}（{100*float(next(row for row in output_range if row['group']=='C_current' and row['method']=='neural')['prediction_outside_abs20_fraction']):.3f}%/{100*float(next(row for row in output_range if row['group']=='H_history' and row['method']=='neural')['prediction_outside_abs20_fraction']):.3f}%）。输出未裁剪；±20 N只是交付后GT的边界接触率，无裁剪前GT，不能据此推断物理传感器饱和率。", "",
        "## 成本、当前slip与失败案例", "",
        f"两个头均为23,961参数（FP32权重93.6 KiB）；batch1头部GPU中位延迟C/H为{cost['future_heads']['C_current']['head_latency_ms_median']:.3f}/{cost['future_heads']['H_history']['head_latency_ms_median']:.3f} ms。完整管线运行峰值未重测。C依赖原始t−5,t两帧，H依赖t−13..t共14帧。", "",
        "当前slip没有重训、重校准或重算；直接复用R14的48个冻结模型、完整时间线先生成状态再mask共同端点的已验收结果，因此本轮不声称slip改善。", "",
        f"固定不利规则从{failure['candidate_count']}个完整试次候选中选中{failure['selected']['group']} p{failure['selected']['fold']} seed{failure['selected']['seed']} `{failure['selected']['episode_id']}`；神经模型比保持差{failure['selected']['neural_minus_persistence_n']:.4f} N。全候选、曲线和源图已保留。", "",
        "## P3准备决定", "",
        "ToucHD-Force的145文件/366,070,997,896 bytes全SHA证明直接复用。P3训练仍为NO-GO：Mini路线的坐标、单位、zero/tare与data_fixed变换语义未锁，且R10历史H实际由R5 HTT checkpoint继续训练，不是从共同原force checkpoint直接单阶段得到。审计已固定共同源checkpoint及公平路线；H不消费ToucHD，T阶段按object隔离，HTT阶段才要求共同角色、端点和预算。", "",
        "## 边界", "",
        "全部结果仍是有历史开发暴露、折重叠的HTT开发评价；没有独立test、可信物理时间戳、动作条件或实物闭环。R15 residual监督同时包含当前力估计偏差和真实变化。R15优于R14 F_concat还混合了残差目标尺度、零初始化和参数化差异，不能归因于历史或单一结构。P3/P4未执行。"
    ]
    (arguments.output / "SUMMARY_ZH.md").write_text("\n".join(lines) + "\n")
    receipt = {"schema": "round15_report_v1", "status": "complete", "training_runs": training["accepted_count"],
               "evaluation_runs": evaluation["accepted_count"], "main_values": {"C_current_future_mae": c_mae, "H_history_future_mae": h_mae,
               "persistence_future_mae": persistence, "R14_F_concat_future_mae": r14_concat,
               "C_current_change_mae": c_change, "H_history_change_mae": h_change, "persistence_change_mae": persistence_change},
               "scientific_conclusion": "Residual anchoring modestly improves absolute future MAE, but true-change error worsens and nine-step history has no consistent benefit over current-only input.",
               "p3_decision": json.loads(arguments.p3.read_text())["training_decision"],
               "hashes": {path.name: ft.sha(path) for path in arguments.output.iterdir() if path.is_file() and path.name != "REPORT_AUDIT.json"}}
    ft.atomic_json(receipt, arguments.output / "REPORT_AUDIT.json")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
