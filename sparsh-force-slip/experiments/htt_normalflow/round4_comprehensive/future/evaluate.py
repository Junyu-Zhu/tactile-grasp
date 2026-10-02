#!/usr/bin/env python3
"""Evaluate all frozen-protocol round-4 future-risk runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ("mlp", "gru")
SEEDS = (20260914, 20260915, 20260916)
SCORE_COLUMNS = {"mlp": "p_future", "gru": "p_future",
                 "adapted_current_slip": "p_slip_current",
                 "adapted_history4_mean": "p_slip_history4_mean"}
THRESHOLD_RULES = ("max_balanced_accuracy", "fpr_1pct", "fpr_5pct", "fpr_10pct")
HORIZON = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def read_predictions(path: Path, expected_role: str) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"episode_id", "t", "stage", "first_gross_index", "eligible_primary",
                    "target_first_gross_h8", "p_future", "p_slip_current", "p_slip_history4_mean", "role"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"prediction schema mismatch: {path}")
        for raw in reader:
            if raw["role"] != expected_role:
                raise ValueError(f"role mismatch in {path}: {raw['role']}")
            row = {"episode_id": raw["episode_id"], "t": int(raw["t"]), "stage": int(raw["stage"]),
                   "first_gross_index": None if raw["first_gross_index"] == "" else int(raw["first_gross_index"]),
                   "eligible_primary": int(raw["eligible_primary"]),
                   "target_first_gross_h8": None if raw["target_first_gross_h8"] == "" else int(raw["target_first_gross_h8"]),
                   "p_future": float(raw["p_future"]), "p_slip_current": float(raw["p_slip_current"]),
                   "p_slip_history4_mean": float(raw["p_slip_history4_mean"]), "role": raw["role"]}
            if not all(math.isfinite(row[key]) and 0 <= row[key] <= 1 for key in ("p_future", "p_slip_current", "p_slip_history4_mean")):
                raise ValueError(f"invalid probability in {path}")
            if row["eligible_primary"] and (row["stage"] != 0 or row["target_first_gross_h8"] not in (0, 1)):
                raise ValueError(f"invalid eligible target in {path}")
            rows.append(row)
    if not rows or len({(row["episode_id"], row["t"]) for row in rows}) != len(rows):
        raise ValueError(f"empty or duplicate predictions: {path}")
    return rows


def identity(rows: list[dict]) -> list[tuple]:
    return [(r["episode_id"], r["t"], r["stage"], r["first_gross_index"],
             r["eligible_primary"], r["target_first_gross_h8"], r["role"],
             r["p_slip_current"], r["p_slip_history4_mean"]) for r in rows]


def eligible_arrays(rows: list[dict], score_column: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["eligible_primary"]]
    y = np.asarray([row["target_first_gross_h8"] for row in selected], dtype=np.int8)
    scores = np.asarray([row[score_column] for row in selected], dtype=np.float64)
    if len(y) == 0 or set(y.tolist()) != {0, 1}:
        raise ValueError("eligible cohort must contain both classes")
    return y, scores


def confusion(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    positive = y == 1; negative = ~positive
    tp = int(np.sum(pred & positive)); fn = int(np.sum(~pred & positive))
    fp = int(np.sum(pred & negative)); tn = int(np.sum(~pred & negative))
    tpr = tp / (tp + fn); fpr = fp / (fp + tn)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "tpr": tpr, "fpr": fpr,
            "balanced_accuracy": (tpr + (1 - fpr)) / 2}


def threshold_candidates(scores: np.ndarray) -> list[float]:
    unique = sorted(set(float(x) for x in scores))
    return unique + [math.nextafter(1.0, math.inf)]


def calibrate(y: np.ndarray, scores: np.ndarray) -> dict[str, dict]:
    candidates = [(threshold, confusion(y, scores, threshold)) for threshold in threshold_candidates(scores)]
    best_ba = max(candidates, key=lambda item: (item[1]["balanced_accuracy"], -item[1]["fpr"], item[0]))
    result = {"max_balanced_accuracy": {"threshold": best_ba[0], "calibration": best_ba[1]}}
    for percent in (1, 5, 10):
        cap = percent / 100
        valid = [item for item in candidates if item[1]["fpr"] <= cap + 1e-15]
        chosen = max(valid, key=lambda item: (item[1]["tpr"], -item[1]["fpr"], item[0]))
        result[f"fpr_{percent}pct"] = {"threshold": chosen[0], "calibration": chosen[1], "calibration_fpr_cap": cap}
    return result


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y_sorted, scores_sorted = y[order], scores[order]
    cumulative_tp = np.cumsum(y_sorted == 1)
    cumulative_fp = np.cumsum(y_sorted == 0)
    ends = np.r_[np.flatnonzero(scores_sorted[:-1] != scores_sorted[1:]), len(scores_sorted) - 1]
    precision = cumulative_tp[ends] / (cumulative_tp[ends] + cumulative_fp[ends])
    recall = cumulative_tp[ends] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def frame_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return {**confusion(y, scores, threshold), "average_precision": average_precision(y, scores),
            "brier": float(np.mean((scores - y) ** 2)), "positive_prevalence": float(np.mean(y)),
            "frames": len(y), "positive_frames": int(np.sum(y)), "negative_frames": int(np.sum(y == 0))}


def event_metrics(rows: list[dict], score_column: str, threshold: float) -> dict:
    episodes: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        episodes[row["episode_id"]].append(row)
    records = []
    for episode_id, episode in episodes.items():
        positive = [r for r in episode if r["eligible_primary"] and r["target_first_gross_h8"] == 1]
        if not positive:
            continue
        onset_values = {r["first_gross_index"] for r in episode}
        if len(onset_values) != 1 or None in onset_values:
            raise ValueError(f"ambiguous first gross onset: {episode_id}")
        onset = next(iter(onset_values))
        alarms = sorted(r["t"] for r in positive if r[score_column] >= threshold)
        hit = bool(alarms); earliest = alarms[0] if alarms else None
        late = (not hit) and any(onset <= r["t"] <= onset + HORIZON and r[score_column] >= threshold for r in episode)
        records.append({"episode_id": episode_id, "onset": onset, "hit": hit, "miss": not hit,
                        "earliest_alarm": earliest, "lead_frames": None if earliest is None else onset - earliest,
                        "late_detection": late})
    if not records:
        raise ValueError("no supported onset events in validation")
    leads = [row["lead_frames"] for row in records if row["lead_frames"] is not None]
    return {"events": len(records), "hits": sum(row["hit"] for row in records),
            "misses": sum(row["miss"] for row in records), "event_recall": sum(row["hit"] for row in records) / len(records),
            "late_detections_among_misses": sum(row["late_detection"] for row in records),
            "lead_frames_mean_hits": None if not leads else float(np.mean(leads)),
            "lead_frames_median_hits": None if not leads else float(np.median(leads)), "records": records}


def operational_alarm_rates(rows: list[dict], score_column: str, threshold: float) -> dict:
    stage0 = [row for row in rows if row["stage"] == 0]
    no_onset_stage0 = [row for row in stage0 if row["first_gross_index"] is None]
    eligible_negatives = [row for row in rows if row["eligible_primary"] and row["target_first_gross_h8"] == 0]
    rate = lambda selected: None if not selected else float(np.mean([row[score_column] >= threshold for row in selected]))
    return {"all_current_static_alarm_rate": rate(stage0), "all_current_static_frames": len(stage0),
            "no_onset_episode_static_alarm_rate": rate(no_onset_stage0), "no_onset_episode_static_frames": len(no_onset_stage0),
            "eligible_no_onset_within_h8_alarm_rate": rate(eligible_negatives),
            "eligible_no_onset_within_h8_frames": len(eligible_negatives),
            "note": "the model scores every frame online; the primary truth cohort uses the current-static label only for evaluation"}


def validate_run(run_dir: Path, model: str, seed: int, protocol_sha: str) -> tuple[dict, dict[str, list[dict]]]:
    summary_path = run_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or summary.get("model") != model or summary.get("seed") != seed:
        raise ValueError(f"run identity/status mismatch: {run_dir}")
    config = summary["config"]
    if config.get("fold") != "htt_leave_p1" or config.get("horizon") != HORIZON or config.get("history") != 4:
        raise ValueError(f"protocol config mismatch: {run_dir}")
    if config.get("protocol_sha256") != protocol_sha or config.get("smoke"):
        raise ValueError(f"wrong formal protocol: {run_dir}")
    for key in ("best_checkpoint", "latest_checkpoint"):
        if sha256(Path(summary[key])) != summary[f"{key}_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {run_dir}")
    roles = {}
    for role in ("calibration", "validation"):
        info = summary["predictions"][role]; path = Path(info["path"])
        if sha256(path) != info["sha256"]:
            raise ValueError(f"prediction hash mismatch: {run_dir}/{role}")
        roles[role] = read_predictions(path, role)
    return summary, roles


def support(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["eligible_primary"]]
    pos_episodes = {row["episode_id"] for row in eligible if row["target_first_gross_h8"] == 1}
    neg_episodes = {row["episode_id"] for row in eligible if row["target_first_gross_h8"] == 0}
    return {"eligible_frames": len(eligible), "positive_frames": sum(row["target_first_gross_h8"] == 1 for row in eligible),
            "negative_frames": sum(row["target_first_gross_h8"] == 0 for row in eligible),
            "positive_episodes": len(pos_episodes), "negative_episodes": len(neg_episodes)}


def plot_results(records: list[dict], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["adapted_current_slip", "adapted_history4_mean", "mlp", "gru"]
    labels = ["current slip", "history-4 mean", "MLP future", "GRU future"]
    paths = []
    for metric, ylabel, filename, title in (
        ("balanced_accuracy", "Validation balanced accuracy", "validation_balanced_accuracy", "Calibration max-BA threshold"),
        ("average_precision", "Validation average precision", "validation_average_precision", "Threshold-free ranking"),
    ):
        values, errors = [], []
        for method in methods:
            vals = [r["validation"][metric] for r in records
                    if r["method"] == method and r["threshold_rule"] == "max_balanced_accuracy"]
            values.append(float(np.mean(vals))); errors.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(labels, values, yerr=errors, capsize=4, color=["#777777", "#aaaaaa", "#377eb8", "#e41a1c"])
        ax.set_ylabel(ylabel); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
        ax.set_title(f"HTT first-gross risk: {title}")
        fig.tight_layout()
        for suffix in ("svg", "png"):
            path = output / f"{filename}.{suffix}"; fig.savefig(path, dpi=180); paths.append(str(path))
        plt.close(fig)
    return paths


def evaluate(runs_root: Path, protocol: Path, output: Path) -> dict:
    protocol_sha = sha256(protocol.resolve())
    run_data = {}; reference = None
    run_status = []
    for model in MODELS:
        for seed in SEEDS:
            run_dir = runs_root.resolve() / model / f"seed_{seed}"
            summary, roles = validate_run(run_dir, model, seed, protocol_sha)
            for role in roles:
                current_identity = identity(roles[role])
                if reference is None:
                    reference = {name: identity(rows) for name, rows in roles.items()}
                elif current_identity != reference[role]:
                    raise ValueError(f"cohort/baseline mismatch: {run_dir}/{role}")
            run_data[(model, seed)] = roles
            run_status.append({"model": model, "seed": seed, "path": str(run_dir), "status": summary["status"],
                               "best_epoch": summary["best_epoch"], "best_validation_weighted_bce": summary["best_validation_weighted_bce"]})
    records = []
    first_roles = run_data[("mlp", SEEDS[0])]
    methods = [(model, seed, run_data[(model, seed)]) for model in MODELS for seed in SEEDS]
    methods += [("adapted_current_slip", None, first_roles), ("adapted_history4_mean", None, first_roles)]
    for method, seed, roles in methods:
        column = SCORE_COLUMNS[method]
        cal_y, cal_scores = eligible_arrays(roles["calibration"], column)
        val_y, val_scores = eligible_arrays(roles["validation"], column)
        for rule, chosen in calibrate(cal_y, cal_scores).items():
            threshold = chosen["threshold"]
            validation_metrics = frame_metrics(val_y, val_scores, threshold)
            records.append({"method": method, "seed": seed, "threshold_rule": rule, "threshold": threshold,
                            "threshold_is_explicit_never_alarm": threshold > 1.0,
                            "validation_observed_no_alarm": validation_metrics["tp"] + validation_metrics["fp"] == 0,
                            "calibration": {**frame_metrics(cal_y, cal_scores, threshold),
                                            **({"fpr_cap": chosen["calibration_fpr_cap"]} if "calibration_fpr_cap" in chosen else {})},
                            "validation": validation_metrics,
                            "validation_events": event_metrics(roles["validation"], column, threshold),
                            "validation_operational": operational_alarm_rates(roles["validation"], column, threshold)})
    aggregates = []
    for model in MODELS:
        for rule in THRESHOLD_RULES:
            selected = [r for r in records if r["method"] == model and r["threshold_rule"] == rule]
            for metric in ("balanced_accuracy", "fpr", "tpr", "average_precision", "brier"):
                vals = [r["validation"][metric] for r in selected]
                aggregates.append({"model": model, "threshold_rule": rule, "metric": metric,
                                   "mean_across_seeds": float(np.mean(vals)), "std_across_seeds": float(np.std(vals, ddof=1)),
                                   "note": "descriptive variability across three seeds; not an independent confidence interval"})
    output.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_results(records, output)
    payload = {"status": "complete", "task": "current-static first gross in (t,t+8]", "fold": "htt_leave_p1",
               "protocol": str(protocol.resolve()), "protocol_sha256": protocol_sha, "runs": run_status,
               "support": {role: support(rows) for role, rows in first_roles.items()},
               "results": records, "seed_aggregates": aggregates, "plots": plot_paths,
               "claim_limits": ["validation selected checkpoints and is therefore development evidence, not a blind test",
                                "calibration FPR caps are not guarantees for validation FPR",
                                "lead is in frames because no verified sampling interval is available",
                                "the target is first-gross risk, distinct from historical future-any-slip",
                                "no action input is present, so this is a compact causal predictive head rather than a full action-conditioned world model"]}
    atomic_json(output / "evaluation.json", payload)
    flat_path = output / "metrics.csv"
    fields = ["method", "seed", "threshold_rule", "threshold", "calibration_fpr", "validation_fpr",
              "validation_balanced_accuracy", "validation_ap", "validation_brier", "event_recall", "misses",
              "late_detections_among_misses", "mean_lead_frames_hits"]
    temp = flat_path.with_name(flat_path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for r in records:
            e = r["validation_events"]
            writer.writerow({"method": r["method"], "seed": r["seed"], "threshold_rule": r["threshold_rule"],
                             "threshold": r["threshold"], "calibration_fpr": r["calibration"]["fpr"],
                             "validation_fpr": r["validation"]["fpr"], "validation_balanced_accuracy": r["validation"]["balanced_accuracy"],
                             "validation_ap": r["validation"]["average_precision"], "validation_brier": r["validation"]["brier"],
                             "event_recall": e["event_recall"], "misses": e["misses"],
                             "late_detections_among_misses": e["late_detections_among_misses"],
                             "mean_lead_frames_hits": e["lead_frames_mean_hits"]})
    os.replace(temp, flat_path)
    maxba = [r for r in records if r["threshold_rule"] == "max_balanced_accuracy"]
    lines = ["# R4 HTT 短时 first-gross 风险评估", "",
             "本报告只陈述 `htt_leave_p1` 开发划分上的测量结果。validation 同时用于 checkpoint 选择，因此不是独立盲测。", "",
             "## 数据支持", "", f"- calibration: `{payload['support']['calibration']}`", f"- validation: `{payload['support']['validation']}`", "",
             "## calibration 最大 balanced accuracy 阈值", "",
             "| 方法 | seed | validation BA | validation FPR | AP | Brier | event recall | 平均提前帧（命中） | miss | late |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in maxba:
        v, e = r["validation"], r["validation_events"]
        lead = "NA" if e["lead_frames_mean_hits"] is None else f"{e['lead_frames_mean_hits']:.2f}"
        lines.append(f"| {r['method']} | {r['seed'] or '-'} | {v['balanced_accuracy']:.4f} | {v['fpr']:.4f} | {v['average_precision']:.4f} | {v['brier']:.4f} | {e['event_recall']:.4f} | {lead} | {e['misses']} | {e['late_detections_among_misses']} |")
    lines += ["", "## 各阈值规则的三 seed 汇总", "",
              "| 方法 | 阈值规则 | validation BA | validation FPR | validation TPR | event recall |",
              "|---|---|---:|---:|---:|---:|"]
    for method in MODELS:
        for rule in THRESHOLD_RULES:
            selected = [r for r in records if r["method"] == method and r["threshold_rule"] == rule]
            means = lambda key: float(np.mean([r["validation"][key] for r in selected]))
            event_recall = float(np.mean([r["validation_events"]["event_recall"] for r in selected]))
            lines.append(f"| {method} | {rule} | {means('balanced_accuracy'):.4f} | {means('fpr'):.4f} | {means('tpr'):.4f} | {event_recall:.4f} |")
    learned = {model: [r for r in maxba if r["method"] == model] for model in MODELS}
    baseline = next(r for r in maxba if r["method"] == "adapted_current_slip")
    mlp_ba = float(np.mean([r["validation"]["balanced_accuracy"] for r in learned["mlp"]]))
    gru_ba = float(np.mean([r["validation"]["balanced_accuracy"] for r in learned["gru"]]))
    mlp_ap = float(np.mean([r["validation"]["average_precision"] for r in learned["mlp"]]))
    gru_ap = float(np.mean([r["validation"]["average_precision"] for r in learned["gru"]]))
    lines += ["", "## 测量结果与解释", "",
              f"- 在 calibration 最大 BA 阈值下，MLP/GRU 的三 seed 平均 validation BA 为 `{mlp_ba:.4f}`/`{gru_ba:.4f}`，当前 slip 基线为 `{baseline['validation']['balanced_accuracy']:.4f}`。提升幅度较小且 seed 波动明显，不能据此宣称稳定的阈值检测改善。",
              f"- 阈值无关的 AP 更清楚：MLP/GRU 平均为 `{mlp_ap:.4f}`/`{gru_ap:.4f}`，当前 slip 基线为 `{baseline['validation']['average_precision']:.4f}`。这支持 learned head 改善了该开发划分上的 first-gross 风险排序。",
              "- calibration 只有 56 个负帧，最小非零 FPR 步长约为 1.79%。严格 FPR 阈值下，多数组合在 validation 上变成零告警或接近零召回；当前证据不支持低误报率实物部署。",
              "- `evaluation.json` 分开标记协议显式选择的 never-alarm 阈值与在 validation 上碰巧没有告警的普通阈值。",
              "- 最大 BA 阈值下的高 event recall 伴随很高的 validation FPR 和全部 static 帧告警率，因此不能把它单独解释为可靠提前检测。", "",
              "## 证据边界", "",
              "- 1%、5%、10% FPR 是 calibration 上的选择约束；报告中的 validation FPR 才是未改阈值后的实际值。",
              "- 三个 seed 的标准差只描述初始化波动，不是独立样本置信区间。",
              "- learned head 与两个基线使用相同的冻结 MAE 表征和因果信息；结果可检验紧凑动力学头是否改善此代理任务。",
              "- 在线推理不会获得真实 static 标签门控；`evaluation.json` 另报全部当前 static 帧、无 onset 试次 static 帧和 H=8 eligible negative 的告警率。",
              "- event lead/hit 只在预先定义的 first-gross 正窗口统计；late 是对 onset 后完整轨迹的诊断，不能解释成真实物理提前预警。",
              "- 该实验没有动作输入、真实力输入或未来帧输入，不能单独证明完整世界模型或真实物体泛化。", ""]
    atomic_text(output / "REPORT_ZH.md", "\n".join(lines))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); print(json.dumps({k: v for k, v in evaluate(args.runs_root, args.protocol, args.output).items() if k != "results"}, indent=2, ensure_ascii=False))
