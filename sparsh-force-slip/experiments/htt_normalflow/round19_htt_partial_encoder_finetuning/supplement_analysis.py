#!/usr/bin/env python3
"""Descriptive R19 validation diagnostics from frozen scores and force-error exports."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return a / b if b else None


def exact_roc(scores, stages):
    keep = np.isin(stages, (0, 2))
    s = scores[keep]
    y = stages[keep]
    static_n = int((y == 0).sum())
    gross_n = int((y == 2).sum())
    if not static_n or not gross_n:
        return [], static_n, gross_n
    distinct, inverse = np.unique(s, return_inverse=True)
    static_at = np.bincount(inverse[y == 0], minlength=len(distinct))
    gross_at = np.bincount(inverse[y == 2], minlength=len(distinct))
    static_cum = np.cumsum(static_at[::-1])
    gross_cum = np.cumsum(gross_at[::-1])
    points = [{"threshold": float(np.nextafter(distinct[-1], np.inf)), "static_fp": 0, "gross_tp": 0, "frame_static_FPR": 0.0, "gross_recall": 0.0, "never_alarm": True}]
    for threshold, fp, tp in zip(distinct[::-1], static_cum, gross_cum):
        points.append({"threshold": float(threshold), "static_fp": int(fp), "gross_tp": int(tp), "frame_static_FPR": float(fp / static_n), "gross_recall": float(tp / gross_n), "never_alarm": False})
    return points, static_n, gross_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = read(args.evaluation / "scores/RAW_SCORES.csv")
    trials = read(args.evaluation / "metrics/TRIAL_METRICS.csv")
    score_groups = defaultdict(list)
    m2_support = defaultdict(list)
    for row in raw:
        fold, seed = int(row["fold"]), int(row["seed"])
        if row["role"] == "validation":
            score_groups[(row["group"], fold, seed)].append(row)
        if row["group"] == "M2" and row["role"] in ("fit", "validation"):
            m2_support[(fold, seed, row["role"])].append(row)

    exact_rows = []
    matched_rows = []
    for (group, fold, seed), rows in sorted(score_groups.items()):
        s = np.asarray([float(x["score"]) for x in rows], dtype=np.float64)
        y = np.asarray([int(x["stage"]) for x in rows], dtype=np.int8)
        points, static_n, gross_n = exact_roc(s, y)
        meta = {"group": group, "fold": fold, "seed": seed, "role": "validation", "use": "descriptive_non_deployable"}
        for point in points:
            exact_rows.append({**meta, "static_support": static_n, "gross_support": gross_n, **point})
        for family, targets in (("same_FPR_max_recall", (.01, .05, .10)), ("same_recall_min_FPR", (.50, .75, .90))):
            for target in targets:
                if not points:
                    matched_rows.append({**meta, "family": family, "target": target, "support_status": "unsupported_class", "threshold": None, "achieved_frame_static_FPR": None, "achieved_gross_recall": None, "exact_target_reachable": False, "unreachable_exact": True, "never_alarm": None, "static_support": static_n, "gross_support": gross_n})
                    continue
                field = "frame_static_FPR" if family.startswith("same_FPR") else "gross_recall"
                feasible = [p for p in points if p[field] <= target + 1e-12] if family.startswith("same_FPR") else [p for p in points if p[field] >= target - 1e-12]
                if family.startswith("same_FPR"):
                    chosen = max(feasible, key=lambda p: (p["gross_recall"], -p["frame_static_FPR"], p["threshold"])) if feasible else None
                else:
                    chosen = min(feasible, key=lambda p: (p["frame_static_FPR"], -p["gross_recall"], -p["threshold"])) if feasible else None
                exact = any(abs(p[field] - target) <= 1e-12 for p in points)
                matched_rows.append({**meta, "family": family, "target": target, "support_status": "supported" if chosen is not None else "unreachable_constraint", "threshold": chosen["threshold"] if chosen else None, "achieved_frame_static_FPR": chosen["frame_static_FPR"] if chosen else None, "achieved_gross_recall": chosen["gross_recall"] if chosen else None, "exact_target_reachable": exact, "unreachable_exact": not exact, "never_alarm": chosen["never_alarm"] if chosen else None, "static_support": static_n, "gross_support": gross_n})

    trial_rows = []
    for (fold, seed, _), _unused in sorted(m2_support.items()):
        if _ != "fit":
            continue
        fit_support = m2_support[(fold, seed, "fit")]
        validation_support = m2_support[(fold, seed, "validation")]
        pred = np.load(args.predictions / "M2" / f"p{fold}_s{seed}" / "PREDICTIONS.npz", allow_pickle=False)
        fit_error = np.asarray(pred["fit_force_error_mae_n"], dtype=np.float64)
        val_error = np.asarray(pred["validation_force_error_mae_n"], dtype=np.float64)
        val_anomaly = np.asarray(pred["validation_anomaly"], dtype=bool)
        if len(fit_support) != len(fit_error) or len(validation_support) != len(val_error):
            raise ValueError(f"M2 force-error support mismatch p{fold} s{seed}")
        fit_by_episode = defaultdict(list)
        for row, error in zip(fit_support, fit_error):
            fit_by_episode[row["episode"]].append(float(error))
        cutoff = float(np.quantile([np.mean(v) for v in fit_by_episode.values()], .90))
        val_by_episode = defaultdict(list)
        for row, error, anomaly in zip(validation_support, val_error, val_anomaly):
            val_by_episode[row["episode"]].append((float(error), bool(anomaly)))
        for row in trials:
            if row["group"] not in ("V2", "M2") or row["role"] != "validation" or row["policy"] != "FPR5|raw" or int(row["fold"]) != fold or int(row["seed"]) != seed:
                continue
            episode = row["episode"]
            force_values = val_by_episode[episode]
            if not force_values:
                raise ValueError(f"missing force support {episode}")
            mean_error = float(np.mean([x for x, _ in force_values]))
            trial_rows.append({"group": row["group"], "fold": fold, "seed": seed, "role": "validation", "policy": "FPR5|raw", "episode": episode, "leakage_group": row["leakage_group"], "fit_trial_mean_force_error_q90_n": cutoff, "trial_mean_force_error_n": mean_error, "force_error_stratum": "high" if mean_error >= cutoff else "other", "force_error_endpoints": len(force_values), "film_anomaly_fraction": float(np.mean([a for _, a in force_values])) if row["group"] == "M2" else None, "static_frames": int(row["static_frames"]), "static_fp": int(row["static_fp"]), "gross_frames": int(row["gross_frames"]), "gross_tp": int(row["gross_tp"]), "false_starts": int(row["false_starts"]), "event_hits": int(row["event_hits"]), "events": int(row["events"])})

    by_stratum = defaultdict(list)
    for row in trial_rows:
        by_stratum[(row["group"], row["fold"], row["seed"], row["force_error_stratum"])].append(row)
    strata_rows = []
    for group, fold, seed in sorted({(r["group"], r["fold"], r["seed"]) for r in trial_rows}):
     for stratum in ("high", "other"):
        rows = by_stratum.get((group, fold, seed, stratum), [])
        static_n = sum(x["static_frames"] for x in rows)
        static_fp = sum(x["static_fp"] for x in rows)
        gross_n = sum(x["gross_frames"] for x in rows)
        gross_tp = sum(x["gross_tp"] for x in rows)
        fpr = ratio(static_fp, static_n)
        recall = ratio(gross_tp, gross_n)
        strata_rows.append({"group": group, "fold": fold, "seed": seed, "role": "validation", "policy": "FPR5|raw", "force_error_stratum": stratum, "trial_count": len(rows), "static_frames": static_n, "static_fp": static_fp, "gross_frames": gross_n, "gross_tp": gross_tp, "frame_static_FPR": fpr, "gross_recall": recall, "balanced_accuracy": (1 - fpr + recall) / 2 if fpr is not None and recall is not None else None, "false_starts": sum(x["false_starts"] for x in rows), "event_hits": sum(x["event_hits"] for x in rows), "events": sum(x["events"] for x in rows), "film_anomaly_fraction_trial_mean": float(np.mean([x["film_anomaly_fraction"] for x in rows])) if group == "M2" and rows else None, "support_status": "both_classes" if fpr is not None and recall is not None else "class_missing"})

    original = defaultdict(list)
    for row in trials:
        if row["group"] in ("V2", "M2") and row["role"] == "validation" and row["policy"] == "FPR5|raw":
            original[(row["group"], int(row["fold"]), int(row["seed"]))].append(row)
    if len(original) != 24 or len(strata_rows) != 48 or len(matched_rows) != 288:
        raise ValueError("supplement grid incomplete")
    for key, rows in original.items():
        subset = [r for r in trial_rows if (r["group"], r["fold"], r["seed"]) == key]
        for field in ("static_frames", "static_fp", "gross_frames", "gross_tp", "false_starts", "event_hits", "events"):
            if sum(r[field] for r in subset) != sum(int(r[field]) for r in rows):
                raise ValueError(f"trial metric replay mismatch {key} {field}")
    for row in matched_rows:
        if row["support_status"] != "supported":
            continue
        value = row["achieved_frame_static_FPR"] if row["family"].startswith("same_FPR") else row["achieved_gross_recall"]
        if row["family"].startswith("same_FPR") and value > row["target"] + 1e-12:
            raise ValueError("matched FPR constraint violation")
        if row["family"].startswith("same_recall") and value < row["target"] - 1e-12:
            raise ValueError("matched recall constraint violation")

    write(args.output / "EXACT_VALIDATION_ROC.csv", exact_rows)
    write(args.output / "MATCHED_VALIDATION_OPERATING_POINTS.csv", matched_rows)
    write(args.output / "FORCE_ERROR_TRIALS.csv", trial_rows)
    write(args.output / "FORCE_ERROR_STRATA.csv", strata_rows)
    summary = {"schema": "round19_supplement_analysis_v1", "status": "complete", "scope": "Validation-only descriptive analysis from unchanged score exports and accepted FPR5 calibration thresholds; no model selection or deployable threshold refit.", "force_error_rule": "For each fold/seed, mean physical predicted-vs-current-GT force MAE (N) over fit endpoints within each complete trial; 90th percentile of those fit trial means is applied to validation trial means. This supplemental descriptive rule was specified after viewing aggregate R19 results and cannot support confirmatory claims. GT force is used only for stratification, never as model input or threshold calibration.", "matched_rule": "Exact distinct validation score thresholds including an empirical validation-only never-alarm sentinel just above the observed maximum (not a guaranteed deployable never-alarm threshold). Same-FPR maximizes recall under achieved static FPR <= target; same-recall minimizes static FPR under achieved gross recall >= target. Exact target unreachable is flagged for discrete empirical rates. These validation-derived thresholds are nondeployable and separate from locked calibration FPR1/5/10 points.", "rows": {"exact_roc": len(exact_rows), "matched": len(matched_rows), "force_error_trials": len(trial_rows), "force_error_strata": len(strata_rows)}, "audit": {"run_grid": len(original), "trial_metric_replay": True, "matched_constraints": True}, "test_consumed": False}
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
