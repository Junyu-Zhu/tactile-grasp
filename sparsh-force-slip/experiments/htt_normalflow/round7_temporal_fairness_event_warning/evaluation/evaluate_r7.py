#!/usr/bin/env python3
"""Round-7 complete-timeline event evaluation under the frozen protocol."""
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
PROTOCOL_MD = HERE / "PROTOCOL.md"
AMENDMENTS = tuple(HERE / f"AMENDMENT_0{i}.json" for i in (1, 2, 3, 4, 5))
GROUPS = ("A_visual", "B_force", "C_force_delta", "D_visual_delta")
SEEDS = (20260914, 20260915, 20260916)
RULES = ("raw", "ema_0p5", "confirm2", "ema_0p5_confirm2")
OPS = (("fixed_0.5", "fixed", .5), ("maxBA", "maxBA", None),
       ("frame_FPR_0.01", "frame_fpr", .01), ("frame_FPR_0.05", "frame_fpr", .05), ("frame_FPR_0.10", "frame_fpr", .10),
       ("trial_FA_0.05", "trial_fa", .05), ("trial_FA_0.10", "trial_fa", .10), ("trial_FA_0.20", "trial_fa", .20),
       ("event_recall_0.50", "event_recall", .50), ("event_recall_0.70", "event_recall", .70), ("event_recall_0.90", "event_recall", .90))
BASELINES = ("current_p_slip_raw", "history9_p_slip_mean_raw", "history9_p_slip_slope_fit", "latest_force_delta_fit", "elapsed_position_fit", "fit_prevalence")
_CAL_CACHE: dict[tuple[str,str,str],tuple[list[dict],list[dict]]] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"); os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError(f"empty output {path}")
    keys = list(rows[0])
    if any(list(row) != keys for row in rows): raise ValueError(f"schema drift {path}")
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool): return value
    if str(value) in ("True", "true", "1"): return True
    if str(value) in ("False", "false", "0"): return False
    raise ValueError(f"invalid bool {value!r}")


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8); score = np.asarray(score, dtype=float); positives = int(y.sum())
    if positives == 0: return math.nan
    order = np.argsort(-score, kind="mergesort"); ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]; tp = np.cumsum(ys)[ends]
    return float(np.sum(np.diff(np.r_[0., tp / positives]) * (tp / (ends + 1))))


def causal_rule(values: np.ndarray, rule: str) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)): raise ValueError("invalid probabilities")
    if rule == "raw": return p.copy()
    ema = np.empty_like(p); ema[0] = p[0]
    for i in range(1, len(p)): ema[i] = .5 * p[i] + .5 * ema[i - 1]
    if rule == "ema_0p5": return ema
    base = p if rule == "confirm2" else ema
    if rule not in ("confirm2", "ema_0p5_confirm2"): raise ValueError(rule)
    out = np.zeros_like(base); out[1:] = np.minimum(base[:-1], base[1:]); return out


def add_rules(rows: list[dict], raw_key: str, prefix: str) -> None:
    by_episode = defaultdict(list)
    for row in rows: by_episode[row["episode_id"]].append(row)
    for episode, seq in by_episode.items():
        seq.sort(key=lambda r: r["t"]); times = [r["t"] for r in seq]
        if len(times) != len(set(times)) or times != list(range(times[0], times[-1] + 1)): raise ValueError(f"non-contiguous timeline {episode}")
        if seq[0]["timeline_contiguous"] or any(not r["timeline_contiguous"] for r in seq[1:]): raise ValueError(f"continuity flag drift {episode}")
        values = np.asarray([r[raw_key] for r in seq])
        for rule in RULES:
            for row, value in zip(seq, causal_rule(values, rule)): row[f"{prefix}{rule}"] = float(value)


def frame_confusion(rows: list[dict], score_key: str, threshold: float, mask_key: str) -> dict:
    cohort = [r for r in rows if r[mask_key]]; y = np.asarray([r["target"] for r in cohort], dtype=bool); score = np.asarray([r[score_key] for r in cohort])
    if set(y.tolist()) != {False, True}: raise ValueError("frame cohort lacks both classes")
    alarm = score >= threshold; tp = int(np.sum(alarm & y)); fp = int(np.sum(alarm & ~y)); fn = int(np.sum(~alarm & y)); tn = int(np.sum(~alarm & ~y))
    recall = tp / (tp + fn); fpr = fp / (fp + tn); specificity = 1 - fpr
    f1p = 2 * tp / max(1, 2 * tp + fp + fn); f1n = 2 * tn / max(1, 2 * tn + fp + fn)
    return {"n": len(cohort), "positive": int(y.sum()), "prevalence": float(y.mean()), "average_precision": average_precision(y, score),
            "brier": float(np.mean((score - y) ** 2)), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "frame_fpr": fpr,
            "frame_recall": recall, "balanced_accuracy": .5 * (recall + specificity), "macro_f1": .5 * (f1p + f1n), "never_alarm": bool(not alarm.any())}


def alarm_runs(times: list[int], alarm: list[bool]) -> tuple[int, int, int, float]:
    lengths, current, previous = [], 0, None
    for t, active in zip(times, alarm):
        if active:
            if previous is not None and t == previous + 1: current += 1
            else:
                if current: lengths.append(current)
                current = 1
            previous = t
        else:
            if current: lengths.append(current)
            current, previous = 0, None
    if current: lengths.append(current)
    return len(lengths), sum(lengths), max(lengths, default=0), float(np.mean(lengths)) if lengths else 0.0


def event_metrics(rows: list[dict], score_key: str, threshold: float, mask_key: str) -> tuple[dict, list[dict]]:
    records = [];by_episode=defaultdict(list)
    for row in rows:by_episode[row["episode_id"]].append(row)
    for episode in sorted(by_episode):
        seq = sorted(by_episode[episode], key=lambda r: r["t"]); eligible = [r for r in seq if r[mask_key]]
        onset = seq[0]["onset"]; left = onset is not None and onset <= seq[0]["t"]
        negative = [r for r in eligible if r["target"] == 0]; positive = [r for r in eligible if r["target"] == 1]
        alarm = [r[score_key] >= threshold for r in seq]
        # The first observable row may be in an already-active alarm run. Its true
        # start is left-censored, so it cannot establish causal lead time.
        new_start = [False] + [alarm[i] and not alarm[i - 1] for i in range(1, len(seq))]
        alarm_by_t = {r["t"]: active for r, active in zip(seq, alarm)}
        start_by_t = {r["t"]: start for r, start in zip(seq, new_start)}
        false_alarm = [alarm_by_t[r["t"]] for r in negative]
        false_runs, duration, longest, mean_duration = alarm_runs([r["t"] for r in negative], false_alarm)
        starts = sum(start_by_t[r["t"]] for r in negative)
        overlaps = [r["t"] for r in positive if alarm_by_t[r["t"]]]
        hits = [r["t"] for r in positive if start_by_t[r["t"]]]
        uncensored = onset is not None and not left and bool(positive)
        late = [] if onset is None else [r["t"] for r in seq if r["t"] >= onset and start_by_t[r["t"]]]
        right = onset is None or (not left and not positive)
        records.append({"episode_id": episode, "leakage_group": seq[0]["leakage_group"], "onset_t": onset, "timeline_first_t": seq[0]["t"], "timeline_last_t": seq[-1]["t"],
                        "left_censored": left, "right_censored_or_no_observed_onset": right, "negative_frames": len(negative), "positive_frames": len(positive),
                        "trial_false_alarm": bool(any(false_alarm)), "false_alarm_runs": false_runs, "false_alarm_starts": starts, "false_alarm_duration": duration,
                        "max_false_alarm_duration": longest, "mean_false_alarm_duration": mean_duration, "uncensored_event": uncensored,
                        "event_detected": bool(hits) if uncensored else None, "first_alarm_t": min(hits) if hits else None,
                        "lead_frames": onset - min(hits) if hits else None, "late_alarm": bool(late) if uncensored and not hits else False,
                        "first_late_alarm_t": min(late) if uncensored and not hits and late else None,
                        "active_overlap_detected": bool(overlaps) if uncensored else None,
                        "first_active_overlap_t": min(overlaps) if overlaps else None,
                        "ongoing_from_before_positive_window": bool(overlaps and not hits) if uncensored else None,
                        "active_at_first_observable_row_start_censored": bool(alarm[0]),
                        "observed_no_alarm_complete_timeline": not any(r[score_key] >= threshold for r in seq)})
    negative_trials = [r for r in records if r["negative_frames"]]; events = [r for r in records if r["uncensored_event"]]; hits = [r for r in events if r["event_detected"]]
    return {"trials": len(records), "negative_window_trials": len(negative_trials),
            "trial_false_alarm_rate": float(np.mean([r["trial_false_alarm"] for r in negative_trials])) if negative_trials else math.nan,
            "false_alarm_runs": sum(r["false_alarm_runs"] for r in records), "false_alarm_starts": sum(r["false_alarm_starts"] for r in records), "false_alarm_duration": sum(r["false_alarm_duration"] for r in records),
            "mean_false_alarm_run_duration": sum(r["false_alarm_duration"] for r in records)/sum(r["false_alarm_runs"] for r in records) if sum(r["false_alarm_runs"] for r in records) else 0.,
            "max_false_alarm_duration": max((r["max_false_alarm_duration"] for r in records),default=0),
            "uncensored_events": len(events), "left_censored_events": sum(r["left_censored"] for r in records),
            "right_censored_or_no_observed_onset_trials": sum(r["right_censored_or_no_observed_onset"] for r in records),
            "event_recall": float(np.mean([r["event_detected"] for r in events])) if events else math.nan,
            "active_overlap_recall": float(np.mean([r["active_overlap_detected"] for r in events])) if events else math.nan,
            "ongoing_from_before_positive_window_events": sum(r["ongoing_from_before_positive_window"] for r in events),
            "active_at_first_observable_row_start_censored_trials": sum(r["active_at_first_observable_row_start_censored"] for r in records),
            "mean_lead_frames": float(np.mean([r["lead_frames"] for r in hits])) if hits else math.nan,
            "median_lead_frames": float(np.median([r["lead_frames"] for r in hits])) if hits else math.nan,
            "lead_ge_3_recall": float(np.mean([(r["lead_frames"] or -1) >= 3 for r in events])) if events else math.nan,
            "lead_ge_5_recall": float(np.mean([(r["lead_frames"] or -1) >= 5 for r in events])) if events else math.nan,
            "misses": sum(not r["event_detected"] for r in events), "late_alarms": sum(r["late_alarm"] for r in events),
            "never_alarm_trials": sum(r["observed_no_alarm_complete_timeline"] for r in records)}, records


def threshold_curve(rows: list[dict], score_key: str, mask_key: str) -> list[dict]:
    cohort = [r for r in rows if r[mask_key]]; y = np.asarray([r["target"] for r in cohort], dtype=np.int8); score = np.asarray([r[score_key] for r in cohort])
    if set(y.tolist()) != {0, 1}: raise ValueError("curve lacks both classes")
    order = np.argsort(-score, kind="mergesort"); ys, ss = y[order], score[order]; ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    positives, negatives = int(y.sum()), int(len(y)-y.sum()); tp = np.cumsum(ys)[ends]; fp = ends+1-tp
    result = [{"threshold": math.nextafter(1., math.inf), "frame_recall": 0., "frame_fpr": 0., "balanced_accuracy": .5}]
    for i, end in enumerate(ends):
        rec, fpr = float(tp[i]/positives), float(fp[i]/negatives)
        result.append({"threshold": float(ss[end]), "frame_recall": rec, "frame_fpr": fpr, "balanced_accuracy": .5*(rec+1-fpr)})
    return result


def compact_event_curve(rows: list[dict], score_key: str, mask_key: str, thresholds: list[float]) -> list[dict]:
    by_episode=defaultdict(list)
    for row in rows:by_episode[row["episode_id"]].append(row)
    negative_max=[];events=[];negative_sequences=[]
    for seq in by_episode.values():
        seq=sorted(seq,key=lambda r:r["t"])
        eligible=[r for r in seq if r[mask_key]];negative=[r for r in eligible if r["target"]==0];positive=[r for r in eligible if r["target"]==1]
        if negative:negative_max.append(max(r[score_key] for r in negative))
        previous={seq[i]["t"]:(math.inf if i==0 else seq[i-1][score_key]) for i in range(len(seq))}
        if negative:negative_sequences.append((np.asarray([r[score_key] for r in negative]),np.asarray([previous[r["t"]] for r in negative])))
        onset=seq[0]["onset"]
        if onset is not None and onset>min(r["t"] for r in seq) and positive:
            events.append((onset,np.asarray([r["t"] for r in positive]),np.asarray([r[score_key] for r in positive]),np.asarray([previous[r["t"]] for r in positive])))
    threshold_array=np.asarray(thresholds,float);negative_max=np.asarray(negative_max,float)
    false_alarm=np.mean(negative_max[:,None]>=threshold_array[None,:],axis=0) if len(negative_max) else np.full(len(threshold_array),np.nan)
    false_starts=np.zeros(len(threshold_array),dtype=np.int64);false_duration=np.zeros(len(threshold_array),dtype=np.int64)
    for scores,previous in negative_sequences:
        active=scores[None,:]>=threshold_array[:,None];false_duration+=active.sum(1);false_starts+=np.sum(active&(previous[None,:]<threshold_array[:,None]),axis=1)
    detected_count=np.zeros(len(threshold_array),dtype=np.int64);lead_sum=np.zeros(len(threshold_array),float);lead_count=np.zeros(len(threshold_array),dtype=np.int64)
    for onset,times,scores,previous in events:
        hit=(scores[None,:]>=threshold_array[:,None])&(previous[None,:]<threshold_array[:,None]);detected=hit.any(1);detected_count+=detected
        if detected.any():
            first=np.where(hit,times[None,:],np.iinfo(np.int64).max).min(1);lead=onset-first;lead_sum[detected]+=lead[detected];lead_count[detected]+=1
    recall=detected_count/len(events) if events else np.full(len(threshold_array),np.nan);mean_lead=np.divide(lead_sum,lead_count,out=np.full(len(threshold_array),np.nan),where=lead_count>0)
    return [{"threshold":float(t),"trial_false_alarm_rate":float(fa),"event_recall":float(rec),"mean_lead_frames":float(lead),"false_alarm_starts":int(starts),"false_alarm_duration":int(duration)} for t,fa,rec,lead,starts,duration in zip(threshold_array,false_alarm,recall,mean_lead,false_starts,false_duration)]


def event_tradeoff_thresholds(rows: list[dict], score_key: str, mask_key: str) -> list[float]:
    by_episode=defaultdict(list)
    for row in rows:by_episode[row["episode_id"]].append(row)
    values={math.nextafter(1.,math.inf)}
    for seq in by_episode.values():
        eligible=[r for r in seq if r[mask_key]];negative=[r[score_key] for r in eligible if r["target"]==0];positive=[r[score_key] for r in eligible if r["target"]==1]
        if negative:values.add(max(negative))
        values.update(positive)
    return sorted(values,reverse=True)


def calibration_content_key(rows: list[dict], score_key: str, mask_key: str) -> tuple[str,str,str]:
    h=hashlib.sha256()
    for row in rows:
        h.update(str(row["episode_id"]).encode());h.update(b"\0");h.update(np.int64(row["t"]).tobytes())
        h.update(np.float64(row[score_key]).tobytes());h.update(bytes((bool(row[mask_key]),)))
        target=-1 if row.get("target") is None else int(row["target"]);h.update(np.int8(target).tobytes())
        onset=-1 if row.get("onset") is None else int(row["onset"]);h.update(np.int64(onset).tobytes())
    return h.hexdigest(),score_key,mask_key


def select_threshold(rows: list[dict], score_key: str, mask_key: str, kind: str, target: float | None) -> tuple[float | None, str, dict | None]:
    cache_key=calibration_content_key(rows,score_key,mask_key)
    if cache_key not in _CAL_CACHE:
        curve=threshold_curve(rows,score_key,mask_key);_CAL_CACHE[cache_key]=(curve,compact_event_curve(rows,score_key,mask_key,[r["threshold"] for r in curve]))
    curve,compact_cached=_CAL_CACHE[cache_key]
    if kind == "fixed": threshold = .5
    elif kind == "maxBA": threshold = max(curve, key=lambda r:(r["balanced_accuracy"],-r["frame_fpr"],r["threshold"]))["threshold"]
    elif kind == "frame_fpr": threshold = max((r for r in curve if r["frame_fpr"] <= float(target)+1e-15), key=lambda r:(r["frame_recall"],-r["frame_fpr"],r["threshold"]))["threshold"]
    else:
        summaries = [(r["threshold"], detail) for r,detail in zip(curve,compact_cached)]
        if kind == "trial_fa": valid = [x for x in summaries if x[1]["trial_false_alarm_rate"] <= float(target)+1e-15]
        elif kind == "event_recall": valid = [x for x in summaries if x[1]["event_recall"] >= float(target)-1e-15]
        else: raise ValueError(kind)
        if not valid: return None, "unavailable_calibration_constraint", None
        if kind == "trial_fa":
            best=max(x[1]["event_recall"] for x in valid);valid=[x for x in valid if x[1]["event_recall"]==best]
            best=min(x[1]["trial_false_alarm_rate"] for x in valid);valid=[x for x in valid if x[1]["trial_false_alarm_rate"]==best]
        else:
            best=min(x[1]["trial_false_alarm_rate"] for x in valid);valid=[x for x in valid if x[1]["trial_false_alarm_rate"]==best]
        if kind == "trial_fa":threshold,_=max(valid,key=lambda x:(-x[1]["false_alarm_starts"],-x[1]["false_alarm_duration"],x[0]))
        else:threshold,_=max(valid,key=lambda x:(-x[1]["false_alarm_starts"],-x[1]["false_alarm_duration"],x[1]["mean_lead_frames"] if math.isfinite(x[1]["mean_lead_frames"]) else -math.inf,x[0]))
        return threshold,"available",event_metrics(rows,score_key,threshold,mask_key)[0]
    return threshold, "available", event_metrics(rows, score_key, threshold, mask_key)[0]


def select_rule(rows: list[dict], prefix: str, mask_key: str) -> tuple[str, list[dict]]:
    evidence = []
    for order, rule in enumerate(RULES):
        threshold, status, detail = select_threshold(rows, prefix+rule, mask_key, "trial_fa", .20)
        if status != "available": raise ValueError(f"rule selection unavailable {rule}")
        evidence.append({"rule":rule,"fixed_order":order,"threshold":threshold,**detail})
    selected = max(evidence,key=lambda r:(r["event_recall"],-r["trial_false_alarm_rate"],-r["false_alarm_starts"],-r["false_alarm_duration"],-r["fixed_order"]))["rule"]
    return selected, evidence


def parse_timeline(path: Path, horizons: list[int]) -> list[dict]:
    raw = list(csv.DictReader(path.open())); rows=[]; seen=set()
    required={"episode_id","leakage_group","t","first_current_slip_t","current_slip_label","p_slip_current","timeline_contiguous","common_population"}
    if not raw or required-set(raw[0]): raise ValueError(f"timeline schema {path}")
    for source in raw:
        key=(source["episode_id"],int(source["t"]));
        if key in seen: raise ValueError(f"duplicate {key}")
        seen.add(key); onset=source["first_current_slip_t"]
        row={"episode_id":source["episode_id"],"leakage_group":source["leakage_group"],"t":int(source["t"]),"onset":None if onset in ("","None") else int(onset),
             "current_slip_label":int(source["current_slip_label"]),"p_slip_current":float(source["p_slip_current"]),"timeline_contiguous":as_bool(source["timeline_contiguous"]),
             "common_population":as_bool(source["common_population"]),"force_abs_fz":float(source["force_abs_fz"]),"force_ft":float(source["force_ft"]),
             "force_ft_over_fn":float(source["force_ft_over_fn"]),"delta_x":float(source["latest_force_delta_x"]),"delta_y":float(source["latest_force_delta_y"]),"delta_z":float(source["latest_force_delta_z"]),"delta_valid":as_bool(source["latest_force_delta_valid"])}
        for h in horizons:
            eligible=as_bool(source[f"eligible_H{h}"]); target=source[f"target_future_H{h}"]
            row[f"eligible_H{h}"]=eligible; row[f"common_H{h}"]=as_bool(source[f"common_eligible_H{h}"])
            row[f"right_censored_H{h}"]=as_bool(source[f"right_censored_H{h}"]); row[f"target_H{h}"]=None if target=="" else int(target)
            row[f"p_H{h}"]=float(source[f"p_future_H{h}_raw"])
            if eligible != (row[f"target_H{h}"] is not None): raise ValueError(f"target/mask drift {key}/H{h}")
        rows.append(row)
    return rows


def role_rows(rows: list[dict], horizon: int, population: str) -> list[dict]:
    mask = "common_population" if population == "primary" else f"eligible_H{horizon}"
    result=[]
    for row in rows:
        copy=dict(row); copy["metric_mask"] = bool(row[mask]); copy["target"] = row[f"target_H{horizon}"] if copy["metric_mask"] else None; result.append(copy)
    return result


def verify_run(folder: Path, group: str, seed: int, prepared_sha: str, horizons: list[int]) -> tuple[dict, dict[str,list[dict]], dict[str,str]]:
    summary_path=folder/"summary.json"; summary=json.loads(summary_path.read_text())
    if summary.get("status")!="complete" or not summary.get("formal") or summary.get("smoke") or summary.get("group")!=group or int(summary.get("seed"))!=seed: raise ValueError(f"invalid formal {folder}")
    if summary["run_config"].get("data_sha256")!=prepared_sha or summary["run_config"].get("horizons")!=horizons: raise ValueError(f"formal input drift {folder}")
    sources={str(summary_path.resolve()):sha256(summary_path)}; roles={}
    for role in ("selection","calibration","outer"):
        spec=summary["artifacts"]["predictions"][role]["timeline"]; path=Path(spec["path"])
        if path.resolve()!=(folder/f"predictions_timeline_{role}.csv").resolve() or sha256(path)!=spec["sha256"]: raise ValueError(f"timeline provenance {folder}/{role}")
        sources[str(path.resolve())]=sha256(path); roles[role]=parse_timeline(path,horizons)
    return summary,roles,sources


def load_manifest_timelines(manifest_path: Path, expected_schema: str, groups: tuple[str,...], horizons: list[int]) -> tuple[dict,dict[str,str]]:
    manifest=json.loads(manifest_path.read_text());sources={str(manifest_path.resolve()):sha256(manifest_path)}
    if manifest.get("status")!="complete" or manifest.get("schema")!=expected_schema:raise ValueError(f"invalid manifest {manifest_path}")
    artifacts={};expected={(kind,g,s,r) for kind in (["reference"] if expected_schema.startswith("round7_r6") else ["zero_fit_mean","causal_lag1"]) for g in groups for s in SEEDS for r in ("selection","calibration","outer")}
    for spec in manifest["artifacts"]:
        kind="reference" if expected_schema.startswith("round7_r6") else spec["intervention"];key=(kind,spec["group"],int(spec["seed"]),spec["role"]);path=Path(spec["path"])
        if key in artifacts or sha256(path)!=spec["sha256"]:raise ValueError(f"manifest artifact drift {key}")
        rows=parse_timeline(path,horizons);artifacts[key]=rows;sources[str(path.resolve())]=sha256(path)
    if set(artifacts)!=expected:raise ValueError(f"manifest identity mismatch missing={expected-set(artifacts)} extra={set(artifacts)-expected}")
    return artifacts,sources


def fit_logistic(x: np.ndarray, y: np.ndarray, lam: float=.001) -> dict:
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); mean=x.mean(0); std=np.maximum(x.std(0),1e-6); z=(x-mean)/std
    design=np.c_[np.ones(len(z)),z]; w=np.zeros(design.shape[1]); penalty=np.diag(np.r_[0.,np.full(z.shape[1],lam)])
    for _ in range(100):
        logits=np.clip(design@w,-40,40); p=1/(1+np.exp(-logits)); grad=design.T@(p-y)/len(y)+penalty@w
        hess=(design.T*(p*(1-p)))@design/len(y)+penalty+np.eye(len(w))*1e-10; step=np.linalg.solve(hess,grad); w-=step
        if np.max(np.abs(step))<1e-10: break
    if not np.isfinite(w).all(): raise FloatingPointError("baseline fit")
    return {"mean":mean,"std":std,"weights":w,"lambda":lam}


def predict_logistic(model: dict, x: np.ndarray) -> np.ndarray:
    z=(np.asarray(x)-model["mean"])/model["std"]; logits=np.clip(np.c_[np.ones(len(z)),z]@model["weights"],-40,40); return 1/(1+np.exp(-logits))


def fit_monotone_platt(rows: list[dict], score_key: str, mask_key: str) -> dict:
    cohort=[r for r in rows if r[mask_key]];y=torch.tensor([r["target"] for r in cohort],dtype=torch.float64)
    if int(torch.unique(y).numel())!=2:return {"status":"unavailable_single_class"}
    q=torch.tensor([r[score_key] for r in cohort],dtype=torch.float64).clamp(1e-6,1-1e-6);x=torch.logit(q)
    raw_slope=torch.tensor(math.log(math.expm1(1.0)),dtype=torch.float64,requires_grad=True);intercept=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([raw_slope,intercept],lr=1.,max_iter=100,line_search_fn="strong_wolfe",tolerance_grad=1e-12,tolerance_change=1e-12)
    def closure():
        optimizer.zero_grad();slope=torch.nn.functional.softplus(raw_slope);loss=torch.nn.functional.binary_cross_entropy_with_logits(slope*x+intercept,y);loss.backward();return loss
    optimizer.step(closure);slope=float(torch.nn.functional.softplus(raw_slope).detach());bias=float(intercept.detach())
    if not math.isfinite(slope) or not math.isfinite(bias) or slope<=0:return {"status":"unavailable_nonfinite"}
    return {"status":"available","slope":slope,"intercept":bias,"epsilon":1e-6,"fit_rows":len(cohort)}


def apply_platt(score: float, model: dict) -> float:
    q=min(max(float(score),model["epsilon"]),1-model["epsilon"]);logit=math.log(q/(1-q));z=max(min(model["slope"]*logit+model["intercept"],40),-40);return 1/(1+math.exp(-z))


def reliability_rows(rows: list[dict], score_key: str, mask_key: str, group: str, seed: int, horizon: int, role: str, method: str) -> list[dict]:
    cohort=[r for r in rows if r[mask_key]];result=[]
    for index in range(10):
        lo,hi=index/10,(index+1)/10;selected=[r for r in cohort if lo<=r[score_key]<(hi if index<9 else hi+1e-15)]
        result.append({"group":group,"seed":seed,"horizon":horizon,"role":role,"method":method,"bin":index,"lower":lo,"upper":hi,"n":len(selected),
                       "mean_probability":float(np.mean([r[score_key] for r in selected])) if selected else "","positive_fraction":float(np.mean([r["target"] for r in selected])) if selected else ""})
    return result


def record_summary(records: list[dict]) -> dict[str,float]:
    negative=[r for r in records if r["negative_frames"]]
    events=[r for r in records if r["uncensored_event"]]
    detected=[r for r in events if r["event_detected"]]
    return {
        "trial_false_alarm_rate":float(np.mean([r["trial_false_alarm"] for r in negative])) if negative else math.nan,
        "event_recall":float(np.mean([r["event_detected"] for r in events])) if events else math.nan,
        "false_alarm_starts_per_trial":float(np.mean([r["false_alarm_starts"] for r in records])) if records else math.nan,
        "false_alarm_duration_per_trial":float(np.mean([r["false_alarm_duration"] for r in records])) if records else math.nan,
        "mean_lead_frames":float(np.mean([r["lead_frames"] for r in detected])) if detected else math.nan,
    }


def paired_bootstrap_ci(trial_rows: list[dict], horizons: list[int]) -> list[dict]:
    """Shared leakage-group draws across methods and seeds; seeds are averaged, not resampled."""
    result=[]
    indexed=defaultdict(list)
    for row in trial_rows:
        if row["population"]=="primary" and row["method_type"]=="neural" and row["operating_point"] in ("trial_FA_0.10","event_recall_0.70"):
            indexed[(row["group"],int(row["seed"]),int(row["horizon"]),row["operating_point"])].append(row)
    for h in horizons:
        for op in ("trial_FA_0.10","event_recall_0.70"):
            keys=[(g,s,h,op) for g in GROUPS for s in SEEDS]
            if any(k not in indexed for k in keys): continue
            group_sets=[{r["leakage_group"] for r in indexed[k]} for k in keys]
            if any(gs!=group_sets[0] for gs in group_sets[1:]): raise ValueError(f"bootstrap group identity drift H{h}/{op}")
            groups=sorted(group_sets[0]); rng=np.random.default_rng(700000+h*100+(10 if op.startswith("trial") else 70))
            draws=[rng.choice(groups,size=len(groups),replace=True).tolist() for _ in range(200)]
            def aggregate(group: str, seed: int, draw: list[str] | None) -> dict:
                rows=indexed[(group,seed,h,op)]; by_group=defaultdict(list)
                for row in rows:by_group[row["leakage_group"]].append(row)
                sample=rows if draw is None else [r for g in draw for r in by_group[g]]
                return record_summary(sample)
            for comparator in ("A_visual","B_force","D_visual_delta"):
                for metric in ("event_recall","trial_false_alarm_rate","false_alarm_starts_per_trial","false_alarm_duration_per_trial","mean_lead_frames"):
                    sign=1. if metric in ("event_recall","mean_lead_frames") else -1.
                    point=float(np.mean([sign*(aggregate("C_force_delta",s,None)[metric]-aggregate(comparator,s,None)[metric]) for s in SEEDS]))
                    boot=[]
                    for draw in draws:
                        values=[sign*(aggregate("C_force_delta",s,draw)[metric]-aggregate(comparator,s,draw)[metric]) for s in SEEDS]
                        if all(math.isfinite(x) for x in values):boot.append(float(np.mean(values)))
                    result.append({"comparison":f"C_force_delta_vs_{comparator}","horizon":h,"operating_point":op,"metric":metric,"benefit_positive":True,"estimate":point,
                                   "ci_low":float(np.percentile(boot,2.5)) if boot else "","ci_high":float(np.percentile(boot,97.5)) if boot else "","valid_bootstrap_replicates":len(boot),"requested_bootstrap_replicates":200})
            # Incremental attribution: (C-B) - (D-A); cost metrics are sign-reversed so positive remains favorable.
            for metric in ("event_recall","trial_false_alarm_rate","false_alarm_starts_per_trial","false_alarm_duration_per_trial","mean_lead_frames"):
                sign=1. if metric in ("event_recall","mean_lead_frames") else -1.
                def did(seed: int, draw: list[str] | None) -> float:
                    val={g:aggregate(g,seed,draw)[metric] for g in GROUPS}
                    return sign*((val["C_force_delta"]-val["B_force"])-(val["D_visual_delta"]-val["A_visual"]))
                point=float(np.mean([did(s,None) for s in SEEDS]));boot=[]
                for draw in draws:
                    values=[did(s,draw) for s in SEEDS]
                    if all(math.isfinite(x) for x in values):boot.append(float(np.mean(values)))
                result.append({"comparison":"incremental_(C-B)-(D-A)","horizon":h,"operating_point":op,"metric":metric,"benefit_positive":True,"estimate":point,
                               "ci_low":float(np.percentile(boot,2.5)) if boot else "","ci_high":float(np.percentile(boot,97.5)) if boot else "","valid_bootstrap_replicates":len(boot),"requested_bootstrap_replicates":200})
    return result


def rank_correlation(x: list[float], y: list[float]) -> float:
    def ranks(values: list[float]) -> np.ndarray:
        values=np.asarray(values,float);order=np.argsort(values,kind="mergesort");result=np.empty(len(values),float)
        start=0
        while start<len(values):
            end=start+1
            while end<len(values) and values[order[end]]==values[order[start]]:end+=1
            result[order[start:end]]=(start+end-1)/2+1;start=end
        return result
    rx,ry=ranks(x),ranks(y)
    return float(np.corrcoef(rx,ry)[0,1]) if len(x)>=2 and rx.std()>0 and ry.std()>0 else math.nan


def render_failure_cases(out: Path, all_runs: dict, selected_rules: dict, selected_thresholds: dict, horizon: int=3) -> list[dict]:
    cases=[];figdir=out/"figures"/"failure_cases";figdir.mkdir(parents=True,exist_ok=True);seed=SEEDS[0];op="trial_FA_0.10"
    categories=("early_false_alarm","force_change_in_observed_negative_window","late_or_censored","position_dependence","missed_uncensored_event","no_post_onset_alert","actual_late_new_start","confirmation_lost_detection","confirmation_delay","longest_stable_trial","current_p_slip_copy_correlation")
    for group in GROUPS:
        rows=role_rows(all_runs[(group,seed)]["roles"]["outer"],horizon,"primary");rule=selected_rules[(group,seed,horizon)];score_key=f"H{horizon}_{rule}";threshold=selected_thresholds[(group,seed,horizon,op)]
        by_episode=defaultdict(list)
        for row in rows:by_episode[row["episode_id"]].append(row)
        _,records=event_metrics(rows,score_key,threshold,"metric_mask");record_by_id={r["episode_id"]:r for r in records};candidates={}
        early=[r for r in records if r["negative_frames"] and r["false_alarm_duration"]]
        if early:candidates["early_false_alarm"]=max(early,key=lambda r:(r["false_alarm_duration"],r["false_alarm_starts"],r["episode_id"]))
        force=[]
        for record in records:
            seq=by_episode[record["episode_id"]];alarmed=[r for r in seq if r["metric_mask"] and r["target"]==0 and r[score_key]>=threshold]
            if alarmed:force.append((max(float(np.linalg.norm([r["delta_x"],r["delta_y"],r["delta_z"]])) for r in alarmed),record))
        if force:candidates["force_change_in_observed_negative_window"]=max(force,key=lambda x:(x[0],x[1]["episode_id"]))[1]
        censored=[]
        for record in records:
            if record["right_censored_or_no_observed_onset"]:
                seq=by_episode[record["episode_id"]];alarms=[r["t"] for r in seq if r[score_key]>=threshold]
                if alarms:censored.append((max(alarms)-seq[-1]["t"],record))
        if censored:candidates["late_or_censored"]=max(censored,key=lambda x:(x[0],x[1]["episode_id"]))[1]
        position=[];copycorr=[]
        for record in records:
            seq=[r for r in by_episode[record["episode_id"]] if r["metric_mask"]]
            if len(seq)>=10:
                rho=rank_correlation([r["t"]-seq[0]["t"] for r in seq],[r[score_key] for r in seq])
                corr=rank_correlation([r["p_slip_current"] for r in seq],[r[score_key] for r in seq])
                if math.isfinite(rho):position.append((abs(rho),rho,record))
                if math.isfinite(corr):copycorr.append((abs(corr),corr,record))
        if position:candidates["position_dependence"]=max(position,key=lambda x:(x[0],x[2]["episode_id"]))[2]
        if copycorr:candidates["current_p_slip_copy_correlation"]=max(copycorr,key=lambda x:(x[0],x[2]["episode_id"]))[2]
        missed=[r for r in records if r["uncensored_event"] and not r["event_detected"]]
        if missed:candidates["missed_uncensored_event"]=max(missed,key=lambda r:(max((x[score_key] for x in by_episode[r["episode_id"]] if x["metric_mask"] and x["target"]==1),default=-1),r["episode_id"]))
        no_post=[];actual_late=[]
        for record in records:
            if not record["uncensored_event"]:continue
            seq=by_episode[record["episode_id"]];post=[r for r in seq if r["t"]>=record["onset_t"] and r[score_key]>=threshold]
            if not record["event_detected"] and not post:no_post.append(record)
            if record["late_alarm"]:actual_late.append(record)
        if no_post:candidates["no_post_onset_alert"]=min(no_post,key=lambda r:r["episode_id"])
        if actual_late:candidates["actual_late_new_start"]=min(actual_late,key=lambda r:(r["first_late_alarm_t"]-r["onset_t"],r["episode_id"]))
        raw_summary,raw_records=event_metrics(rows,f"H{horizon}_raw",threshold,"metric_mask");raw_by_id={r["episode_id"]:r for r in raw_records}
        lost=[record_by_id[e] for e,r in raw_by_id.items() if r["uncensored_event"] and r["event_detected"] and not record_by_id[e]["event_detected"]]
        if lost:candidates["confirmation_lost_detection"]=min(lost,key=lambda r:r["episode_id"])
        delays=[]
        for e,raw_record in raw_by_id.items():
            selected_record=record_by_id[e]
            if raw_record["uncensored_event"] and raw_record["event_detected"] and selected_record["event_detected"]:
                delays.append((selected_record["first_alarm_t"]-raw_record["first_alarm_t"],selected_record))
        if delays and max(x[0] for x in delays)>0:candidates["confirmation_delay"]=max(delays,key=lambda x:(x[0],x[1]["episode_id"]))[1]
        stable=[r for r in records if r["negative_frames"]]
        if stable:candidates["longest_stable_trial"]=max(stable,key=lambda r:(r["negative_frames"],r["episode_id"]))
        for category in categories:
            record=candidates.get(category)
            base={"group":group,"seed":seed,"horizon":horizon,"category":category,"availability":"available" if record else "unavailable","episode_id":"" if record is None else record["episode_id"],"rule":rule,"operating_point":op,"threshold":threshold}
            if record is None:
                cases.append({**base,"false_alarm_starts":"","false_alarm_duration":"","event_detected":"","active_overlap_detected":"","lead_frames":"","position_score_spearman":"","current_p_slip_score_spearman":"","input_background_cause":"unobservable_from_bound_timeline_schema"});continue
            seq=by_episode[record["episode_id"]];metric_seq=[r for r in seq if r["metric_mask"]];rho=rank_correlation([r["t"]-metric_seq[0]["t"] for r in metric_seq],[r[score_key] for r in metric_seq]) if metric_seq else math.nan;copy=rank_correlation([r["p_slip_current"] for r in metric_seq],[r[score_key] for r in metric_seq]) if metric_seq else math.nan
            cases.append({**base,"false_alarm_starts":record["false_alarm_starts"],"false_alarm_duration":record["false_alarm_duration"],"event_detected":record["event_detected"],"active_overlap_detected":record["active_overlap_detected"],"lead_frames":record["lead_frames"] if record["lead_frames"] is not None else "","position_score_spearman":rho if math.isfinite(rho) else "","current_p_slip_score_spearman":copy if math.isfinite(copy) else "","input_background_cause":"unobservable_from_bound_timeline_schema"})
            fig,ax=plt.subplots(figsize=(8.5,4));times=[r["t"] for r in seq];ax.plot(times,[r[score_key] for r in seq],label="selected warning");ax.plot(times,[r[f"H{horizon}_raw"] for r in seq],alpha=.65,label="raw warning");ax.plot(times,[r["p_slip_current"] for r in seq],alpha=.65,label="current pSlip");ax.axhline(threshold,color="tab:red",ls="--",label="cal threshold")
            onset=seq[0]["onset"]
            if onset is not None:ax.axvline(onset,color="black",ls=":",label="dataset-label onset")
            ax2=ax.twinx();ax2.plot(times,[float(np.linalg.norm([r["delta_x"],r["delta_y"],r["delta_z"]])) for r in seq],color="tab:green",alpha=.35,label="|force delta|");ax2.set_ylabel("predicted force-change norm")
            ax.set(title=f"{group} {category}: {record['episode_id']}",xlabel="frame t",ylabel="probability",ylim=(0,1));ax.grid(alpha=.25);lines=ax.lines+ax2.lines;ax.legend(lines,[x.get_label() for x in lines],fontsize=7,ncol=2);fig.tight_layout();fig.savefig(figdir/f"{group}_{category}.png",dpi=160);plt.close(fig)
    return cases


def baseline_features(row: dict, name: str) -> list[float]:
    history=row["p_history"]
    if name=="current_p_slip_raw": return [history[-1]]
    if name=="history9_p_slip_mean_raw": return [float(np.mean(history))]
    if name=="history9_p_slip_slope_fit": return [float(np.polyfit(np.arange(9),history,1)[0])]
    if name=="latest_force_delta_fit": return [row["delta_x"],row["delta_y"],row["delta_z"],float(np.linalg.norm([row["delta_x"],row["delta_y"],row["delta_z"]]))]
    if name=="elapsed_position_fit": return [row["t"]-18]
    raise ValueError(name)


def prepared_rows(payload: dict, role: str, timeline: bool, horizons: list[int]) -> list[dict]:
    source=payload["timelines"][role] if timeline else payload["roles"][role]; rows=[]
    for i in range(len(source["t"])):
        row={"episode_id":source["episode_id"][i],"leakage_group":source["leakage_group"][i],"t":int(source["t"][i]),"p_history":source["base"][i,:,768].numpy(),
             "delta_x":float(source["force_delta_slots"][i,-1,0]),"delta_y":float(source["force_delta_slots"][i,-1,1]),"delta_z":float(source["force_delta_slots"][i,-1,2])}
        for j,h in enumerate(horizons): row[f"eligible_H{h}"]=bool(source["horizon_mask"][i,j]);row[f"target_H{h}"]=int(source["y"][i,j]) if row[f"eligible_H{h}"] else None
        row["common_population"]=bool(source["common_mask"][i]);rows.append(row)
    return rows


def evaluate(args) -> dict:
    cfg=json.loads(PROTOCOL.read_text()); prepared=torch.load(args.prepared,map_location="cpu",weights_only=False); horizons=[int(x) for x in prepared["horizons"]]
    if horizons != cfg["horizons"] or prepared.get("status")!="complete" or not prepared.get("formal"): raise ValueError("prepared protocol mismatch")
    prepared_sha=sha256(args.prepared); sources={str(args.prepared.resolve()):prepared_sha,str(PROTOCOL.resolve()):sha256(PROTOCOL),str(PROTOCOL_MD.resolve()):sha256(PROTOCOL_MD),str(Path(__file__).resolve()):sha256(Path(__file__).resolve())}
    for p in AMENDMENTS:sources[str(p.resolve())]=sha256(p)
    all_runs={}; reference_identity={}
    for group in GROUPS:
        for seed in SEEDS:
            folder=args.formal_root/f"future_{group}_{seed}"; summary,roles,run_sources=verify_run(folder,group,seed,prepared_sha,horizons);sources.update(run_sources)
            for role,rows in roles.items():
                identity=[(r["episode_id"],r["t"],r["onset"],r["current_slip_label"],tuple(r[f"target_H{h}"] for h in horizons),tuple(r[f"eligible_H{h}"] for h in horizons)) for r in rows]
                if role in reference_identity and identity!=reference_identity[role]:raise ValueError(f"cross-run population drift {group}/{seed}/{role}")
                reference_identity.setdefault(role,identity)
                for h in horizons:add_rules(rows,f"p_H{h}",f"H{h}_")
            all_runs[(group,seed)]={"summary":summary,"roles":roles}
    metrics=[]; selections=[]; transfers=[]; trials_by_file=defaultdict(list); tradeoff=[]; reliability=[]; selected_cache={};selected_thresholds={};platt_models=[]
    for (group,seed),run in all_runs.items():
        for h in horizons:
            for population in ("primary","per_h_max"):
                roles={role:role_rows(rows,h,population) for role,rows in run["roles"].items()}
                _CAL_CACHE.clear()
                selected,evidence=select_rule(roles["selection"],f"H{h}_","metric_mask")
                if population=="primary":selected_cache[(group,seed,h)]=selected
                for row in evidence:selections.append({"group":group,"seed":seed,"horizon":h,"population":population,"selected":row["rule"]==selected,**row})
                for rule in RULES:
                    score_key=f"H{h}_{rule}"
                    for op,kind,target in OPS:
                        threshold,status,_=select_threshold(roles["calibration"],score_key,"metric_mask",kind,target)
                        if population=="primary" and rule==selected:selected_thresholds[(group,seed,h,op)]=threshold
                        base={"method_type":"neural","group":group,"seed":seed,"horizon":h,"population":population,"rule":rule,"rule_selected":rule==selected,"operating_point":op,"threshold_status":status,"threshold":threshold}
                        if threshold is None:
                            metrics.append({**base,**{k:"" for k in ("n","positive","prevalence","average_precision","brier","tn","fp","fn","tp","frame_fpr","frame_recall","balanced_accuracy","macro_f1","never_alarm","trials","negative_window_trials","trial_false_alarm_rate","false_alarm_runs","false_alarm_starts","false_alarm_duration","mean_false_alarm_run_duration","max_false_alarm_duration","uncensored_events","left_censored_events","right_censored_or_no_observed_onset_trials","event_recall","active_overlap_recall","ongoing_from_before_positive_window_events","active_at_first_observable_row_start_censored_trials","mean_lead_frames","median_lead_frames","lead_ge_3_recall","lead_ge_5_recall","misses","late_alarms","never_alarm_trials")}});continue
                        frame=frame_confusion(roles["outer"],score_key,threshold,"metric_mask");event,records=event_metrics(roles["outer"],score_key,threshold,"metric_mask");metrics.append({**base,**frame,**event})
                        if rule==selected:
                            if population=="primary":
                                calframe=frame_confusion(roles["calibration"],score_key,threshold,"metric_mask");calevent=event_metrics(roles["calibration"],score_key,threshold,"metric_mask")[0]
                                transfers.append({"group":group,"seed":seed,"horizon":h,"rule":rule,"operating_point":op,"threshold":threshold,"calibration_frame_fpr":calframe["frame_fpr"],"outer_frame_fpr":frame["frame_fpr"],"calibration_frame_recall":calframe["frame_recall"],"outer_frame_recall":frame["frame_recall"],"calibration_trial_false_alarm_rate":calevent["trial_false_alarm_rate"],"outer_trial_false_alarm_rate":event["trial_false_alarm_rate"],"calibration_event_recall":calevent["event_recall"],"outer_event_recall":event["event_recall"],"calibration_mean_lead_frames":calevent["mean_lead_frames"],"outer_mean_lead_frames":event["mean_lead_frames"],"outer_constraint_failure":bool((kind=="frame_fpr" and frame["frame_fpr"]>float(target)+1e-15) or (kind=="trial_fa" and event["trial_false_alarm_rate"]>float(target)+1e-15) or (kind=="event_recall" and event["event_recall"]<float(target)-1e-15))})
                            key=(h,group,population)
                            for record in records:trials_by_file[key].append({"method_type":"neural","group":group,"seed":seed,"horizon":h,"population":population,"rule":rule,"operating_point":op,"threshold":threshold,**record})
                    if population=="primary":
                        thresholds=event_tradeoff_thresholds(roles["outer"],score_key,"metric_mask");compact=compact_event_curve(roles["outer"],score_key,"metric_mask",thresholds)
                        for event in compact:tradeoff.append({"group":group,"seed":seed,"horizon":h,"rule":rule,"scope":"outer_descriptive_only","threshold":event["threshold"],"event_recall":event["event_recall"],"trial_false_alarm_rate":event["trial_false_alarm_rate"],"mean_lead_frames":event["mean_lead_frames"]})
                if population=="primary":
                    selected_key=f"H{h}_{selected}"
                    platt=fit_monotone_platt(roles["calibration"],selected_key,"metric_mask")
                    platt_models.append({"group":group,"seed":seed,"horizon":h,"selected_rule":selected,**platt})
                    for role_name in ("calibration","outer"):
                        reliability.extend(reliability_rows(roles[role_name],selected_key,"metric_mask",group,seed,h,role_name,"selected_rule_raw"))
                    if platt["status"]=="available":
                        platt_key=f"H{h}_selected_platt"
                        for role_name in ("calibration","outer"):
                            for row in roles[role_name]:row[platt_key]=apply_platt(row[selected_key],platt)
                            reliability.extend(reliability_rows(roles[role_name],platt_key,"metric_mask",group,seed,h,role_name,"selected_rule_monotone_platt"))
                        _CAL_CACHE.clear()
                        for op,kind,target in OPS:
                            threshold,status,_=select_threshold(roles["calibration"],platt_key,"metric_mask",kind,target)
                            base={"method_type":"neural_platt","group":group,"seed":seed,"horizon":h,"population":"primary","rule":selected,"rule_selected":True,"operating_point":op,"threshold_status":status,"threshold":threshold}
                            if threshold is None: continue
                            metrics.append({**base,**frame_confusion(roles["outer"],platt_key,threshold,"metric_mask"),**event_metrics(roles["outer"],platt_key,threshold,"metric_mask")[0]})
                    # Fixed multiplication gate: raw only, no rule/model selection.
                    gate_key=f"H{h}_multiplication_gate_raw"
                    for role_name in ("calibration","outer"):
                        for row in roles[role_name]:row[gate_key]=row[f"p_H{h}"]*row["p_slip_current"]
                    _CAL_CACHE.clear()
                    for op,kind,target in OPS:
                        threshold,status,_=select_threshold(roles["calibration"],gate_key,"metric_mask",kind,target)
                        base={"method_type":"multiplication_gate_ablation","group":group,"seed":seed,"horizon":h,"population":"primary","rule":"raw","rule_selected":False,"operating_point":op,"threshold_status":status,"threshold":threshold}
                        if threshold is None: continue
                        metrics.append({**base,**frame_confusion(roles["outer"],gate_key,threshold,"metric_mask"),**event_metrics(roles["outer"],gate_key,threshold,"metric_mask")[0]})
    # Multi-horizon nesting diagnostic on the primary common population.
    consistency=[]
    for (group,seed),run in all_runs.items():
        for role in ("selection","calibration","outer"):
            rows=[r for r in run["roles"][role] if r["common_population"]]; violation=[not (r["p_H1"]<=r["p_H3"]<=r["p_H5"]) for r in rows]
            consistency.append({"group":group,"seed":seed,"role":role,"rows":len(rows),"violations":sum(violation),"violation_fraction":float(np.mean(violation))})
    # Fit-only simple baselines and constant prevalence, then use the same selection/calibration boundaries.
    fit=prepared_rows(prepared,"fit_train",False,horizons); timeline={role:prepared_rows(prepared,role,True,horizons) for role in ("selection","calibration","outer")}; baseline_models=[]
    for h in horizons:
        fit_h=[r for r in fit if r["common_population"]]
        for name in BASELINES:
            y=np.asarray([r[f"target_H{h}"] for r in fit_h],dtype=float)
            if name=="fit_prevalence": model={"prevalence":float(y.mean())}
            elif name in ("current_p_slip_raw","history9_p_slip_mean_raw"): model={"literal_raw":True}
            else:model=fit_logistic(np.asarray([baseline_features(r,name) for r in fit_h]),y)
            baseline_models.append({"name":name,"horizon":h,"parameters":{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in model.items()}})
            role_sets={}
            for role,base_rows in timeline.items():
                if name=="fit_prevalence":pred=np.full(len(base_rows),model["prevalence"])
                elif model.get("literal_raw"):pred=np.asarray([baseline_features(r,name)[0] for r in base_rows])
                else:pred=predict_logistic(model,np.asarray([baseline_features(r,name) for r in base_rows]))
                neural_ref=all_runs[("A_visual",SEEDS[0])]["roles"][role];rows=[]
                if len(base_rows)!=len(neural_ref):raise ValueError(f"baseline/neural length drift {role}")
                for b,n,p in zip(base_rows,neural_ref,pred):
                    if (b["episode_id"],b["t"])!=(n["episode_id"],n["t"]):raise ValueError(f"baseline/neural identity drift {role}")
                    row=dict(n);row[f"baseline_H{h}"]=float(p);rows.append(row)
                add_rules(rows,f"baseline_H{h}",f"baseline_H{h}_");role_sets[role]=role_rows(rows,h,"primary")
            _CAL_CACHE.clear();selected,evidence=select_rule(role_sets["selection"],f"baseline_H{h}_","metric_mask")
            for row in evidence:selections.append({"group":name,"seed":"","horizon":h,"population":"primary","selected":row["rule"]==selected,**row})
            score_key=f"baseline_H{h}_{selected}"
            for op,kind,target in OPS:
                threshold,status,_=select_threshold(role_sets["calibration"],score_key,"metric_mask",kind,target);base={"method_type":"fit_only_baseline","group":name,"seed":"","horizon":h,"population":"primary","rule":selected,"rule_selected":True,"operating_point":op,"threshold_status":status,"threshold":threshold}
                if threshold is None:metrics.append({**base,**{k:"" for k in ("n","positive","prevalence","average_precision","brier","tn","fp","fn","tp","frame_fpr","frame_recall","balanced_accuracy","macro_f1","never_alarm","trials","negative_window_trials","trial_false_alarm_rate","false_alarm_runs","false_alarm_starts","false_alarm_duration","mean_false_alarm_run_duration","max_false_alarm_duration","uncensored_events","left_censored_events","right_censored_or_no_observed_onset_trials","event_recall","active_overlap_recall","ongoing_from_before_positive_window_events","active_at_first_observable_row_start_censored_trials","mean_lead_frames","median_lead_frames","lead_ge_3_recall","lead_ge_5_recall","misses","late_alarms","never_alarm_trials")}});continue
                metrics.append({**base,**frame_confusion(role_sets["outer"],score_key,threshold,"metric_mask"),**event_metrics(role_sets["outer"],score_key,threshold,"metric_mask")[0]})
    # Frozen sensitivity interventions use the unperturbed selected rule and
    # unperturbed calibration threshold; they never recalibrate after intervention.
    backend_audit=json.loads(args.sensitivity_backend_audit.read_text())
    if backend_audit.get("status")!="pass":raise ValueError("sensitivity backend audit did not pass")
    sources[str(args.sensitivity_backend_audit.resolve())]=sha256(args.sensitivity_backend_audit)
    sensitivity_artifacts,sensitivity_sources=load_manifest_timelines(args.sensitivity_manifest,"round7_force_input_sensitivity_v1",("B_force","C_force_delta"),horizons);sources.update(sensitivity_sources);sensitivity_metrics=[]
    for intervention in ("zero_fit_mean","causal_lag1"):
        for group in ("B_force","C_force_delta"):
            for seed in SEEDS:
                role_sets={}
                for role in ("selection","calibration","outer"):
                    rows=sensitivity_artifacts[(intervention,group,seed,role)]
                    formal_rows=all_runs[(group,seed)]["roles"][role]
                    if [(r["episode_id"],r["t"]) for r in rows]!=[(r["episode_id"],r["t"]) for r in formal_rows]:raise ValueError(f"sensitivity identity drift {intervention}/{group}/{seed}/{role}")
                    for h in horizons:add_rules(rows,f"p_H{h}",f"H{h}_")
                    role_sets[role]=rows
                for h in horizons:
                    selected=selected_cache[(group,seed,h)];score_key=f"H{h}_{selected}";outer=role_rows(role_sets["outer"],h,"primary");formal_outer=role_rows(all_runs[(group,seed)]["roles"]["outer"],h,"primary")
                    mean_delta=float(np.mean([r[score_key]-b[score_key] for r,b in zip(outer,formal_outer) if r["metric_mask"]]))
                    for op,_,_ in OPS:
                        threshold=selected_thresholds[(group,seed,h,op)]
                        if threshold is None:continue
                        sensitivity_metrics.append({"intervention":intervention,"group":group,"seed":seed,"horizon":h,"population":"primary","rule":selected,"operating_point":op,"threshold_source":"unperturbed_calibration","threshold":threshold,"mean_score_delta_vs_unperturbed":mean_delta,
                                                    **frame_confusion(outer,score_key,threshold,"metric_mask"),**event_metrics(outer,score_key,threshold,"metric_mask")[0]})
    # Full-timeline Round-6 H1 reference is historical only: history4, legacy-seen selection.
    r6_artifacts,r6_sources=load_manifest_timelines(args.r6_reference_manifest,"round7_r6_h1_reference_manifest_v1",("A_visual","B_force","C_force_delta"),[1]);sources.update(r6_sources);r6_metrics=[];r6_selections=[]
    for group in ("A_visual","B_force","C_force_delta"):
        for seed in SEEDS:
            roles={role:r6_artifacts[("reference",group,seed,role)] for role in ("selection","calibration","outer")}
            for rows in roles.values():add_rules(rows,"p_H1","H1_")
            role_sets={role:role_rows(rows,1,"primary") for role,rows in roles.items()};_CAL_CACHE.clear();selected,evidence=select_rule(role_sets["selection"],"H1_","metric_mask")
            for row in evidence:r6_selections.append({"group":group,"seed":seed,"horizon":1,"population":"primary","selected":row["rule"]==selected,"reference_scope":"history4_H1_legacy_selection_seen",**row})
            for op,kind,target in OPS:
                threshold,status,_=select_threshold(role_sets["calibration"],f"H1_{selected}","metric_mask",kind,target)
                if threshold is None:continue
                r6_metrics.append({"method_type":"round6_full_timeline_historical_reference","group":group,"seed":seed,"horizon":1,"population":"primary","rule":selected,"operating_point":op,"threshold_status":status,"threshold":threshold,
                                   **frame_confusion(role_sets["outer"],f"H1_{selected}",threshold,"metric_mask"),**event_metrics(role_sets["outer"],f"H1_{selected}",threshold,"metric_mask")[0]})
    all_trial_rows=[row for rows in trials_by_file.values() for row in rows];paired_ci=paired_bootstrap_ci(all_trial_rows,horizons)
    aggregate=[]
    for group in GROUPS:
        for h in horizons:
            for op,_,_ in OPS:
                rows=[r for r in metrics if r["method_type"]=="neural" and r["group"]==group and r["horizon"]==h and r["population"]=="primary" and r["rule_selected"] and r["operating_point"]==op and r["threshold_status"]=="available"]
                if len(rows)!=len(SEEDS):continue
                aggregate.append({"group":group,"horizon":h,"operating_point":op,"seeds":";".join(str(r["seed"]) for r in rows),**{f"{metric}_{stat}":float(getattr(np,stat)([r[metric] for r in rows])) for metric in ("average_precision","frame_fpr","frame_recall","trial_false_alarm_rate","event_recall","active_overlap_recall","mean_lead_frames") for stat in ("mean","std")}})
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);failure_cases=render_failure_cases(out,all_runs,selected_cache,selected_thresholds,3 if 3 in horizons else horizons[0])
    atomic_csv(out/"metrics.csv",metrics);atomic_csv(out/"rule_selection.csv",selections);atomic_csv(out/"outer_event_tradeoff_descriptive.csv",tradeoff);atomic_csv(out/"multi_horizon_consistency.csv",consistency)
    atomic_csv(out/"reliability.csv",reliability);atomic_csv(out/"paired_bootstrap_ci.csv",paired_ci)
    atomic_csv(out/"failure_cases.csv",failure_cases)
    atomic_csv(out/"threshold_transfer.csv",transfers);atomic_csv(out/"all_seed_summary.csv",aggregate)
    atomic_csv(out/"sensitivity_metrics.csv",sensitivity_metrics);atomic_csv(out/"r6_full_timeline_reference_metrics.csv",r6_metrics);atomic_csv(out/"r6_full_timeline_reference_rule_selection.csv",r6_selections)
    for (h,group,pop),rows in trials_by_file.items():atomic_csv(out/"trials"/f"H{h}_{group}_{pop}.csv",rows)
    atomic_json(out/"baseline_models.json",{"status":"complete","fit_role":"fit_train","models":baseline_models})
    atomic_json(out/"platt_models.json",{"status":"complete","fit_role":"calibration","models":platt_models})
    # Compact figures; detailed values remain in CSV.
    figdir=out/"figures";figdir.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,h in zip(axes,horizons):
        for group in GROUPS:
            for i,seed in enumerate(SEEDS):
                x=[r for r in tradeoff if r["horizon"]==h and r["group"]==group and r["rule"]=="raw" and r["seed"]==seed];x=sorted(x,key=lambda r:r["trial_false_alarm_rate"])
                ax.plot([r["trial_false_alarm_rate"] for r in x],[r["event_recall"] for r in x],label=group if i==0 else None,alpha=.9 if i==0 else .35)
        ax.set(title=f"H{h} outer descriptive, all seeds",xlabel="trial-any false alarm",ylabel="new-start event recall",xlim=(0,1),ylim=(0,1));ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/"event_recall_vs_trial_false_alarm.png",dpi=180);plt.close(fig)
    key_rows=[r for r in aggregate if r["operating_point"] in ("trial_FA_0.10","event_recall_0.70")]
    lines=["# 第七轮事件告警评估", "", "本报告使用完整因果时间线。事件检出只认可滑移前正窗口内新开始的告警段；已经从更早时刻持续的告警仅作为 active-overlap 诊断，不能产生实际提前量。", "", "## 三随机种子汇总", "", "|组|H|工作点|AP|实际逐试次误报率|新起点事件召回|实际提前帧|", "|---|---:|---|---:|---:|---:|---:|"]
    for r in key_rows:lines.append(f"|{r['group']}|{r['horizon']}|{r['operating_point']}|{r['average_precision_mean']:.4f}±{r['average_precision_std']:.4f}|{r['trial_false_alarm_rate_mean']:.4f}±{r['trial_false_alarm_rate_std']:.4f}|{r['event_recall_mean']:.4f}±{r['event_recall_std']:.4f}|{r['mean_lead_frames_mean']:.3f}±{r['mean_lead_frames_std']:.3f}|")
    lines += ["", "## 解释边界", "", "- calibration 选择的阈值原样迁移到 outer；外层实际误报率和约束失败见 `threshold_transfer.csv`。", "- outer 仍是开发数据，不是独立盲测；数据集标签起点也不是独立物理滑移真值。", "- D 只是一个固定 PCA3 视觉差分对照；C 同时包含力值与力差分，因此 C-D 不能单独归因为力差分。", "- 多时间窗分类只提供有限未来风险证据，不能据此宣称已验证世界模型。", "- R6 结果是 history4/H1 且 selection 曾用于历史开发的参照，不属于与 R7 的严格同协议比较。", ""]
    (out/"SUMMARY_ZH.md").write_text("\n".join(lines))
    outputs={str(p.resolve()):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="summary.json"}
    summary={"format":"round7_event_evaluation_v1","status":"complete","expected_runs":12,"evaluated_runs":len(all_runs),"horizons":horizons,"groups":list(GROUPS),"seeds":list(SEEDS),"tests":{"required":"python -m unittest test_evaluation.py -q","expected_count":20},"metric_definitions":{"event_recall":"new causal alarm-run start in eligible positive window before onset","active_overlap_recall":"diagnostic alarm activity in positive window regardless of earlier start","mean_false_alarm_run_duration":"micro mean across all observed maximal false-alarm runs"},"source_hashes":sources,"output_hashes":outputs,"limitations":["outer is reused development data, not a blind test","dataset-label onset is not independent physical slip truth","D is one PCA3 visual-delta control","multi-window classification is not a validated world model","causal_lag1 shifts the C auxiliary validity flag and is not a pure value-only intervention","Round-6 reference is history4/H1 with legacy-seen selection and is not a head-to-head Round-7 comparator"]}
    atomic_json(out/"summary.json",summary);print(json.dumps({"status":"complete","runs":len(all_runs),"metrics":len(metrics),"tradeoff":len(tradeoff)},indent=2));return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--formal-root",type=Path,required=True);p.add_argument("--sensitivity-manifest",type=Path,required=True);p.add_argument("--sensitivity-backend-audit",type=Path,required=True);p.add_argument("--r6-reference-manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();evaluate(a)


if __name__=="__main__":main()
