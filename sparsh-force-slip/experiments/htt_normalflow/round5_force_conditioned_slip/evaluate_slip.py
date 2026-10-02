#!/usr/bin/env python3
"""Round-5 calibration-only slip evaluation with formal provenance checks."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "r4_alarm", HERE.parent / "round4_comprehensive/alarms/evaluate_alarms.py"
)
r4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r4)

CONTRACT_FORMAT = "round5_htt_mae_tokens_targets_v1"
CACHE_AUDIT_FORMAT = "round5_cache_full_hash_audit_v1"
CURRENT_SUMMARY_FORMAT = "round5_slip_training_summary_v1"
FORCE_TARGET_SEMANTICS = (
    "clip((6d_force-ref_force)[:3],-20,20) N; shear_x,shear_y,normal"
)
MODEL_IDS = ("mae-r3-b", "V", "F-old", "F-adapt")
BOOTSTRAP_REPETITIONS = 200


def partial_auc(rows):
    from sklearn.metrics import auc, roc_curve

    primary = [row for row in rows if row["stage"] in (0, 2)]
    labels = np.array([row["stage"] == 2 for row in primary])
    scores = np.array([row["score"] for row in primary])
    if len(np.unique(labels)) != 2:
        raise ValueError("pAUC requires both primary classes")
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    stop = np.searchsorted(fpr, 0.1, side="right")
    x = np.r_[fpr[:stop], 0.1]
    values = np.r_[tpr[:stop], np.interp(0.1, fpr, tpr)]
    return float(auc(x, values) / 0.1)


def bootstrap_partial_auc(rows, groups, seed, repetitions=BOOTSTRAP_REPETITIONS):
    """Frame-weighted pAUC interval from complete leakage-group resampling."""
    by_group = {}
    for row in rows:
        by_group.setdefault(groups[row["episode"]], []).append(row)
    names = sorted(by_group)
    if not names:
        raise ValueError("pAUC bootstrap requires leakage groups")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(names, size=len(names), replace=True)
        replicate = []
        for draw, name in enumerate(sampled):
            replicate.extend({**row, "_bootstrap_draw": draw} for row in by_group[name])
        try:
            values.append(partial_auc(replicate))
        except ValueError:
            continue
    if not values:
        raise ValueError("No valid two-class pAUC bootstrap replicate")
    return {
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "bootstrap_unit": "complete schema-2 leakage_group",
        "estimand": "frame-weighted normalized TPR AUC over FPR [0,0.1]",
        "groups": len(names),
        "replicates_requested": repetitions,
        "replicates_valid_two_class": len(values),
    }


def _resolved(path):
    return str(Path(path).resolve())


def _prediction_record(summary, role):
    record = summary.get("predictions", {}).get(role)
    if isinstance(record, dict):
        return record.get("path"), record.get("sha256")
    return record, summary.get("prediction_sha256", {}).get(role)


def validate_training_chain(summary_path, calibration, validation, fold, seed,
                            model_id, historical_baseline):
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or summary.get("formal") is False:
        raise ValueError("training summary is not a completed formal run")
    if summary.get("fold") != fold or summary.get("seed") != seed:
        raise ValueError("training summary fold/seed mismatch")
    if historical_baseline:
        if model_id != "mae-r3-b":
            raise ValueError("historical baseline requires model-id mae-r3-b")
        config = summary.get("config", {})
        if summary.get("init") != "fresh" or config.get("init") != "fresh":
            raise ValueError("historical baseline is not the fresh Round-3 B head")
        if config.get("smoke") is not False or summary.get("role_isolation", {}).get("pass") is not True:
            raise ValueError("historical baseline is smoke or failed role isolation")
        source_kind = "round3_mae_fresh_b_historical_baseline"
    else:
        if model_id == "mae-r3-b":
            raise ValueError("mae-r3-b must be declared with --historical-baseline")
        if summary.get("format") != CURRENT_SUMMARY_FORMAT:
            raise ValueError("unexpected current training summary format")
        if summary.get("variant") != model_id or summary.get("smoke") is not False:
            raise ValueError("current training summary identity is not formal")
        source_kind = "round5_force_conditioned_slip"
    for role, actual in (("calibration", calibration), ("validation", validation)):
        recorded_path, recorded_sha = _prediction_record(summary, role)
        if not recorded_path or _resolved(recorded_path) != _resolved(actual):
            raise ValueError(f"training summary {role} prediction path mismatch")
        if not recorded_sha or r4.sha256(actual) != recorded_sha:
            raise ValueError(f"training summary {role} prediction hash mismatch")
    checkpoint = Path(summary.get("best_checkpoint", ""))
    checkpoint_sha = summary.get("best_checkpoint_sha256")
    if not checkpoint.is_file() or not checkpoint_sha or r4.sha256(checkpoint) != checkpoint_sha:
        raise ValueError("training summary best checkpoint chain mismatch")
    return {
        "source_kind": source_kind,
        "training_summary_path": str(summary_path),
        "training_summary_sha256": r4.sha256(summary_path),
        "best_checkpoint": str(checkpoint.resolve()),
        "best_checkpoint_sha256": checkpoint_sha,
    }


def load_authoritative_labels(contract_path, cache_audit_path, manifest, manifest_path, fold):
    contract_path = contract_path.resolve()
    cache_audit_path = cache_audit_path.resolve()
    contract = json.loads(contract_path.read_text())
    audit = json.loads(cache_audit_path.read_text())
    contract_sha = r4.sha256(contract_path)
    if contract.get("status") != "complete" or contract.get("format") != CONTRACT_FORMAT:
        raise ValueError("incomplete or incompatible Round-5 contract")
    if contract.get("force_target_semantics") != FORCE_TARGET_SEMANTICS:
        raise ValueError("contract force target semantics mismatch")
    if contract.get("split_manifest_sha256") != r4.sha256(manifest_path):
        raise ValueError("contract and schema-2 split manifest mismatch")
    if audit.get("status") != "pass" or audit.get("format") != CACHE_AUDIT_FORMAT:
        raise ValueError("full cache hash audit has not passed")
    if audit.get("cache_manifest_sha256") != contract_sha:
        raise ValueError("cache audit does not certify this contract")
    if audit.get("entries") != len(contract.get("entries", [])) or audit.get("files_verified") != 2 * len(contract.get("entries", [])):
        raise ValueError("cache audit coverage is incomplete")
    episodes = {row["id"]: row for row in manifest["episodes"]}
    expected = {
        episode_id: role
        for role in ("calibration", "validation")
        for episode_id in manifest["splits"][fold][role]
        if episodes[episode_id]["task"] == "slip"
    }
    entries = {row["episode_id"]: row for row in contract["entries"] if row["task"] == "slip"}
    labels = {}
    label_hashes = {}
    for episode_id, role in expected.items():
        entry = entries.get(episode_id)
        if entry is None or entry.get("roles_by_fold", {}).get(fold) != role:
            raise ValueError(f"contract role mismatch: {episode_id}")
        label_path = Path(entry["label_path"])
        if r4.sha256(label_path) != entry.get("label_sha256"):
            raise ValueError(f"authoritative label hash mismatch: {episode_id}")
        array = np.load(label_path, allow_pickle=False)
        if array.shape != (entry["frames"],) or entry["frames"] != episodes[episode_id]["frames"]:
            raise ValueError(f"authoritative label shape mismatch: {episode_id}")
        if not np.isin(array, (0, 1, 2)).all():
            raise ValueError(f"invalid authoritative labels: {episode_id}")
        labels[episode_id] = array
        label_hashes[episode_id] = entry["label_sha256"]
    if set(labels) != set(expected):
        raise ValueError("authoritative label episode set mismatch")
    return labels, {
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha,
        "cache_audit_path": str(cache_audit_path),
        "cache_audit_sha256": r4.sha256(cache_audit_path),
        "authoritative_label_hashes": label_hashes,
    }


def read_rows(path, manifest, fold, role, authoritative_labels):
    allowed = set(manifest["splits"][fold][role])
    episodes = {row["id"]: row for row in manifest["episodes"]}
    result, seen = [], set()
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            episode_id = row.get("episode_id", row.get("episode"))
            frame = int(row.get("t", row.get("frame", -1)))
            if episode_id not in allowed or episodes[episode_id]["task"] != "slip":
                raise ValueError("forbidden role/task: " + str(episode_id))
            if row.get("role") not in (None, "", role):
                raise ValueError("prediction role column mismatch")
            stage = int(row["stage"])
            score = float(row.get("p_slip", row.get("probability", row.get("score", "nan"))))
            if stage not in (0, 1, 2) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid prediction")
            if episode_id not in authoritative_labels or frame < 0 or frame >= len(authoritative_labels[episode_id]):
                raise ValueError("prediction outside authoritative labels")
            if stage != int(authoritative_labels[episode_id][frame]):
                raise ValueError(f"prediction stage differs from authoritative label: {episode_id}:{frame}")
            if (episode_id, frame) in seen:
                raise ValueError("duplicate frame")
            seen.add((episode_id, frame))
            if frame >= 10:
                result.append({"episode": episode_id, "t": frame, "stage": stage, "score": score})
    result.sort(key=lambda row: (row["episode"], row["t"]))
    if not result:
        raise ValueError("empty evaluation")
    for episode_id, rows in r4.split_episodes(result).items():
        expected = list(range(10, episodes[episode_id]["frames"]))
        if [row["t"] for row in rows] != expected:
            raise ValueError("incomplete strict sequence " + episode_id)
    expected_episodes = {
        episode_id for episode_id in allowed
        if episodes[episode_id]["task"] == "slip" and episodes[episode_id]["frames"] > 10
    }
    if set(r4.split_episodes(result)) != expected_episodes:
        raise ValueError("missing role episodes")
    return result


def metrics(rows, threshold, confirmation_k, release_ratio, groups):
    result, trials = r4.sequential_metrics(rows, threshold, confirmation_k, release_ratio, groups)
    result["segments_beginning_at_evaluation_boundary"] = result.pop("segments_beginning_at_frame_zero")
    result["partial_tpr_auc_0_0p1"] = partial_auc(rows)
    result["evaluation_start_frame"] = 10
    for row in trials:
        row["evaluation_start_frame"] = 10
    return result, trials


def evaluate(calibration, validation, groups, seed):
    if {groups[row["episode"]] for row in calibration} & {groups[row["episode"]] for row in validation}:
        raise ValueError("leakage group overlap")
    operations = {"fixed_0.5": {"threshold": 0.5}, "max_ba": r4.choose_max_ba(calibration)}
    for budget in r4.BUDGETS:
        operations[f"fpr_{budget:.2f}"] = r4.choose_budget(calibration, budget)
    result, trial_rows = {}, []
    for name, chosen in operations.items():
        threshold = chosen["threshold"]
        item = {"selection": "fixed" if name == "fixed_0.5" else "calibration_only", "raw": {}}
        for role, rows in (("calibration", calibration), ("validation", validation)):
            values, trials = metrics(rows, threshold, 1, 1.0, groups)
            item["raw"][role] = values
            if role == "validation":
                item["raw"]["validation_trial_bootstrap_95ci"] = r4.bootstrap_trials(
                    trials, seed + sum(map(ord, name))
                )
                item["raw"]["validation_partial_tpr_auc_0_0p1_trial_bootstrap_95ci"] = bootstrap_partial_auc(
                    rows, groups, seed + 2000 + sum(map(ord, name))
                )
            trial_rows.extend(dict(operating_point=name, mode="raw", role=role, **row) for row in trials)
        if name.startswith("fpr_"):
            rule, _ = r4.choose_rule(calibration, threshold, float(name.split("_")[1]), groups)
            item["sequential"] = {}
            for role, rows in (("calibration", calibration), ("validation", validation)):
                values, trials = metrics(rows, rule["threshold"], rule["confirmation_k"], rule["release_ratio"], groups)
                item["sequential"][role] = values
                if role == "validation":
                    item["sequential"]["validation_trial_bootstrap_95ci"] = r4.bootstrap_trials(
                        trials, seed + 1000 + sum(map(ord, name))
                    )
                    item["sequential"]["validation_partial_tpr_auc_0_0p1_trial_bootstrap_95ci"] = bootstrap_partial_auc(
                        rows, groups, seed + 3000 + sum(map(ord, name))
                    )
                trial_rows.extend(dict(operating_point=name, mode="sequential", role=role, **row) for row in trials)
        result[name] = item
    return result, trial_rows


def write_outputs(output, payload, trials):
    """Commit trials first; metrics status=complete is the final atomic marker."""
    if not trials:
        raise ValueError("cannot commit evaluation without trial rows")
    output.mkdir(parents=True, exist_ok=True)
    running = {**payload, "status": "running", "formal": False}
    r4.atomic_json(output / "metrics.json", running)
    trial_path = output / "trials.csv"
    temporary = trial_path.with_name(trial_path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
        os.replace(temporary, trial_path)
    finally:
        temporary.unlink(missing_ok=True)
    committed = {
        **payload,
        "status": "complete",
        "formal": True,
        "artifacts": {
            "trials_csv": str(trial_path.resolve()),
            "trials_csv_sha256": r4.sha256(trial_path),
            "trial_rows": len(trials),
        },
    }
    r4.atomic_json(output / "metrics.json", committed)
    return committed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    parser.add_argument("--historical-baseline", action="store_true")
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2 or args.fold not in manifest.get("splits", {}):
        raise ValueError("incompatible schema-2 split manifest/fold")
    labels, contract_provenance = load_authoritative_labels(
        args.contract, args.cache_audit, manifest, manifest_path, args.fold
    )
    training_provenance = validate_training_chain(
        args.training_summary, args.calibration, args.validation, args.fold, args.seed,
        args.model_id, args.historical_baseline,
    )
    groups = {row["id"]: row["leakage_group"] for row in manifest["episodes"]}
    calibration = read_rows(args.calibration, manifest, args.fold, "calibration", labels)
    validation = read_rows(args.validation, manifest, args.fold, "validation", labels)
    operations, trials = evaluate(calibration, validation, groups, args.seed)
    provenance_paths = [
        args.calibration.resolve(), args.validation.resolve(), manifest_path,
        Path(__file__).resolve(), Path(r4.__file__).resolve(), HERE / "PROTOCOL.md",
    ]
    payload = {
        "model_id": args.model_id,
        "historical_baseline": args.historical_baseline,
        "fold": args.fold,
        "seed": args.seed,
        "scope": "overlapping development folds; calibration and validation roles are disjoint; no physical deployment claim",
        "protocol": "strict t>=10; sequence cold-start at evaluation boundary; same filter for historical MAE-B",
        "provenance": {
            "files": {str(path): r4.sha256(path) for path in provenance_paths},
            **contract_provenance,
            **training_provenance,
        },
        "operating_points": operations,
    }
    committed = write_outputs(args.output.resolve(), payload, trials)
    print(json.dumps({"status": committed["status"], "formal": True, "trials": len(trials)}))


if __name__ == "__main__":
    main()
