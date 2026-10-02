#!/usr/bin/env python3
"""Raw common-population metrics and paired leakage-group bootstrap for R7."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


GROUPS = ("A_visual", "B_force", "C_force_delta", "D_visual_delta")
SEEDS = (20260914, 20260915, 20260916)
HORIZONS = (1, 3, 5)
ROLES = ("selection", "calibration", "outer")
BOOTSTRAP_ROLE = "outer"
BOOTSTRAP_REPETITIONS = 200
BOOTSTRAP_SEED = 20260915


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    """Uninterpolated AP with tied scores included at one threshold."""
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if target.ndim != 1 or score.shape != target.shape or target.size == 0:
        raise ValueError("target and score must be non-empty equal-length vectors")
    if not np.isfinite(score).all() or not np.isin(target, (0, 1)).all():
        raise ValueError("non-finite score or non-binary target")
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = target[order]
    s = score[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), target.size - 1]
    true_positives = np.cumsum(y, dtype=np.int64)[ends]
    predicted = ends + 1
    recall = true_positives / positives
    precision = true_positives / predicted
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    if target.size == 0 or target.shape != score.shape:
        raise ValueError("invalid metric population")
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("probability outside [0,1]")
    return {
        "n": int(target.size),
        "positives": int(target.sum()),
        "prevalence": float(target.mean()),
        "average_precision": average_precision(target, score),
        "brier": float(np.mean((score - target) ** 2)),
    }


def percentile_interval(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return {
        "valid_replicates": int(finite.size),
        "lower_2p5": float(np.percentile(finite, 2.5)) if finite.size else None,
        "upper_97p5": float(np.percentile(finite, 97.5)) if finite.size else None,
    }


def load_run(path: Path) -> tuple[list[tuple[str, str, int]], dict[int, dict[str, np.ndarray]]]:
    identities: list[tuple[str, str, int]] = []
    by_horizon: dict[int, dict[str, list]] = {
        horizon: {"identity": [], "leakage_group": [], "target": [], "score": []}
        for horizon in HORIZONS
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"episode_id", "leakage_group", "t", "common_population"}
        for horizon in HORIZONS:
            required |= {
                f"target_future_H{horizon}",
                f"eligible_H{horizon}",
                f"common_eligible_H{horizon}",
                f"p_future_H{horizon}_raw",
            }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"missing columns in {path}: {sorted(required - set(reader.fieldnames or []))}")
        previous = None
        for row in reader:
            identity = (row["episode_id"], row["leakage_group"], int(row["t"]))
            if identity == previous:
                raise ValueError(f"adjacent duplicate identity in {path}: {identity}")
            previous = identity
            identities.append(identity)
            common_population = as_bool(row["common_population"])
            for horizon in HORIZONS:
                common = as_bool(row[f"common_eligible_H{horizon}"])
                eligible = as_bool(row[f"eligible_H{horizon}"])
                if common and (not eligible or not common_population):
                    raise ValueError(f"invalid common mask in {path}, H{horizon}, {identity}")
                if not common:
                    continue
                target = int(row[f"target_future_H{horizon}"])
                score = float(row[f"p_future_H{horizon}_raw"])
                if target not in (0, 1):
                    raise ValueError(f"non-binary target in {path}, H{horizon}, {identity}")
                by_horizon[horizon]["identity"].append(identity)
                by_horizon[horizon]["leakage_group"].append(row["leakage_group"])
                by_horizon[horizon]["target"].append(target)
                by_horizon[horizon]["score"].append(score)
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate identity in {path}")
    converted = {}
    for horizon, values in by_horizon.items():
        converted[horizon] = {
            "identity": tuple(values["identity"]),
            "leakage_group": np.asarray(values["leakage_group"], dtype=object),
            "target": np.asarray(values["target"], dtype=np.int8),
            "score": np.asarray(values["score"], dtype=np.float64),
        }
    return identities, converted


def group_indices(groups: np.ndarray) -> tuple[list[str], dict[str, np.ndarray]]:
    names = sorted(set(groups.tolist()))
    return names, {name: np.flatnonzero(groups == name) for name in names}


def sampled_indices(draw: np.ndarray, names: list[str], indices: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([indices[names[index]] for index in draw])


def metric_difference(
    data: dict,
    horizon: int,
    sampled: np.ndarray | None,
    candidate: str,
    baseline: str,
    metric: str,
) -> float:
    differences = []
    for seed in SEEDS:
        left = data[(candidate, seed, horizon)]
        right = data[(baseline, seed, horizon)]
        left_metric = metrics(left["target"] if sampled is None else left["target"][sampled], left["score"] if sampled is None else left["score"][sampled])[metric]
        right_metric = metrics(right["target"] if sampled is None else right["target"][sampled], right["score"] if sampled is None else right["score"][sampled])[metric]
        differences.append(float(left_metric) - float(right_metric))
    return float(np.mean(differences))


def validate_inventory(root: Path) -> tuple[list[dict], dict[str, str]]:
    inventory_path = root / "RUN_INVENTORY.json"
    audit_path = root / "TRAINING_AUDIT.json"
    inventory = json.loads(inventory_path.read_text())
    audit = json.loads(audit_path.read_text())
    if inventory.get("status") != "all_formal_training_complete" or inventory.get("training_runs_executed") != 12:
        raise ValueError("formal inventory is not complete with exactly 12 runs")
    if audit.get("status") != "pass" or audit.get("verified_runs") != 12 or audit.get("failures"):
        raise ValueError("formal training audit is not a clean 12-run pass")
    runs = inventory["runs"]
    actual = {(run["group"], int(run["seed"])) for run in runs}
    expected = {(group, seed) for group in GROUPS for seed in SEEDS}
    if actual != expected or len(runs) != len(actual) or any(run.get("status") != "complete" for run in runs):
        raise ValueError("formal run identity is not the exact A/B/C/D x three-seed grid")
    return runs, {
        str(inventory_path): sha256(inventory_path),
        str(audit_path): sha256(audit_path),
    }


def validate_protocols() -> dict[str, str]:
    analysis_protocol_path = Path(__file__).with_name("protocol.json")
    evaluation_protocol_path = Path(__file__).resolve().parent.parent / "evaluation" / "protocol.json"
    analysis_protocol = json.loads(analysis_protocol_path.read_text())
    evaluation_protocol = json.loads(evaluation_protocol_path.read_text())
    expected = {
        "roles_for_per_run_metrics": list(ROLES),
        "bootstrap_role": BOOTSTRAP_ROLE,
        "horizons": list(HORIZONS),
        "groups": list(GROUPS),
        "seeds": list(SEEDS),
    }
    for key, value in expected.items():
        if analysis_protocol.get(key) != value:
            raise ValueError(f"analysis protocol mismatch: {key}")
    bootstrap = analysis_protocol.get("bootstrap", {})
    if bootstrap.get("unit") != "whole_leakage_group" or bootstrap.get("repetitions") != BOOTSTRAP_REPETITIONS or bootstrap.get("seed") != BOOTSTRAP_SEED:
        raise ValueError("analysis bootstrap protocol mismatch")
    upstream_bootstrap = evaluation_protocol.get("bootstrap", {})
    if upstream_bootstrap.get("unit") != "leakage_group" or upstream_bootstrap.get("repetitions") != BOOTSTRAP_REPETITIONS:
        raise ValueError("frozen evaluation protocol does not authorize the 200-group bootstrap")
    return {
        str(analysis_protocol_path): sha256(analysis_protocol_path),
        str(evaluation_protocol_path): sha256(evaluation_protocol_path),
    }


def run_analysis(root: Path, output_dir: Path) -> dict:
    runs, source_hashes = validate_inventory(root)
    source_hashes.update(validate_protocols())
    datasets: dict[tuple[str, int, str, int], dict[str, np.ndarray]] = {}
    reference_full: dict[str, tuple[tuple[str, str, int], ...]] = {}
    reference_common: dict[tuple[str, int], tuple[tuple[str, str, int], ...]] = {}
    reference_target: dict[tuple[str, int], np.ndarray] = {}
    per_run_rows: list[dict] = []

    for run in sorted(runs, key=lambda item: (item["group"], int(item["seed"]))):
        group, seed = run["group"], int(run["seed"])
        summary_path = Path(run["summary"])
        expected_summary_path = Path(run["output"]) / "summary.json"
        if summary_path != expected_summary_path or not summary_path.is_file():
            raise ValueError(f"inventory summary path mismatch for {group}/{seed}")
        summary = json.loads(summary_path.read_text())
        if not summary.get("formal") or summary.get("smoke") or summary.get("status") != "complete" or summary.get("group") != group or int(summary.get("seed")) != seed:
            raise ValueError(f"nonformal or mismatched summary for {group}/{seed}")
        source_hashes[str(summary_path)] = sha256(summary_path)
        for role in ROLES:
            path = Path(run["output"]) / f"predictions_timeline_{role}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            artifact = summary["artifacts"]["predictions"][role]["timeline"]
            actual_hash = sha256(path)
            if Path(artifact["path"]) != path or artifact["sha256"] != actual_hash:
                raise ValueError(f"summary-bound prediction identity mismatch for {group}/{seed}/{role}")
            source_hashes[str(path)] = actual_hash
            full_identity, horizon_data = load_run(path)
            if int(artifact["rows"]) != len(full_identity):
                raise ValueError(f"summary-bound row count mismatch for {group}/{seed}/{role}")
            full_tuple = tuple(full_identity)
            if role in reference_full and full_tuple != reference_full[role]:
                raise ValueError(f"full timeline identity mismatch for {group}/{seed}/{role}")
            reference_full.setdefault(role, full_tuple)
            for horizon in HORIZONS:
                values = horizon_data[horizon]
                key = (role, horizon)
                if key in reference_common and values["identity"] != reference_common[key]:
                    raise ValueError(f"common identity mismatch for {group}/{seed}/{role}/H{horizon}")
                if key in reference_target and not np.array_equal(values["target"], reference_target[key]):
                    raise ValueError(f"common target mismatch for {group}/{seed}/{role}/H{horizon}")
                reference_common.setdefault(key, values["identity"])
                reference_target.setdefault(key, values["target"])
                datasets[(group, seed, role, horizon)] = values
                result = metrics(values["target"], values["score"])
                per_run_rows.append({
                    "role": role,
                    "group": group,
                    "seed": seed,
                    "horizon": horizon,
                    "leakage_groups": len(set(values["leakage_group"].tolist())),
                    **result,
                })

    # A common population is required for paired comparisons, including an exact
    # common group universe across all horizons.  Scores differ; identities do not.
    outer_identities = [reference_common[(BOOTSTRAP_ROLE, horizon)] for horizon in HORIZONS]
    if any(identities != outer_identities[0] for identities in outer_identities[1:]):
        raise ValueError("outer exact-common identities differ across horizons")
    bootstrap_data = {
        (group, seed, horizon): datasets[(group, seed, BOOTSTRAP_ROLE, horizon)]
        for group in GROUPS for seed in SEEDS for horizon in HORIZONS
    }
    first = bootstrap_data[(GROUPS[0], SEEDS[0], HORIZONS[0])]
    names, indices = group_indices(first["leakage_group"])
    if len(names) < 2:
        raise ValueError("fewer than two outer leakage groups")
    for values in bootstrap_data.values():
        other_names, _ = group_indices(values["leakage_group"])
        if other_names != names or not np.array_equal(values["leakage_group"], first["leakage_group"]):
            raise ValueError("outer leakage-group identity mismatch")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(names), size=len(names)) for _ in range(BOOTSTRAP_REPETITIONS)]
    comparisons = []
    for horizon in HORIZONS:
        for metric in ("average_precision", "brier"):
            replicates: dict[str, list[float]] = defaultdict(list)
            point = {}
            for baseline in ("A_visual", "B_force", "D_visual_delta"):
                name = f"C_force_delta-minus-{baseline}"
                point[name] = metric_difference(bootstrap_data, horizon, None, "C_force_delta", baseline, metric)
            point["difference_in_differences_(C-B)-(D-A)"] = (
                point["C_force_delta-minus-B_force"]
                - metric_difference(bootstrap_data, horizon, None, "D_visual_delta", "A_visual", metric)
            )
            for draw in draws:
                sampled = sampled_indices(draw, names, indices)
                for baseline in ("A_visual", "B_force", "D_visual_delta"):
                    name = f"C_force_delta-minus-{baseline}"
                    replicates[name].append(metric_difference(bootstrap_data, horizon, sampled, "C_force_delta", baseline, metric))
                replicates["difference_in_differences_(C-B)-(D-A)"].append(
                    replicates["C_force_delta-minus-B_force"][-1]
                    - metric_difference(bootstrap_data, horizon, sampled, "D_visual_delta", "A_visual", metric)
                )
            for comparison, estimate in point.items():
                comparisons.append({
                    "role": BOOTSTRAP_ROLE,
                    "horizon": horizon,
                    "metric": metric,
                    "comparison": comparison,
                    "estimate_mean_of_three_paired_seed_differences": estimate,
                    "benefit_direction": "positive" if metric == "average_precision" else "negative",
                    **percentile_interval(replicates[comparison]),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = output_dir / "AP_BRIER_PER_RUN.csv"
    fields = ["role", "group", "seed", "horizon", "leakage_groups", "n", "positives", "prevalence", "average_precision", "brier"]
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(per_run_rows)
    atomic_text(per_run_path, stream.getvalue())

    result = {
        "status": "complete",
        "scope": {
            "per_run_roles": list(ROLES),
            "bootstrap_role": BOOTSTRAP_ROLE,
            "population": "exact common_eligible_H1/H3/H5 from full timeline CSVs",
            "metrics": ["raw_average_precision", "raw_brier", "prevalence"],
            "no_thresholds": True,
            "no_model_or_rule_selection": True,
            "outer_role_limit": "source validation outer role only; no test or real-object data",
            "inference_limit": "percentile intervals describe paired leakage-group resampling on this development dataset; seeds are averaged within each shared draw and are not independent replicates",
        },
        "bootstrap": {
            "unit": "whole_leakage_group",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "rng": "numpy.random.default_rng(PCG64)",
            "seed": BOOTSTRAP_SEED,
            "shared_draw": "same sampled leakage-group draw across A/B/C/D, all three seeds, all horizons and both metrics for each replicate index",
            "outer_leakage_groups": names,
            "comparisons": comparisons,
        },
        "identity_checks": {
            "formal_grid": "A/B/C/D x seeds 20260914/20260915/20260916",
            "full_timeline_identity_equal_across_all_runs_per_role": True,
            "common_identity_and_target_equal_across_all_runs_per_role_and_horizon": True,
            "outer_common_identity_equal_across_horizons": True,
        },
        "counts": {"per_run_metric_rows": len(per_run_rows), "bootstrap_comparisons": len(comparisons)},
        "source_sha256": dict(sorted(source_hashes.items())),
        "code_sha256": sha256(Path(__file__)),
        "outputs": {str(per_run_path): sha256(per_run_path)},
    }
    result_path = output_dir / "AP_BOOTSTRAP.json"
    atomic_text(result_path, json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Round-7 server result root")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(args.root.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": result["status"], **result["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
