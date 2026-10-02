#!/usr/bin/env python3
"""Build Round-8 equal-history signed-force and event-time prepared data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
NUMERIC_PROTOCOL = HERE.parent / "NUMERIC_PROTOCOL.json"
ROLES = ("fit_train", "selection", "calibration", "outer")
PREDICTION_ROLES = ("selection", "calibration", "outer")
HORIZONS = (1, 3, 5)
EVENT_STEPS = tuple(range(1, 6))
HISTORY = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_group(metadata: dict) -> str:
    return f"source/{metadata['dataset']}/{metadata['trajectory']}"


def finite_tensor(value: Any, name: str, ndim: int | None = None) -> torch.Tensor:
    if not torch.is_tensor(value) or value.requires_grad:
        raise ValueError(f"{name} is not a frozen tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} has wrong rank")
    if not bool(torch.isfinite(value.float()).all()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def population_stats(values: torch.Tensor) -> dict:
    if values.ndim != 2 or not len(values):
        raise ValueError("normalizer population must be non-empty rank two")
    mean = values.double().mean(0).float()
    variance = values.double().var(0, unbiased=False).float()
    std = variance.sqrt().clamp_min(1e-6)
    return {"mean": mean, "std": std, "variance": variance}


def load_source_cache(path: Path, split: str) -> tuple[dict, dict]:
    derived = torch.load(path, map_location="cpu", weights_only=False)
    if derived.get("schema") != "round5_source_pred_delta_v1" or derived.get("split") != split:
        raise ValueError(f"invalid derived source cache: {path}")
    if derived.get("horizons") != list(HORIZONS):
        raise ValueError("derived cache horizon drift")
    for key in ("z", "p_slip", "force_features", "predicted_delta_force_xyz", "future_slip"):
        finite_tensor(derived.get(key), key)
    parent_path = Path(derived["source_path"])
    if sha256(parent_path) != derived["source_sha256"]:
        raise ValueError("parent cache hash drift")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("split") not in {split, "validation" if split == "val" else split}:
        raise ValueError("parent cache split drift")
    force = finite_tensor(parent.get("force_pred_n"), "force_pred_n", 2).float()
    if force.shape[1] != 3 or len(force) != len(parent.get("metadata", [])):
        raise ValueError("invalid signed predicted-force parent field")
    return derived, parent


def build_force_lookup(caches: list[tuple[dict, dict]]) -> tuple[dict[tuple[str, int], torch.Tensor], dict[tuple[str, int], int], dict]:
    lookup: dict[tuple[str, int], torch.Tensor] = {}
    label_lookup: dict[tuple[str, int], int] = {}
    checked_deltas = 0
    max_delta_error = 0.0
    parents = []
    for derived, parent in caches:
        parent_index: dict[tuple[str, int], int] = {}
        current_slip = finite_tensor(parent.get("current_slip"), "parent current_slip", 1).long()
        if len(current_slip) != len(parent["metadata"]) or bool(((current_slip != 0) & (current_slip != 1)).any()):
            raise ValueError("invalid parent current-slip observation field")
        for index, metadata in enumerate(parent["metadata"]):
            key = (source_group(metadata), int(metadata["sample"]))
            if key in lookup or key in parent_index:
                raise ValueError(f"duplicate parent force row {key}")
            parent_index[key] = index
            lookup[key] = parent["force_pred_n"][index].detach().float()
            label_lookup[key] = int(current_slip[index])
        for index, metadata in enumerate(derived["metadata"]):
            key = (source_group(metadata), int(metadata["sample"]))
            lag_key = (key[0], key[1] - 5)
            if key not in parent_index or lag_key not in parent_index:
                raise ValueError(f"derived force row is absent from parent {key}")
            expected = lookup[key] - lookup[lag_key]
            actual = derived["predicted_delta_force_xyz"][index].float()
            error = float((expected - actual).abs().max())
            max_delta_error = max(max_delta_error, error)
            checked_deltas += 1
            if error > 1e-7:
                raise ValueError(f"signed delta mismatch at {key}: {error}")
        parents.append({
            "path": str(Path(derived["source_path"]).resolve()),
            "sha256": derived["source_sha256"],
            "split": derived["split"],
            "signed_force_field": "force_pred_n",
            "checkpoint": parent.get("checkpoint"),
            "encoder": parent.get("encoder"),
            "aux_schema": parent.get("aux_schema"),
            "rows": len(parent["metadata"]),
        })
    return lookup, label_lookup, {
        "parents": parents,
        "signed_force_source": "frozen parent cache force_pred_n",
        "checked_derived_lag5_rows": checked_deltas,
        "maximum_lag5_reconstruction_error": max_delta_error,
        "label_observation_source": "same hashed frozen parent cache current_slip",
        "label_observation_rows": len(label_lookup),
    }


def event_arrays(
    episode_ids: list[str],
    times: torch.Tensor,
    current_slip: torch.Tensor,
    onsets: list[int | None],
    right_censored: torch.Tensor,
    force_lookup: dict[tuple[str, int], torch.Tensor],
    label_lookup: dict[tuple[str, int], int],
) -> dict:
    count = len(times)
    occurrence = torch.zeros((count, len(EVENT_STEPS)), dtype=torch.bool)
    observed = torch.zeros_like(occurrence)
    event_bin = torch.zeros((count,), dtype=torch.int64)
    future_force = torch.zeros((count, len(EVENT_STEPS), 3), dtype=torch.float32)
    future_force_mask = torch.zeros((count, len(EVENT_STEPS)), dtype=torch.bool)
    if tuple(right_censored.shape) != (count, len(HORIZONS)):
        raise ValueError("invalid Round-7 censor metadata")
    for row, (episode, time_value, current, onset) in enumerate(zip(episode_ids, times.tolist(), current_slip.tolist(), onsets)):
        time_value = int(time_value)
        if (episode, time_value) not in label_lookup or label_lookup[(episode, time_value)] != int(current):
            raise ValueError("Round-7 current label disagrees with hashed parent observation")
        stable = int(current) == 0 and (onset is None or time_value < int(onset))
        complete_horizons = [horizon for index, horizon in enumerate(HORIZONS) if not bool(right_censored[row, index])]
        observed_through = max(complete_horizons, default=0)
        for column, step in enumerate(EVENT_STEPS):
            key = (episode, time_value + step)
            force_available = key in force_lookup
            future_force_mask[row, column] = force_available
            if force_available:
                future_force[row, column] = force_lookup[key]
            if not stable or step > observed_through:
                continue
            before_or_at_event = onset is None or time_value + step <= int(onset)
            if before_or_at_event:
                observed[row, column] = True
            if onset is not None and time_value + step == int(onset):
                occurrence[row, column] = True
                event_bin[row] = step
    if bool((occurrence & ~observed).any()) or bool((occurrence.sum(1) > 1).any()):
        raise ValueError("invalid discrete event representation")
    if any(row != sorted(row, reverse=True) for row in observed.tolist()):
        raise ValueError("event observation must be a contiguous at-risk prefix")
    return {
        "event_occurrence": occurrence,
        "event_observed_mask": observed,
        "event_time_bin": event_bin,
        "future_force_target": future_force,
        "future_force_observed_mask": future_force_mask,
    }


def transform_rows(rows: dict, force_lookup: dict[tuple[str, int], torch.Tensor], label_lookup: dict[tuple[str, int], int]) -> dict:
    old_base = finite_tensor(rows["base"], "round7 base", 3).float()
    if old_base.shape[1:] != (HISTORY, 772):
        raise ValueError("Round-7 base layout drift")
    episodes = list(rows["episode_id"])
    times = finite_tensor(rows["t"], "endpoint t", 1).long()
    history_force = torch.stack([
        torch.stack([force_lookup[(episode, int(t) - 8 + offset)] for offset in range(HISTORY)])
        for episode, t in zip(episodes, times.tolist())
    ]).float()
    base = torch.cat((old_base[:, :, :769], history_force), dim=2).contiguous()
    delta = torch.zeros((len(times), HISTORY, 3), dtype=torch.float32)
    delta[:, 5:] = history_force[:, 5:] - history_force[:, :4]
    old_delta = finite_tensor(rows["force_delta_slots"], "Round-7 force delta", 3).float()
    if old_delta.shape != delta.shape or not torch.equal(old_delta, delta):
        maximum = float((old_delta - delta).abs().max()) if old_delta.shape == delta.shape else float("inf")
        raise ValueError(f"Round-7 delta reconstruction mismatch: {maximum}")
    valid = torch.zeros((len(times), HISTORY, 1), dtype=torch.bool)
    valid[:, 5:] = True
    result = {
        key: value for key, value in rows.items()
        if key not in {"base", "force_delta_slots", "visual_delta_slots", "identity_sha256"}
    }
    result.update({
        "base": base,
        "signed_force_xyz": history_force,
        "force_delta_slots": delta,
        "shared_aux_valid": valid,
    })
    result.update(event_arrays(
        episodes,
        times,
        finite_tensor(rows["current_slip_label"], "current slip", 1),
        list(rows["first_current_slip_t"]),
        finite_tensor(rows["right_censored"], "Round-7 right censor", 2).bool(),
        force_lookup,
        label_lookup,
    ))
    for name in ("base", "signed_force_xyz", "force_delta_slots", "shared_aux_valid", "future_force_target"):
        finite_tensor(result[name], name)
    content_identity = identity_sha([
        (episodes[i], int(times[i]), result["horizon_mask"][i].tolist(), result["y"][i].tolist())
        for i in range(len(times))
    ])
    result["identity_sha256"] = rows["identity_sha256"]
    result["content_identity_sha256"] = content_identity
    return result


def p4_support(roles: dict, protocol: dict) -> dict:
    thresholds = protocol["p4_support_thresholds"]
    role_results = {}
    passed = True
    for role in ROLES:
        rows = roles[role]
        threshold = int(thresholds[role])
        per_step = []
        for column, step in enumerate(EVENT_STEPS):
            groups = sorted({
                group for group, available in zip(rows["leakage_group"], rows["future_force_observed_mask"][:, column].tolist())
                if available
            })
            step_pass = len(groups) >= threshold
            passed &= step_pass
            per_step.append({"step": step, "complete_independent_trials": len(groups), "minimum": threshold, "pass": step_pass, "identity_sha256": identity_sha(groups)})
        role_results[role] = per_step
    fit = roles["fit_train"]
    values = fit["future_force_target"][fit["future_force_observed_mask"]]
    variance = values.double().var(0, unbiased=False)
    variance_threshold = float(thresholds["minimum_fit_axis_variance"])
    variance_pass = bool(torch.isfinite(variance).all() and (variance > variance_threshold).all())
    passed &= variance_pass
    return {
        "triggered": bool(passed),
        "target": "frozen predicted force XYZ at t+1..t+5",
        "target_is_physical_ground_truth": False,
        "roles": role_results,
        "fit_axis_variance": variance.tolist(),
        "minimum_fit_axis_variance": variance_threshold,
        "variance_pass": variance_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round7-prepared", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    numeric = json.loads(NUMERIC_PROTOCOL.read_text())
    if protocol["history"] != HISTORY or protocol["horizons"] != list(HORIZONS) or protocol["event_steps"] != list(EVENT_STEPS):
        raise ValueError("prepare protocol constants drift")
    if numeric.get("schema") != "round8_root_numeric_protocol_v1":
        raise ValueError("Round-8 root numeric protocol missing")
    if numeric["history"] != HISTORY or numeric["candidate_horizons"] != list(HORIZONS):
        raise ValueError("root/prepare history or horizon disagreement")
    if numeric["aux_lag"] != 5 or numeric["aux_valid_positions"] != [5, 6, 7, 8]:
        raise ValueError("root/prepare auxiliary-layout disagreement")
    if numeric["input_slots"]["total"] != 776 or numeric["input_slots"]["base"] != 772:
        raise ValueError("root/prepare input-layout disagreement")
    if numeric["p4"]["target"] != "frozen predicted forceXYZ at t+1..t+5; representation not groundtruth physical dynamics":
        raise ValueError("root/prepare P4 target disagreement")
    if numeric["p4"]["minimum_independent_trials_per_step"] != {
        role: protocol["p4_support_thresholds"][role] for role in ROLES
    } or numeric["p4"]["fit_axis_variance_min"] != protocol["p4_support_thresholds"]["minimum_fit_axis_variance"]:
        raise ValueError("root/prepare P4 support-gate disagreement")
    round7 = torch.load(args.round7_prepared, map_location="cpu", weights_only=False)
    if round7.get("schema") != "round7_temporal_fairness_prepared_v1" or not round7.get("formal"):
        raise ValueError("formal Round-7 prepared input required")
    if round7.get("history") != HISTORY or round7.get("horizons") != list(HORIZONS) or round7.get("effective_minimum_t") != 18:
        raise ValueError("Round-7 population/history drift")

    train, train_parent = load_source_cache(args.train_cache, "train")
    val, val_parent = load_source_cache(args.val_cache, "val")
    force_lookup, label_lookup, force_audit = build_force_lookup([(train, train_parent), (val, val_parent)])
    roles = {role: transform_rows(round7["roles"][role], force_lookup, label_lookup) for role in ROLES}
    timelines = {role: transform_rows(round7["timelines"][role], force_lookup, label_lookup) for role in PREDICTION_ROLES}

    if any(roles[role]["identity_sha256"] != round7["roles"][role]["identity_sha256"] for role in ROLES):
        raise ValueError("Round-7 eligible population identity drift")
    if any(timelines[role]["identity_sha256"] != round7["timelines"][role]["identity_sha256"] for role in PREDICTION_ROLES):
        raise ValueError("Round-7 timeline population identity drift")

    fit = roles["fit_train"]
    normalization = {
        "fit_role": "fit_train",
        "base": population_stats(fit["base"].reshape(-1, 772)),
        "force_delta": population_stats(fit["force_delta_slots"][:, 5:].reshape(-1, 3)),
        "future_force": population_stats(fit["future_force_target"][fit["future_force_observed_mask"]]),
    }
    support = p4_support(roles, protocol)
    payload = {
        "schema": "round8_force_dynamics_prepared_v1",
        "status": "complete",
        "formal": True,
        "history": HISTORY,
        "horizons": list(HORIZONS),
        "event_steps": list(EVENT_STEPS),
        "effective_minimum_t": 18,
        "feature_layout": {"z": [0, 768], "p_slip": [768, 769], "signed_force_xyz": [769, 772], "force_delta": [772, 775], "shared_aux_valid": [775, 776], "total": 776},
        "group_active_slots": {
            "B_xyz": ["z", "p_slip", "signed_force_xyz", "shared_aux_valid"],
            "C_xyz_delta": ["z", "p_slip", "signed_force_xyz", "force_delta", "shared_aux_valid"],
        },
        "conditions": {"P3_triggered": True, "P4_triggered": support["triggered"]},
        "role_groups": round7["role_groups"],
        "role_identities": round7["role_identities"],
        "roles": roles,
        "timelines": timelines,
        "normalization": normalization,
        "p4_support": support,
        "provenance": {
            "prepare_protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
            "numeric_protocol": {"path": str(NUMERIC_PROTOCOL.resolve()), "sha256": sha256(NUMERIC_PROTOCOL)},
            "prepare_code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "round7_prepared": {"path": str(args.round7_prepared.resolve()), "sha256": sha256(args.round7_prepared)},
            "train_cache": {"path": str(args.train_cache.resolve()), "sha256": sha256(args.train_cache)},
            "val_cache": {"path": str(args.val_cache.resolve()), "sha256": sha256(args.val_cache)},
            "signed_force": force_audit,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "prepared.pt"
    atomic_torch(output, payload)
    audit = {
        "schema": "round8_force_dynamics_prepare_audit_v1",
        "status": "complete",
        "prepared": {"path": str(output.resolve()), "sha256": sha256(output), "bytes": output.stat().st_size},
        "roles": {
            role: {
                "eligible_rows": len(rows["t"]),
                "groups": len(set(rows["leakage_group"])),
                "identity_sha256": rows["identity_sha256"],
                "event_observed_counts_by_step": rows["event_observed_mask"].sum(0).tolist(),
                "event_counts_by_step": rows["event_occurrence"].sum(0).tolist(),
                "future_force_observed_counts_by_step": rows["future_force_observed_mask"].sum(0).tolist(),
            }
            for role, rows in roles.items()
        },
        "p4_support": support,
        "signed_force_audit": force_audit,
        "checks": {
            "round7_populations_identical": True,
            "exact_history_t_minus_8_to_t": True,
            "effective_minimum_t_18": True,
            "signed_xyz_from_hashed_frozen_parent": True,
            "lag5_delta_exactly_reconstructed": True,
            "lag5_only_latest_four_positions": True,
            "shared_aux_valid_for_all_groups": True,
            "event_unknown_future_not_negative": True,
            "event_endpoint_label_matches_hashed_parent": True,
            "event_future_observation_uses_round7_raw_label_censor_audit": True,
            "event_observation_is_contiguous_at_risk_prefix": True,
            "event_post_onset_risk_masked": True,
            "p4_future_force_is_target_only": True,
            "fit_only_normalization": True,
            "roles_pairwise_disjoint_inherited": True,
            "test_or_global_heldout_read": False
        },
        "provenance": payload["provenance"],
    }
    atomic_json(args.output / "PREPARE_AUDIT.json", audit)
    atomic_json(args.output / "P4_SUPPORT_DECISION.json", {
        "schema": "round8_p4_support_decision_v1",
        "status": "complete",
        **support,
        "reason": "fixed pre-result target passes all support gates" if support["triggered"] else "fixed pre-result target fails one or more support gates",
        "prepared_sha256": audit["prepared"]["sha256"],
    })
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
