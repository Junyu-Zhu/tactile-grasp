#!/usr/bin/env python3
"""Prepare role-isolated source-domain histories for Round-6 conditional future training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
ROLES = ("fit_train", "selection", "calibration", "outer")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _load_cache(path: Path, expected_split: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"schema", "split", "z", "p_slip", "force_features", "predicted_delta_force_xyz", "future_slip", "metadata"}
    if required - set(payload):
        raise KeyError(f"cache missing {sorted(required - set(payload))}")
    if payload["schema"] != "round5_source_pred_delta_v1" or payload["split"] != expected_split:
        raise ValueError(f"wrong cache identity: {path}")
    n = len(payload["metadata"])
    shapes = {
        "z": (n, 768), "p_slip": (n,), "force_features": (n, 3),
        "predicted_delta_force_xyz": (n, 3), "future_slip": (n, 3),
    }
    for key, shape in shapes.items():
        if tuple(payload[key].shape) != shape or not torch.isfinite(payload[key].float()).all():
            raise ValueError(f"invalid {key} in {path}")
    return payload


def _row_key(item: dict) -> tuple[str, int]:
    return f"source/{item['dataset']}/{item['trajectory']}", int(item["sample"])


def prepare(support_decision: Path, source_manifest: Path, train_cache: Path, val_cache: Path, output: Path) -> dict:
    decision = json.loads(support_decision.read_text())
    manifest = json.loads(source_manifest.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    if decision.get("status") != "complete" or not decision.get("training_triggered"):
        raise ValueError("support decision does not authorize conditional training")
    if decision.get("selected_support") != {"domain": "source", "horizon": 1}:
        raise ValueError("prepared-data implementation is locked to selected source H1")
    declared = decision["role_manifests"]["source"]
    if Path(declared["path"]).resolve() != source_manifest.resolve() or declared["sha256"] != sha256(source_manifest):
        raise ValueError("source support manifest identity mismatch")
    if manifest.get("status") != "complete" or manifest["decision"].get("selected_horizon_by_quantity") != 1:
        raise ValueError("source support is incomplete or not H1")
    input_specs = ((train_cache, "train", manifest["inputs"]["train_cache"]), (val_cache, "val", manifest["inputs"]["val_cache"]))
    caches = []
    for path, split, spec in input_specs:
        if Path(spec["path"]).resolve() != path.resolve() or spec["sha256"] != sha256(path):
            raise ValueError(f"cache identity mismatch: {split}")
        caches.append(_load_cache(path, split))

    index = {}
    for cache in caches:
        for i, item in enumerate(cache["metadata"]):
            key = _row_key(item)
            if key in index:
                raise ValueError(f"duplicate deployable row {key}")
            index[key] = (cache, i)

    role_group_sets = {role: set(manifest["roles"][role]["leakage_groups"]) for role in ROLES}
    for i, left in enumerate(ROLES):
        for right in ROLES[:i]:
            if role_group_sets[left] & role_group_sets[right]:
                raise ValueError(f"source leakage group crosses {left}/{right}")

    prepared_roles = {}
    role_identities = {}
    for role in ROLES:
        xs, ys, episode_ids, leakage_groups, onset_times, times = [], [], [], [], [], []
        trials = manifest["horizons"]["1"]["roles"][role]["trials"]
        allowed_groups = set(manifest["roles"][role]["leakage_groups"])
        for trial in trials:
            if trial["leakage_group"] not in allowed_groups:
                raise ValueError(f"trial outside role identity: {role}")
            endpoints = [(int(t), 1.0) for t in trial["positive_endpoints"]] + [(int(t), 0.0) for t in trial["negative_endpoints"]]
            for t, y in endpoints:
                rows = []
                for current in range(t - 3, t + 1):
                    key = (trial["episode_id"], current)
                    if key not in index:
                        raise ValueError(f"missing exact history row {key}")
                    cache, i = index[key]
                    rows.append(torch.cat((cache["z"][i].float(), cache["p_slip"][i:i+1].float(), cache["force_features"][i].float(), cache["predicted_delta_force_xyz"][i].float())))
                xs.append(torch.stack(rows)); ys.append(y); episode_ids.append(trial["episode_id"]); leakage_groups.append(trial["leakage_group"]); onset_times.append(trial["first_current_slip_t"]); times.append(t)
        if not xs:
            raise ValueError(f"empty prepared role: {role}")
        identity = [(episode_ids[i], times[i], int(ys[i])) for i in range(len(ys))]
        expected = manifest["horizons"]["1"]["roles"][role]["summary"]["eligible_endpoint_identity_sha256"]
        if identity_sha(sorted(identity)) != expected:
            raise ValueError(f"endpoint identity drift: {role}")
        x = torch.stack(xs).to(torch.float32)
        y = torch.tensor(ys, dtype=torch.float32)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or x.requires_grad or y.requires_grad:
            raise ValueError(f"non-finite or attached prepared tensors: {role}")
        prepared_roles[role] = {"x_C": x, "y": y, "episode_id": episode_ids, "leakage_group": leakage_groups, "first_current_slip_t": onset_times, "t": torch.tensor(times, dtype=torch.int64)}
        role_identities[role] = {"n": len(y), "positive": int(y.sum()), "negative": int(len(y)-y.sum()), "endpoint_identity_sha256": expected, "x_shape": list(x.shape)}

    fit = prepared_roles["fit_train"]["x_C"].float()
    mean = fit.mean((0, 1)); std = fit.std((0, 1), unbiased=False).clamp_min(float(protocol["normalization"]["std_floor"]))
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("invalid fit-only normalization")
    payload = {
        "schema": protocol["schema"] + "_prepared_v1", "status": "complete", "formal": True,
        "selected_domain": "source", "horizon": 1, "history": int(protocol["history"]),
        "feature_layout_C": {"z": [0, 768], "p_slip": [768, 769], "force_abs_fz_ft_ratio": [769, 772], "predicted_delta_xyz": [772, 775]},
        "group_input_dims": {"A_visual": 769, "B_force": 772, "C_force_delta": 775},
        "normalization_C": {"mean": mean, "std": std, "fit_role": "fit_train"},
        "roles": prepared_roles,
        "provenance": {
            "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
            "support_decision": {"path": str(support_decision.resolve()), "sha256": sha256(support_decision)},
            "source_manifest": {"path": str(source_manifest.resolve()), "sha256": sha256(source_manifest)},
            "train_cache": {"path": str(train_cache.resolve()), "sha256": sha256(train_cache)},
            "val_cache": {"path": str(val_cache.resolve()), "sha256": sha256(val_cache)},
        },
        "role_identities": role_identities,
        "disclosures": ["selection and calibration are derived from legacy source train and were seen by the historical model", "outer is original source validation", "source H1 is a one-step dataset-label onset task, not an independently instrumented physical lead-time guarantee"],
    }
    atomic_torch(output, payload)
    audit = {
        "format": "round6_conditional_prepare_audit_v1", "status": "complete", "formal": True,
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
        "protocol_sha256": sha256(PROTOCOL), "selected_support": decision["selected_support"],
        "role_identities": role_identities, "total_eligible_endpoints": sum(row["n"] for row in role_identities.values()), "expected_total_eligible_endpoints": 48375,
        "normalizer": {"fit_only": True, "dimensions": len(mean), "mean_finite": True, "std_positive_finite": True},
        "inputs": payload["provenance"], "test_content_read": False,
    }
    audit_path = output.with_name("PREPARE_AUDIT.json")
    if audit["total_eligible_endpoints"] != audit["expected_total_eligible_endpoints"]:
        raise ValueError("prepared source H1 endpoint total drift")
    atomic_json(audit_path, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--support-decision", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--source-train-cache", type=Path, required=True)
    p.add_argument("--source-val-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(*(x.resolve() for x in (a.support_decision, a.source_manifest, a.source_train_cache, a.source_val_cache, a.output)))


if __name__ == "__main__":
    main()
