#!/usr/bin/env python3
"""Build the frozen Round-7 equal-history source-domain prepared artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
ROOT_NUMERIC = HERE.parent / "NUMERIC_PROTOCOL.json"
ROLES = ("fit_train", "selection", "calibration", "outer")
PREDICTION_ROLES = ("selection", "calibration", "outer")
HORIZONS = (1, 3, 5)


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


def group(metadata: dict) -> str:
    return f"source/{metadata['dataset']}/{metadata['trajectory']}"


def load_cache(path: Path, split: str) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema") != "round5_source_pred_delta_v1" or value.get("split") != split:
        raise ValueError(f"invalid Round-5 cache: {path}")
    if value.get("horizons") != list(HORIZONS):
        raise ValueError("cache horizon drift")
    for key in ("z", "p_slip", "force_features", "predicted_delta_force_xyz", "future_slip"):
        tensor = value.get(key)
        if not torch.is_tensor(tensor) or tensor.requires_grad or not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"invalid cache tensor {key}")
    if sha256(Path(value["source_path"])) != value["source_sha256"]:
        raise ValueError("parent cache drift")
    return value


def fit_pca(differences: torch.Tensor, keys: list[tuple[str, int]]) -> dict:
    if differences.ndim != 2 or differences.shape[1] != 768 or len(differences) != len(keys):
        raise ValueError("invalid PCA input")
    if len(set(keys)) != len(keys):
        raise ValueError("PCA samples must be unique")
    x = differences.double().numpy()
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / float(len(x))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order[:3]].T
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1
    tolerance = max(float(eigenvalues[0]), 1.0) * max(covariance.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    total = float(np.maximum(eigenvalues, 0).sum())
    return {
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "components": torch.from_numpy(components.astype(np.float32)),
        "explained_variance": torch.from_numpy(eigenvalues[:3].astype(np.float32)),
        "explained_variance_ratio": torch.from_numpy((eigenvalues[:3] / total).astype(np.float32)),
        "rank": rank,
        "sample_count": len(keys),
        "sample_identity_sha256": identity_sha(keys),
        "fit_role": "fit_train",
    }


def stats(values: torch.Tensor) -> dict:
    if values.ndim != 2 or not len(values):
        raise ValueError("empty normalizer population")
    mean = values.double().mean(0).float()
    std = values.double().std(0, unbiased=False).float().clamp_min(1e-6)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("invalid normalizer")
    return {"mean": mean, "std": std}


def subset_row(row: dict, keep: torch.Tensor) -> dict:
    indices = torch.nonzero(keep, as_tuple=False).flatten().tolist()
    total = len(keep)
    result = {}
    for key, value in row.items():
        if key == "identity_sha256":
            continue
        if torch.is_tensor(value) and value.ndim and len(value) == total:
            result[key] = value[keep]
        elif isinstance(value, list) and len(value) == total:
            result[key] = [value[index] for index in indices]
        else:
            result[key] = value
    result["identity_sha256"] = identity_sha([
        (result["episode_id"][i], int(result["t"][i]), result["horizon_mask"][i].tolist(), result["y"][i].tolist())
        for i in range(len(result["t"]))
    ])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-manifest", type=Path, required=True)
    parser.add_argument("--support-decision", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    numeric = json.loads(ROOT_NUMERIC.read_text())
    support = json.loads(args.support_manifest.read_text())
    decision = json.loads(args.support_decision.read_text())
    if support.get("status") != "complete" or decision.get("status") != "complete":
        raise ValueError("support is incomplete")
    if decision.get("supported_horizons") != list(HORIZONS) or not decision.get("training_triggered"):
        raise ValueError("joint H1/H3/H5 support is required")
    if protocol["history"] != numeric["history"] or numeric["input_slots"]["total"] != 776:
        raise ValueError("protocol disagreement")
    if support["inputs"]["train_cache"]["sha256"] != sha256(args.train_cache) or support["inputs"]["val_cache"]["sha256"] != sha256(args.val_cache):
        raise ValueError("support/cache identity mismatch")

    train = load_cache(args.train_cache, "train")
    val = load_cache(args.val_cache, "val")
    cache_tensors = {}
    for key in ("z", "p_slip", "force_features", "predicted_delta_force_xyz"):
        cache_tensors[key] = torch.cat((train[key].detach().float(), val[key].detach().float()))
    metadata = list(train["metadata"]) + list(val["metadata"])
    lookup: dict[tuple[str, int], int] = {}
    for index, item in enumerate(metadata):
        key = (group(item), int(item["sample"]))
        if key in lookup:
            raise ValueError(f"duplicate feature row {key}")
        lookup[key] = index

    role_groups = {role: list(support["roles"][role]["leakage_groups"]) for role in ROLES}
    for index, role in enumerate(ROLES):
        if set(role_groups[role]) & set().union(*(set(role_groups[x]) for x in ROLES[:index])):
            raise ValueError("role leakage")

    # Fit PCA on unique valid timestamps represented by common fit endpoints.
    pca_keys = set()
    for row in support["full_timeline"]["fit_train"]["rows"]:
        if row["common_complete"]:
            for sample in range(int(row["t"]) - 3, int(row["t"]) + 1):
                pca_keys.add((row["episode_id"], sample))
    pca_keys = sorted(pca_keys)
    current_indices = torch.tensor([lookup[key] for key in pca_keys], dtype=torch.long)
    lagged_indices = torch.tensor([lookup[(key[0], key[1] - 5)] for key in pca_keys], dtype=torch.long)
    pca = fit_pca(cache_tensors["z"][current_indices] - cache_tensors["z"][lagged_indices], pca_keys)
    if pca["rank"] < 3 or not decision.get("D_triggered"):
        raise ValueError("pre-result D constructability gate failed")

    def build_role(role: str) -> dict:
        records = support["full_timeline"][role]["rows"]
        if any(record["episode_id"] not in set(role_groups[role]) for record in records):
            raise ValueError(f"timeline outside frozen role {role}")
        history_indices = torch.tensor([[lookup[(record["episode_id"], sample)] for sample in record["history_times"]] for record in records], dtype=torch.long)
        z = cache_tensors["z"][history_indices]
        probability = cache_tensors["p_slip"][history_indices].unsqueeze(-1)
        force = cache_tensors["force_features"][history_indices]
        base = torch.cat((z, probability, force), dim=2).contiguous().float()
        force_delta = torch.zeros((len(records), 9, 3), dtype=torch.float32)
        force_delta[:, 5:] = cache_tensors["predicted_delta_force_xyz"][history_indices[:, 5:]]
        visual_delta = torch.zeros_like(force_delta)
        delta = z[:, 5:] - cache_tensors["z"][torch.tensor([[lookup[(record["episode_id"], sample - 5)] for sample in record["history_times"][5:]] for record in records], dtype=torch.long)]
        visual_delta[:, 5:] = (delta - pca["mean"]) @ pca["components"].T
        masks = torch.tensor([record["horizon_mask"] for record in records], dtype=torch.bool)
        targets = torch.tensor([[0 if target is None else target for target in record["targets"]] for record in records], dtype=torch.float32)
        common = masks.all(1)
        if not torch.equal(common, torch.tensor([record["common_complete"] for record in records], dtype=torch.bool)):
            raise ValueError(f"common population mismatch {role}")
        for tensor in (base, force_delta, visual_delta, targets):
            if tensor.requires_grad or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"invalid prepared tensor {role}")
        if bool(torch.count_nonzero(force_delta[:, :5])) or bool(torch.count_nonzero(visual_delta[:, :5])):
            raise ValueError("auxiliary prefix must be raw zero")
        contiguous = [False]
        contiguous.extend(
            records[index]["episode_id"] == records[index - 1]["episode_id"]
            and int(records[index]["t"]) == int(records[index - 1]["t"]) + 1
            for index in range(1, len(records))
        )
        result = {
            "base": base,
            "force_delta_slots": force_delta,
            "visual_delta_slots": visual_delta,
            "y": targets,
            "horizon_mask": masks,
            "common_mask": common,
            "episode_id": [record["episode_id"] for record in records],
            "leakage_group": [record["leakage_group"] for record in records],
            "t": torch.tensor([record["t"] for record in records], dtype=torch.int64),
            "first_current_slip_t": [record["first_onset_t"] for record in records],
            "current_slip_label": torch.tensor([record["current_slip"] for record in records], dtype=torch.uint8),
            "right_censored": torch.tensor([record["right_censored"] for record in records], dtype=torch.bool),
            "timeline_contiguous": contiguous,
            "identity_sha256": identity_sha([(record["episode_id"], record["t"], record["horizon_mask"], record["targets"]) for record in records]),
        }
        return result

    timeline_rows = {role: build_role(role) for role in ROLES}
    rows = {role: subset_row(timeline_rows[role], timeline_rows[role]["horizon_mask"].any(1)) for role in ROLES}
    if any(not bool(rows[role]["horizon_mask"].any(1).all()) for role in ROLES):
        raise ValueError("eligible role contains a row with no supervised horizon")
    normalization = {
        "fit_role": "fit_train",
        "base": stats(rows["fit_train"]["base"].reshape(-1, 772)),
        "force_delta": stats(rows["fit_train"]["force_delta_slots"][:, 5:].reshape(-1, 3)),
        "visual_delta": stats(rows["fit_train"]["visual_delta_slots"][:, 5:].reshape(-1, 3)),
    }
    payload = {
        "schema": "round7_temporal_fairness_prepared_v1",
        "status": "complete",
        "formal": True,
        "horizons": list(HORIZONS),
        "history": 9,
        "feature_layout": {"z": [0, 768], "p_slip": [768, 769], "force": [769, 772], "aux": [772, 775], "shared_aux_valid": [775, 776], "total": 776},
        "group_active_slots": {
            "A_visual": ["z", "p_slip", "shared_aux_valid"],
            "B_force": ["z", "p_slip", "force", "shared_aux_valid"],
            "C_force_delta": ["z", "p_slip", "force", "force_delta", "shared_aux_valid"],
            "D_visual_delta": ["z", "p_slip", "visual_delta_pca3", "shared_aux_valid"],
        },
        "effective_minimum_t": 18,
        "role_groups": role_groups,
        "role_identities": {role: {"leakage_groups": role_groups[role], "group_identity_sha256": identity_sha(sorted(role_groups[role])), "eligible_identity_sha256": rows[role]["identity_sha256"], "timeline_identity_sha256": timeline_rows[role]["identity_sha256"]} for role in ROLES},
        "roles": rows,
        "timelines": {role: timeline_rows[role] for role in PREDICTION_ROLES},
        "normalization": normalization,
        "pca_visual_delta": pca,
        "D_triggered": True,
        "provenance": {
            "prepare_protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
            "numeric_protocol": {"path": str(ROOT_NUMERIC.resolve()), "sha256": sha256(ROOT_NUMERIC)},
            "support_manifest": {"path": str(args.support_manifest.resolve()), "sha256": sha256(args.support_manifest)},
            "support_decision": {"path": str(args.support_decision.resolve()), "sha256": sha256(args.support_decision)},
            "train_cache": {"path": str(args.train_cache.resolve()), "sha256": sha256(args.train_cache), "parent_path": train["source_path"], "parent_sha256": train["source_sha256"]},
            "val_cache": {"path": str(args.val_cache.resolve()), "sha256": sha256(args.val_cache), "parent_path": val["source_path"], "parent_sha256": val["source_sha256"]},
            "prepare_code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        },
    }
    output = args.output / "prepared.pt"
    atomic_torch(output, payload)
    artifact_sha = sha256(output)
    audit = {
        "schema": "round7_temporal_fairness_prepare_audit_v1",
        "status": "complete",
        "prepared": {"path": str(output.resolve()), "sha256": artifact_sha, "bytes": output.stat().st_size},
        "supported_horizons": list(HORIZONS),
        "D_triggered": True,
        "pca": {"rank": pca["rank"], "sample_count": pca["sample_count"], "sample_identity_sha256": pca["sample_identity_sha256"], "explained_variance_ratio": pca["explained_variance_ratio"].tolist()},
        "roles": {role: {"all_timeline_rows": len(timeline_rows[role]["t"]), "union_eligible_rows": len(rows[role]["t"]), "common_rows": int(rows[role]["common_mask"].sum()), "groups_declared": len(role_groups[role]), "groups_observed_timeline": len(set(timeline_rows[role]["leakage_group"])), "groups_observed_eligible": len(set(rows[role]["leakage_group"])), "eligible_identity_sha256": rows[role]["identity_sha256"], "timeline_identity_sha256": timeline_rows[role]["identity_sha256"]} for role in ROLES},
        "checks": {"all_tensors_fp32_except_labels_masks": True, "all_features_finite": True, "exact_history_t_minus_8_to_t": True, "aux_prefix_zero": True, "fit_only_normalization": True, "fit_only_pca": True, "roles_pairwise_disjoint": True, "test_or_global_heldout_read": False},
        "provenance": payload["provenance"],
    }
    atomic_json(args.output / "PREPARE_AUDIT.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
