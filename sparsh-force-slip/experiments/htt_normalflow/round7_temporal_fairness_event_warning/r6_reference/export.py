#!/usr/bin/env python3
"""Export frozen Round-6 H1 models on Round-7 shared causal timelines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
GROUP_DIMS = {"A_visual": 769, "B_force": 772, "C_force_delta": 775}
ROLES = ("selection", "calibration", "outer")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def module(path: Path):
    spec = importlib.util.spec_from_file_location("round6_frozen_trainer_reference", path)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


def validate_authority(inventory_path: Path, audit_path: Path, prepared_path: Path, trainer_path: Path) -> tuple[dict, dict, dict]:
    inventory = json.loads(inventory_path.read_text())
    audit = json.loads(audit_path.read_text())
    if inventory.get("schema") != "round6_formal_inventory_v1" or audit.get("status") != "pass" or audit.get("verified_runs") != 9:
        raise ValueError("Round-6 authority is incomplete")
    if inventory.get("prepared_data") != str(prepared_path) or inventory.get("trainer") != str(trainer_path):
        raise ValueError("Round-6 authority path mismatch")
    frozen = inventory.get("frozen_inputs", {})
    for path, digest in frozen.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Round-6 frozen input drift: {path}")
    prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
    if prepared.get("schema") != "round6_conditional_gru_v1_prepared_v1" or prepared.get("group_input_dims") != GROUP_DIMS:
        raise ValueError("invalid Round-6 prepared artifact")
    expected = {(group, seed) for group in GROUP_DIMS for seed in (20260914, 20260915, 20260916)}
    if {(run["group"], int(run["seed"])) for run in inventory["runs"]} != expected:
        raise ValueError("Round-6 run identity mismatch")
    return inventory, audit, prepared


def r7_x(row: dict, group: str, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    dim = GROUP_DIMS[group]
    base = row["base"].detach().float()[:, -4:]
    if tuple(base.shape[1:]) != (4, 772):
        raise ValueError("invalid Round-7 base timeline")
    if group == "C_force_delta":
        raw = torch.cat((base, row["force_delta_slots"].detach().float()[:, -4:]), dim=2)
    else:
        raw = base[:, :, :dim]
    result = (raw - mean[:dim]) / std[:dim]
    if result.requires_grad or tuple(result.shape[1:]) != (4, dim) or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid normalized reference input")
    return result


def predict(model: torch.nn.Module, values: torch.Tensor, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            pieces.append(torch.sigmoid(model(values[start:start + batch_size].to(device))).cpu())
    result = torch.cat(pieces).numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite reference prediction")
    return result


def original_predictions(path: Path) -> dict[tuple[str, int], float]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return {(row["episode_id"], int(row["t"])): float(row["p_future_raw"]) for row in rows}


def output_rows(row: dict, probabilities: np.ndarray) -> tuple[list[str], list[list[Any]]]:
    header = [
        "episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label", "p_slip_current",
        "timeline_contiguous", "common_population", "target_future_H1", "eligible_H1", "right_censored_H1",
        "common_eligible_H1", "p_future_H1_raw", "force_abs_fz", "force_ft", "force_ft_over_fn",
        "latest_force_delta_x", "latest_force_delta_y", "latest_force_delta_z", "latest_force_delta_valid",
    ]
    result = []
    for index, probability in enumerate(probabilities):
        eligible = bool(row["horizon_mask"][index, 0])
        common = bool(row["common_mask"][index])
        force = row["base"][index, -1, 769:772].float().tolist()
        delta = row["force_delta_slots"][index, -1].float().tolist()
        onset = row["first_current_slip_t"][index]
        result.append([
            row["episode_id"][index], row["leakage_group"][index], int(row["t"][index]), "" if onset is None else int(onset),
            int(row["current_slip_label"][index]), float(row["base"][index, -1, 768]), bool(row["timeline_contiguous"][index]),
            common, int(row["y"][index, 0]) if eligible else "", eligible, bool(row["right_censored"][index, 0]),
            bool(common and eligible), float(probability), *force, *delta, True,
        ])
    return header, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r6-inventory", type=Path, required=True)
    parser.add_argument("--r6-training-audit", type=Path, required=True)
    parser.add_argument("--r6-prepared", type=Path, required=True)
    parser.add_argument("--r6-trainer", type=Path, required=True)
    parser.add_argument("--r7-prepared", type=Path, required=True)
    parser.add_argument("--r7-prepare-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verification-only", action="store_true")
    args = parser.parse_args()

    inventory, r6_audit, r6_prepared = validate_authority(args.r6_inventory, args.r6_training_audit, args.r6_prepared, args.r6_trainer)
    r7_audit = json.loads(args.r7_prepare_audit.read_text())
    if r7_audit.get("status") != "complete" or r7_audit["prepared"]["sha256"] != sha256(args.r7_prepared):
        raise ValueError("Round-7 prepared authority mismatch")
    r7 = torch.load(args.r7_prepared, map_location="cpu", weights_only=False)
    if r7.get("schema") != "round7_temporal_fairness_prepared_v1" or not r7.get("formal"):
        raise ValueError("invalid Round-7 prepared artifact")
    trainer = module(args.r6_trainer)
    mean = r6_prepared["normalization_C"]["mean"].float()
    std = r6_prepared["normalization_C"]["std"].float()
    if tuple(mean.shape) != (775,) or tuple(std.shape) != (775,) or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("invalid Round-6 normalization")

    r6_role_lookup = {
        role: {(episode, int(t)): index for index, (episode, t) in enumerate(zip(r6_prepared["roles"][role]["episode_id"], r6_prepared["roles"][role]["t"]))}
        for role in ("calibration", "outer")
    }
    verification = []
    artifacts = []
    for run in inventory["runs"]:
        group, seed = run["group"], int(run["seed"])
        summary_path = Path(run["output"]) / "summary.json"
        summary = json.loads(summary_path.read_text())
        best_spec = summary["artifacts"]["best"]
        best_path = Path(best_spec["path"])
        if summary.get("status") != "complete" or not summary.get("formal") or sha256(best_path) != best_spec["sha256"]:
            raise ValueError(f"invalid Round-6 run {group}/{seed}")
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        dim = GROUP_DIMS[group]
        if not torch.equal(checkpoint["normalization"]["mean"], mean[:dim]) or not torch.equal(checkpoint["normalization"]["std"], std[:dim]):
            raise ValueError("checkpoint normalization differs from Round-6 prepared authority")
        model = trainer.GRURisk(dim, 128).to(args.device)
        model.load_state_dict(checkpoint["model_state"], strict=True)

        for role in ("calibration", "outer"):
            r7row = r7["timelines"][role]
            new_by_key = {(episode, int(t)): index for index, (episode, t) in enumerate(zip(r7row["episode_id"], r7row["t"])) if bool(r7row["common_mask"][index])}
            common = sorted(set(new_by_key) & set(r6_role_lookup[role]))[:32]
            if not common:
                raise ValueError(f"no verification intersection {role}")
            new_indices = torch.tensor([new_by_key[key] for key in common], dtype=torch.long)
            old_indices = torch.tensor([r6_role_lookup[role][key] for key in common], dtype=torch.long)
            new_input = r7_x({key: value[new_indices] if torch.is_tensor(value) and value.ndim and len(value) == len(r7row["t"]) else value for key, value in r7row.items()}, group, mean, std)
            old_input = (r6_prepared["roles"][role]["x_C"][old_indices, :, :dim].float() - mean[:dim]) / std[:dim]
            input_error = float((new_input - old_input).abs().max())
            probabilities = predict(model, new_input, args.device)
            originals = original_predictions(Path(run["output"]) / f"predictions_{role}.csv")
            prediction_error = max(abs(float(probabilities[index]) - originals[key]) for index, key in enumerate(common))
            verification.append({"group": group, "seed": seed, "role": role, "rows": len(common), "normalized_input_max_abs_error": input_error, "stored_prediction_max_abs_error": prediction_error})
            if input_error > 1e-7 or prediction_error > 2e-6:
                raise ValueError(f"Round-6 equivalence failed {group}/{seed}/{role}")

        if args.verification_only:
            continue
        run_output = args.output / "references" / f"future_{group}_{seed}"
        for role in ROLES:
            row = r7["timelines"][role]
            probabilities = predict(model, r7_x(row, group, mean, std), args.device)
            header, rows = output_rows(row, probabilities)
            path = run_output / f"predictions_timeline_{role}.csv"
            atomic_csv(path, header, rows)
            artifacts.append({"group": group, "seed": seed, "role": role, "path": str(path.resolve()), "sha256": sha256(path), "rows": len(rows), "identity_sha256": identity_sha([(x[0], x[2]) for x in rows])})

    status = "verified_inputs_only" if args.verification_only else "complete"
    manifest = {
        "schema": "round7_r6_h1_reference_manifest_v1",
        "status": status,
        "reference_only": True,
        "new_training": False,
        "formal_round7_run": False,
        "horizon": 1,
        "history": 4,
        "groups": list(GROUP_DIMS),
        "seeds": [20260914, 20260915, 20260916],
        "roles": list(ROLES),
        "verification": verification,
        "artifacts": artifacts,
        "provenance": {
            "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
            "export_code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "r6_inventory": {"path": str(args.r6_inventory.resolve()), "sha256": sha256(args.r6_inventory)},
            "r6_training_audit": {"path": str(args.r6_training_audit.resolve()), "sha256": sha256(args.r6_training_audit), "verified_runs": r6_audit["verified_runs"]},
            "r6_prepared": {"path": str(args.r6_prepared.resolve()), "sha256": sha256(args.r6_prepared)},
            "r6_trainer": {"path": str(args.r6_trainer.resolve()), "sha256": sha256(args.r6_trainer)},
            "r7_prepared": {"path": str(args.r7_prepared.resolve()), "sha256": sha256(args.r7_prepared)},
            "r7_prepare_audit": {"path": str(args.r7_prepare_audit.resolve()), "sha256": sha256(args.r7_prepare_audit)},
        },
        "disclosures": [
            "Round-6 selection participated in checkpoint choice and is descriptive legacy-seen output.",
            "Round-6 was trained with four outputs and H1 only; it is a historical reference rather than a fair Round-7 architecture/history peer.",
            "All rows come from Round-7 role-local causal timelines; no test or global heldout role is read.",
        ],
    }
    atomic_json(args.output / "REFERENCE_MANIFEST.json", manifest)
    print(json.dumps({"status": status, "verification_rows": sum(x["rows"] for x in verification), "artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
