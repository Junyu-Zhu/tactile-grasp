#!/usr/bin/env python3
"""Train the fixed Round-7 equal-history, multi-window future-risk models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
ROOT_NUMERIC = HERE.parent / "NUMERIC_PROTOCOL.json"
ROLES = ("fit_train", "selection", "calibration", "outer")
PREDICTION_ROLES = ("selection", "calibration", "outer")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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


class MultiWindowGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, horizons: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True, dropout=0)
        self.risk = nn.Linear(hidden_size, horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.risk(self.gru(x)[0][:, -1])


def nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def fixed_initial_state(seed: int, input_dim: int, hidden_size: int, horizons: int) -> tuple[dict, dict]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = MultiWindowGRU(input_dim, hidden_size, horizons)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return state, {key: tensor_sha(value) for key, value in state.items()}


def _tensor(row: dict, name: str, shape_tail: tuple[int, ...], dtype: torch.dtype | None = None) -> torch.Tensor:
    value = row.get(name)
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"invalid {name} shape: {tuple(value.shape)}")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"invalid {name} dtype: {value.dtype}")
    if value.requires_grad or not bool(torch.isfinite(value.float()).all()):
        raise ValueError(f"non-finite or attached {name}")
    return value


def _one_dimensional(row: dict, name: str, n: int) -> list:
    value = row.get(name)
    if torch.is_tensor(value):
        if value.ndim != 1 or len(value) != n:
            raise ValueError(f"invalid {name}")
        return value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"invalid {name}")
    return list(value)


def _validate_row(row: dict, horizons: list[int], history: int, *, timeline: bool) -> None:
    base = _tensor(row, "base", (history, 772))
    n = len(base)
    force_delta = _tensor(row, "force_delta_slots", (history, 3))
    visual_delta = _tensor(row, "visual_delta_slots", (history, 3))
    if len(force_delta) != n or len(visual_delta) != n:
        raise ValueError("feature row count mismatch")
    if bool(torch.count_nonzero(force_delta[:, :5]).item()) or bool(torch.count_nonzero(visual_delta[:, :5]).item()):
        raise ValueError("auxiliary values outside the fixed last-four positions")
    mask = _tensor(row, "horizon_mask", (len(horizons),)).bool()
    target = _tensor(row, "y", (len(horizons),))
    common = _tensor(row, "common_mask", ()).bool()
    if len(mask) != n or len(target) != n or len(common) != n:
        raise ValueError("target row count mismatch")
    if not torch.equal(common, mask.all(1)):
        raise ValueError("common_mask must equal the all-horizon intersection")
    if bool(((target[mask] != 0) & (target[mask] != 1)).any()):
        raise ValueError("eligible targets must be binary")
    if not timeline and not bool(mask.any(1).all()):
        raise ValueError("eligible union contains a row with no valid horizon")
    for name in ("episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label"):
        _one_dimensional(row, name, n)
    current = torch.as_tensor(_one_dimensional(row, "current_slip_label", n))
    if bool(((current != 0) & (current != 1)).any()):
        raise ValueError("current slip labels must be complete and binary")
    if not timeline and bool((current[mask.any(1)] != 0).any()):
        raise ValueError("eligible training population must be current-static")
    identities = [(str(_one_dimensional(row, "episode_id", n)[i]), int(_one_dimensional(row, "t", n)[i])) for i in range(n)]
    if len(set(identities)) != n:
        raise ValueError("duplicate episode/time identity")
    if timeline:
        right = _tensor(row, "right_censored", (len(horizons),)).bool()
        if len(right) != n:
            raise ValueError("right-censor row count mismatch")
        _one_dimensional(row, "timeline_contiguous", n)


def validate_prepared(payload: dict, data_path: Path | None = None) -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    numeric = json.loads(ROOT_NUMERIC.read_text())
    if payload.get("schema") != protocol["prepared_schema"] or payload.get("status") != "complete" or not payload.get("formal"):
        raise ValueError("invalid prepared identity")
    horizons = [int(x) for x in payload.get("horizons", [])]
    if not horizons or horizons != sorted(horizons) or any(x not in protocol["candidate_horizons"] for x in horizons):
        raise ValueError("invalid supported horizons")
    if int(payload.get("history", -1)) != int(protocol["history"]):
        raise ValueError("history mismatch")
    provenance = payload.get("provenance", {})
    numeric_spec = provenance.get("numeric_protocol", {})
    if numeric_spec.get("sha256") != sha256(ROOT_NUMERIC):
        raise ValueError("root numeric protocol drift")
    if numeric["history"] != protocol["history"] or numeric["input_slots"]["total"] != protocol["feature_layout"]["total"]:
        raise ValueError("root and trainer protocols disagree")
    normalizers = payload.get("normalization", {})
    if normalizers.get("fit_role") != "fit_train":
        raise ValueError("normalization must be fit-only")
    for name, width in (("base", 772), ("force_delta", 3), ("visual_delta", 3)):
        item = normalizers.get(name, {})
        mean, std = item.get("mean"), item.get("std")
        if not torch.is_tensor(mean) or not torch.is_tensor(std) or tuple(mean.shape) != (width,) or tuple(std.shape) != (width,):
            raise ValueError(f"invalid {name} normalizer")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
            raise ValueError(f"non-finite {name} normalizer")
    roles = payload.get("roles", {})
    timelines = payload.get("timelines", {})
    if set(roles) != set(ROLES) or any(role not in timelines for role in PREDICTION_ROLES):
        raise ValueError("missing role or causal timeline")
    declared_groups = payload.get("role_groups")
    if not isinstance(declared_groups, dict) or set(declared_groups) != set(ROLES):
        raise ValueError("missing authoritative role_groups")
    groups_by_role: dict[str, set[str]] = {role: set(str(x) for x in declared_groups[role]) for role in ROLES}
    for role in ROLES:
        _validate_row(roles[role], horizons, protocol["history"], timeline=False)
        if not set(str(x) for x in roles[role]["leakage_group"]).issubset(groups_by_role[role]):
            raise ValueError(f"eligible row group outside role: {role}")
        for earlier in ROLES[: ROLES.index(role)]:
            if groups_by_role[role] & groups_by_role[earlier]:
                raise ValueError(f"leakage group crosses {earlier}/{role}")
        role_mask = roles[role]["horizon_mask"].bool()
        role_target = roles[role]["y"]
        for hindex, horizon in enumerate(horizons):
            values = role_target[role_mask[:, hindex], hindex]
            if len(values) == 0 or int(torch.unique(values).numel()) != 2:
                raise ValueError(f"role {role} horizon H{horizon} lacks both classes")
    for role in PREDICTION_ROLES:
        _validate_row(timelines[role], horizons, protocol["history"], timeline=True)
        if not set(str(x) for x in timelines[role]["leakage_group"]).issubset(groups_by_role[role]):
            raise ValueError(f"timeline group outside role: {role}")
    return {"protocol": protocol, "numeric": numeric, "horizons": horizons, "data_sha256": sha256(data_path) if data_path else None}


def active_input_columns(group: str) -> set[int]:
    active = set(range(769)) | {775}
    if group in ("B_force", "C_force_delta"):
        active.update(range(769, 772))
    if group in ("C_force_delta", "D_visual_delta"):
        active.update(range(772, 775))
    return active


def assemble_inputs(payload: dict, row: dict, group: str) -> torch.Tensor:
    normalizers = payload["normalization"]
    base = (row["base"].float() - normalizers["base"]["mean"].float()) / normalizers["base"]["std"].float()
    if group in ("A_visual", "D_visual_delta"):
        base[:, :, 769:772] = 0
    aux = torch.zeros((len(base), base.shape[1], 3), dtype=torch.float32)
    if group == "C_force_delta":
        aux[:, 5:] = (row["force_delta_slots"].float()[:, 5:] - normalizers["force_delta"]["mean"].float()) / normalizers["force_delta"]["std"].float()
    elif group == "D_visual_delta":
        aux[:, 5:] = (row["visual_delta_slots"].float()[:, 5:] - normalizers["visual_delta"]["mean"].float()) / normalizers["visual_delta"]["std"].float()
    valid = torch.zeros((len(base), base.shape[1], 1), dtype=torch.float32)
    valid[:, 5:] = 1
    result = torch.cat((base, aux, valid), 2)
    if tuple(result.shape[1:]) != (9, 776) or not bool(torch.isfinite(result).all()) or result.requires_grad:
        raise ValueError("invalid assembled input")
    inactive = set(range(776)) - active_input_columns(group)
    if inactive and bool(torch.count_nonzero(result[:, :, sorted(inactive)]).item()):
        raise ValueError("inactive input is nonzero after normalization")
    return result


def masked_horizon_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor | None) -> torch.Tensor:
    losses = []
    for index in range(logits.shape[1]):
        valid = mask[:, index].bool()
        if bool(valid.any()):
            losses.append(nn.functional.binary_cross_entropy_with_logits(logits[valid, index], target[valid, index], pos_weight=None if pos_weight is None else pos_weight[index]))
    if not losses:
        raise ValueError("batch has no eligible horizon")
    return torch.stack(losses).mean()


def set_epoch_seed(seed: int, epoch: int) -> None:
    value = seed + 1009 * epoch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def code_bundle_sha() -> tuple[str, dict[str, str]]:
    files = (Path(__file__).resolve(), PROTOCOL.resolve(), ROOT_NUMERIC.resolve())
    hashes = {str(path): sha256(path) for path in files}
    return identity_sha(hashes), hashes


def _metadata_value(row: dict, name: str, index: int) -> Any:
    value = row[name]
    if torch.is_tensor(value):
        value = value[index].item()
    else:
        value = value[index]
    return value


def _prediction_header(horizons: list[int]) -> list[str]:
    result = ["episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label", "p_slip_current", "timeline_contiguous", "common_population"]
    for horizon in horizons:
        result += [f"target_future_H{horizon}", f"eligible_H{horizon}", f"right_censored_H{horizon}", f"common_eligible_H{horizon}", f"p_future_H{horizon}_raw"]
    result += ["force_abs_fz", "force_ft", "force_ft_over_fn", "latest_force_delta_x", "latest_force_delta_y", "latest_force_delta_z", "latest_force_delta_valid"]
    return result


def predict(model: nn.Module, x: torch.Tensor, device: str, batch_size: int) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values.append(torch.sigmoid(model(x[start : start + batch_size].to(device))).cpu())
    result = torch.cat(values).numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite prediction")
    return result


def atomic_predictions(path: Path, row: dict, probabilities: np.ndarray, horizons: list[int], *, timeline: bool) -> dict:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    header = _prediction_header(horizons)
    mask = row["horizon_mask"].bool()
    common = row["common_mask"].bool()
    right = row["right_censored"].bool() if timeline else torch.zeros_like(mask)
    contiguous = row["timeline_contiguous"] if timeline else [True] * len(mask)
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index in range(len(mask)):
            line = [
                _metadata_value(row, "episode_id", index), _metadata_value(row, "leakage_group", index), int(_metadata_value(row, "t", index)),
                _metadata_value(row, "first_current_slip_t", index), int(_metadata_value(row, "current_slip_label", index)), float(row["base"][index, -1, 768]), bool(_metadata_value({"x": contiguous}, "x", index)), bool(common[index]),
            ]
            for hindex, horizon in enumerate(horizons):
                eligible = bool(mask[index, hindex])
                line += [int(row["y"][index, hindex]) if eligible else "", eligible, bool(right[index, hindex]), bool(common[index] and eligible), float(probabilities[index, hindex])]
            force = row["base"][index, -1, 769:772].float().tolist()
            delta = row["force_delta_slots"][index, -1].float().tolist()
            line += force + delta + [True]
            writer.writerow(line)
    os.replace(temporary, path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "rows": len(mask), "columns": header}


def train(data_path: Path, group: str, seed: int, output: Path, device: str, smoke: bool, execute_formal: bool, resume: bool, interrupt_after_epoch: int | None) -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    groups = tuple(protocol["groups"])
    seeds = tuple(int(x) for x in protocol["seeds"])
    if group not in groups or seed not in seeds:
        raise ValueError("unregistered run identity")
    if smoke == execute_formal:
        raise ValueError("select exactly one mode: --smoke or --execute-formal")
    total_epochs = 2 if smoke else int(protocol["max_epochs"])
    if interrupt_after_epoch is not None and not 0 < interrupt_after_epoch < total_epochs:
        raise ValueError("interrupt epoch must be inside the unchanged budget")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    checked = validate_prepared(payload, data_path)
    horizons = checked["horizons"]
    if group == "D_visual_delta" and not payload["D_triggered"]:
        raise ValueError("D is not constructable")
    code_sha, source_hashes = code_bundle_sha()
    run_config = {
        "schema": protocol["schema"], "group": group, "seed": seed, "mode": "smoke" if smoke else "formal",
        "data_path": str(data_path.resolve()), "data_sha256": checked["data_sha256"], "protocol_sha256": sha256(PROTOCOL),
        "numeric_protocol_sha256": sha256(ROOT_NUMERIC), "code_bundle_sha256": code_sha, "source_hashes": source_hashes,
        "horizons": horizons, "history": int(protocol["history"]), "input_dim": 776, "hidden_size": int(protocol["model"]["hidden_size"]),
        "effective_epoch_budget": total_epochs, "formal_max_epochs": int(protocol["max_epochs"]), "patience": int(protocol["patience"]),
        "batch_size": int(protocol["batch_size"]), "optimizer": protocol["optimizer"], "selection": protocol["selection"],
        "active_input_columns": sorted(active_input_columns(group)), "inactive_input_columns": sorted(set(range(776)) - active_input_columns(group)),
    }
    initial, initialization_hashes = fixed_initial_state(seed, 776, int(protocol["model"]["hidden_size"]), len(horizons))
    run_config["initialization_tensor_sha256"] = initialization_hashes
    run_identity = identity_sha(run_config)
    config_path = output / "config.json"
    latest_path, best_path = output / "latest.pth", output / "best.pth"
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output without --resume: {output}")
    if resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != run_config:
            raise ValueError("resume config mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, run_config)
    x_fit = assemble_inputs(payload, payload["roles"]["fit_train"], group)
    y_fit = payload["roles"]["fit_train"]["y"].float()
    mask_fit = payload["roles"]["fit_train"]["horizon_mask"].bool()
    x_selection = assemble_inputs(payload, payload["roles"]["selection"], group)
    y_selection = payload["roles"]["selection"]["y"].float()
    mask_selection = payload["roles"]["selection"]["horizon_mask"].bool()
    positive_weights = []
    for hindex in range(len(horizons)):
        values = y_fit[mask_fit[:, hindex], hindex]
        positives, negatives = float(values.sum()), float(len(values) - values.sum())
        if positives <= 0 or negatives <= 0:
            raise ValueError("fit horizon must have both classes")
        positive_weights.append(negatives / positives)
    positive_weights_tensor = torch.tensor(positive_weights, dtype=torch.float32, device=device)
    model = MultiWindowGRU(776, int(protocol["model"]["hidden_size"]), len(horizons)).to(device)
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["optimizer"]["lr"]), weight_decay=float(protocol["optimizer"]["weight_decay"]))
    start_epoch, history, best_loss, best_epoch, best_state, stale = 1, [], math.inf, None, None, 0
    tensor_updates = {key: False for key in model.state_dict()}
    gradient_columns = torch.zeros(776, dtype=torch.bool)
    finite_gradients = True
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(latest_path)
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state.get("run_identity_sha256") != run_identity:
            raise ValueError("resume identity mismatch")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"]) + 1
        history = state["history"]
        best_loss = float(state["best_selection_loss"])
        best_epoch = state["best_epoch"]
        best_state = state["best_model_state"]
        stale = int(state["stale"])
        tensor_updates = dict(state["audit"]["parameter_tensor_updated"])
        gradient_columns = state["audit"]["input_column_received_nonzero_data_gradient"].bool()
        finite_gradients = bool(state["audit"]["finite_all_parameter_gradients"])
    dataset = TensorDataset(x_fit, y_fit, mask_fit)
    for epoch in range(start_epoch, total_epochs + 1):
        if stale >= int(protocol["patience"]):
            break
        set_epoch_seed(seed, epoch)
        generator = torch.Generator().manual_seed(seed + 7919 * epoch)
        loader = DataLoader(dataset, batch_size=int(protocol["batch_size"]), shuffle=True, generator=generator, num_workers=0)
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.train()
        losses = []
        for x_batch, y_batch, mask_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_horizon_loss(model(x_batch.to(device)), y_batch.to(device), mask_batch.to(device), positive_weights_tensor)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite fit loss")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            finite_now = all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
            finite_gradients = finite_gradients and finite_now
            if not finite_now:
                raise FloatingPointError("missing or non-finite parameter gradient")
            input_gradient = model.gru.weight_ih_l0.grad.detach().abs().sum(0).cpu() > 0
            gradient_columns |= input_gradient
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        for key, value in model.state_dict().items():
            tensor_updates[key] |= not torch.equal(before[key], value.detach().cpu())
        model.eval()
        with torch.no_grad():
            selection_loss = float(masked_horizon_loss(model(x_selection.to(device)), y_selection.to(device), mask_selection.to(device), None).cpu())
        if not math.isfinite(selection_loss):
            raise FloatingPointError("non-finite selection loss")
        history.append({"epoch": epoch, "fit_weighted_masked_bce": float(np.mean(losses)), "selection_unweighted_equal_horizon_bce": selection_loss})
        if selection_loss < best_loss:
            best_loss, best_epoch, stale = selection_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        checkpoint = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "history": history,
            "best_selection_loss": best_loss, "best_epoch": best_epoch, "best_model_state": best_state, "stale": stale,
            "run_config": run_config, "run_identity_sha256": run_identity, "positive_weights": positive_weights,
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            "audit": {"finite_all_parameter_gradients": finite_gradients, "parameter_tensor_updated": tensor_updates, "input_column_received_nonzero_data_gradient": gradient_columns},
        }
        atomic_torch(latest_path, checkpoint)
        if interrupt_after_epoch == epoch:
            audit_json = dict(checkpoint["audit"])
            audit_json["input_column_received_nonzero_data_gradient"] = gradient_columns.tolist()
            summary = {"status": "interrupted", "formal": execute_formal, "smoke": smoke, "group": group, "seed": seed, "run_config": run_config, "run_identity_sha256": run_identity, "history": history, "audit": audit_json, "artifacts": {"latest": {"path": str(latest_path.resolve()), "sha256": sha256(latest_path)}}}
            atomic_json(output / "summary.json", summary)
            return summary
    if best_state is None:
        raise RuntimeError("no best checkpoint")
    active, inactive = active_input_columns(group), set(range(776)) - active_input_columns(group)
    if not finite_gradients or not all(bool(gradient_columns[index]) for index in active):
        raise RuntimeError("an active input column lacks finite nonzero data gradient")
    if any(bool(gradient_columns[index]) for index in inactive):
        raise RuntimeError("an inactive information column received data gradient")
    if not all(tensor_updates.values()):
        raise RuntimeError("a trainable parameter tensor did not update")
    atomic_torch(best_path, {"model_state": best_state, "best_epoch": best_epoch, "best_selection_loss": best_loss, "run_config": run_config, "run_identity_sha256": run_identity, "positive_weights": positive_weights})
    restored = torch.load(best_path, map_location="cpu", weights_only=False)
    if restored.get("run_identity_sha256") != run_identity or not nested_equal(restored["model_state"], best_state):
        raise RuntimeError("best checkpoint round-trip failure")
    model.load_state_dict(best_state, strict=True)
    artifacts: dict[str, Any] = {}
    probability_diagnostics: dict[str, Any] = {}
    if execute_formal:
        for role in PREDICTION_ROLES:
            artifacts[role] = {}
            for population, row in (("eligible", payload["roles"][role]), ("timeline", payload["timelines"][role])):
                x = assemble_inputs(payload, row, group)
                probabilities = predict(model, x, device, int(protocol["batch_size"]))
                path = output / f"predictions_{population}_{role}.csv"
                artifacts[role][population] = atomic_predictions(path, row, probabilities, horizons, timeline=population == "timeline")
                probability_diagnostics[f"{population}_{role}"] = {
                    f"H{horizon}": {"mean": float(probabilities[:, hindex].mean()), "std": float(probabilities[:, hindex].std()), "unique": int(len(np.unique(probabilities[:, hindex])))}
                    for hindex, horizon in enumerate(horizons)
                }
    summary = {
        "status": "complete", "formal": execute_formal, "smoke": smoke, "group": group, "seed": seed,
        "run_config": run_config, "run_identity_sha256": run_identity, "best_epoch": best_epoch, "best_selection_loss": best_loss,
        "history": history, "positive_weights": positive_weights,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "population": {
            role: {
                "union_rows": int(len(payload["roles"][role]["y"])),
                "common_rows": int(payload["roles"][role]["common_mask"].sum()),
                "per_horizon": {
                    f"H{horizon}": {
                        "eligible": int(payload["roles"][role]["horizon_mask"][:, hindex].sum()),
                        "positive": int(payload["roles"][role]["y"][payload["roles"][role]["horizon_mask"][:, hindex], hindex].sum()),
                    }
                    for hindex, horizon in enumerate(horizons)
                },
            }
            for role in ROLES
        },
        "audit": {
            "finite_all_parameter_gradients": finite_gradients, "all_parameter_tensors_updated": all(tensor_updates.values()),
            "parameter_tensor_updated": tensor_updates, "active_input_columns_all_received_nonzero_data_gradient": all(bool(gradient_columns[index]) for index in active),
            "inactive_input_columns_received_no_data_gradient": not any(bool(gradient_columns[index]) for index in inactive),
            "input_column_received_nonzero_data_gradient": gradient_columns.tolist(), "checkpoint_roundtrip_exact": True,
            "selection_only_checkpoint_choice": True, "outer_scored_by_trainer": False,
        },
        "probability_diagnostics": probability_diagnostics,
        "artifacts": {
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            "best": {"path": str(best_path.resolve()), "sha256": sha256(best_path)},
            "latest": {"path": str(latest_path.resolve()), "sha256": sha256(latest_path)},
            "predictions": artifacts,
        },
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--group", choices=json.loads(PROTOCOL.read_text())["groups"])
    parser.add_argument("--seed", type=int, choices=json.loads(PROTOCOL.read_text())["seeds"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute-formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interrupt-after-epoch", type=int)
    parser.add_argument("--validate-data", action="store_true")
    args = parser.parse_args()
    data_path = args.data.resolve()
    if args.validate_data:
        payload = torch.load(data_path, map_location="cpu", weights_only=False)
        checked = validate_prepared(payload, data_path)
        print(json.dumps({"status": "pass", "horizons": checked["horizons"], "data_sha256": checked["data_sha256"]}))
        return
    if args.group is None or args.seed is None or args.output is None:
        parser.error("--group, --seed, and --output are required for training")
    result = train(data_path, args.group, args.seed, args.output.resolve(), args.device, args.smoke, args.execute_formal, args.resume, args.interrupt_after_epoch)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
