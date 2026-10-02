#!/usr/bin/env python3
"""Round-8 force-dynamics, event-time, fusion, and state-supervision trainer."""
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
ROLES = ("fit_train", "selection", "calibration", "outer")
PREDICTION_ROLES = ("selection", "calibration", "outer")
HAZARD_GROUPS = frozenset(("C_hazard", "P3_fusion_hazard"))
P3_GROUPS = frozenset(("P3_concat_independent", "P3_fusion_independent", "P3_fusion_hazard"))
P4_GROUPS = frozenset(("P4_direct", "P4_state"))
DELTA_GROUPS = frozenset(("C_xyz_delta", "C_hazard", *P3_GROUPS, *P4_GROUPS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def head_type(group: str) -> str:
    return "discrete_hazard" if group in HAZARD_GROUPS else "independent_sigmoid"


class BaseGRU(nn.Module):
    def __init__(self, input_dim: int = 776, hidden_size: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True, dropout=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0][:, -1]


class PlainRiskGRU(BaseGRU):
    def __init__(self, output_dim: int, input_dim: int = 776, hidden_size: int = 128):
        super().__init__(input_dim, hidden_size)
        self.risk = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"risk_logits": self.risk(self.encode(x))}


class ProjectedRiskGRU(BaseGRU):
    """Capacity control for fusion: one joint per-step projection before the GRU."""

    def __init__(self, output_dim: int, hidden_size: int = 128):
        super().__init__(hidden_size, hidden_size)
        self.project = nn.Sequential(nn.Linear(776, hidden_size), nn.GELU())
        self.risk = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"risk_logits": self.risk(self.encode(self.project(x)))}


class FusionRiskGRU(BaseGRU):
    """Causal per-step visual/force residual fusion; no future inputs are accepted."""

    def __init__(self, output_dim: int, hidden_size: int = 128):
        super().__init__(hidden_size, hidden_size)
        self.visual = nn.Sequential(nn.Linear(769, 96), nn.GELU())
        self.force = nn.Sequential(nn.Linear(7, 32), nn.GELU())
        self.interaction = nn.Sequential(nn.Linear(128, 98), nn.GELU(), nn.Linear(98, 128))
        self.risk = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        visual = self.visual(x[:, :, :769])
        force = self.force(x[:, :, 769:])
        joined = torch.cat((visual, force), dim=-1)
        gate = torch.sigmoid(force.mean(dim=-1, keepdim=True))
        fused = joined + gate * torch.tanh(self.interaction(joined))
        return {"risk_logits": self.risk(self.encode(fused))}


class StateAugmentedGRU(BaseGRU):
    """Capacity-matched P4 model. Only P4_state applies state supervision."""

    def __init__(self, output_dim: int = 3, hidden_size: int = 128):
        super().__init__(776, hidden_size)
        self.state = nn.Linear(hidden_size, 15)
        self.risk = nn.Linear(hidden_size + 15, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(x)
        state = self.state(hidden).reshape(-1, 5, 3)
        return {"risk_logits": self.risk(torch.cat((hidden, state.flatten(1)), dim=1)), "state_prediction": state}


def make_model(group: str, hidden_size: int = 128) -> nn.Module:
    output_dim = 5 if group in HAZARD_GROUPS else 3
    if group == "P3_concat_independent":
        return ProjectedRiskGRU(output_dim, hidden_size)
    if group in ("P3_fusion_independent", "P3_fusion_hazard"):
        return FusionRiskGRU(output_dim, hidden_size)
    if group in P4_GROUPS:
        return StateAugmentedGRU(output_dim, hidden_size)
    return PlainRiskGRU(output_dim, 776, hidden_size)


def build_model(group: str, hidden_size: int = 128) -> nn.Module:
    """Stable public constructor used by evaluation and deployment benchmarks."""
    protocol = json.loads(PROTOCOL.read_text())
    if group not in protocol["groups"]:
        raise ValueError(f"unknown group: {group}")
    return make_model(group, hidden_size)


def load_best_model(checkpoint_path: Path, group: str, device: str = "cpu") -> nn.Module:
    """Load a strict Round-8 best checkpoint without silently crossing groups."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("run_config", {})
    if config.get("group") != group or config.get("head_type") != head_type(group):
        raise ValueError("checkpoint group or head type mismatch")
    model = build_model(group, int(config.get("hidden_size", 128)))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def fixed_initial_state(seed: int, group: str, hidden_size: int = 128) -> tuple[dict, dict]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_model(group, hidden_size)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return state, {key: tensor_sha(value) for key, value in state.items()}


def hazard_probabilities(logits: torch.Tensor, horizons: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    hazards = torch.sigmoid(logits)
    cumulative_all = 1.0 - torch.cumprod(1.0 - hazards, dim=1)
    return cumulative_all[:, [horizon - 1 for horizon in horizons]], hazards


def model_probabilities(output: dict[str, torch.Tensor], group: str, horizons: list[int]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if group in HAZARD_GROUPS:
        return hazard_probabilities(output["risk_logits"], horizons)
    return torch.sigmoid(output["risk_logits"]), None


def masked_horizon_bce(probabilities: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for index in range(probabilities.shape[1]):
        valid = mask[:, index].bool()
        if bool(valid.any()):
            losses.append(nn.functional.binary_cross_entropy(probabilities[valid, index], target[valid, index]))
    if not losses:
        raise ValueError("batch has no eligible horizon")
    return torch.stack(losses).mean()


def weighted_independent_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    losses = []
    for index in range(logits.shape[1]):
        valid = mask[:, index].bool()
        if bool(valid.any()):
            losses.append(nn.functional.binary_cross_entropy_with_logits(logits[valid, index], target[valid, index], pos_weight=pos_weight[index]))
    if not losses:
        raise ValueError("batch has no eligible horizon")
    return torch.stack(losses).mean()


def censored_hazard_nll(logits: torch.Tensor, occurrence: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Unweighted Bernoulli hazard NLL over the observed at-risk prefix."""
    if logits.shape != occurrence.shape or logits.shape != observed.shape:
        raise ValueError("hazard tensor shape mismatch")
    event_seen = torch.cumsum(occurrence.bool().to(torch.int64), dim=1)
    at_risk = observed.bool() & (event_seen <= 1) & (torch.cat((torch.zeros_like(event_seen[:, :1]), event_seen[:, :-1]), dim=1) == 0)
    if not bool(at_risk.any()):
        raise ValueError("batch has no observed hazard step")
    return nn.functional.binary_cross_entropy_with_logits(logits[at_risk], occurrence.float()[at_risk])


def masked_state_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool().unsqueeze(-1).expand_as(target)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return nn.functional.mse_loss(prediction[valid], target[valid])


def _tensor(row: dict, names: tuple[str, ...], tail: tuple[int, ...]) -> torch.Tensor:
    value = next((row[name] for name in names if name in row), None)
    if not torch.is_tensor(value) or value.ndim != len(tail) + 1 or tuple(value.shape[1:]) != tail:
        raise ValueError(f"invalid {'/'.join(names)} shape")
    if value.requires_grad or not bool(torch.isfinite(value.float()).all()):
        raise ValueError(f"invalid {'/'.join(names)} values")
    return value


def _metadata(row: dict, name: str, n: int) -> list:
    value = row.get(name)
    if torch.is_tensor(value):
        if value.ndim != 1 or len(value) != n:
            raise ValueError(f"invalid {name}")
        return value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"invalid {name}")
    return list(value)


def _row_base(row: dict) -> torch.Tensor:
    return _tensor(row, ("base", "base_seq"), (9, 772))


def validate_row(row: dict, *, timeline: bool) -> None:
    base = _row_base(row)
    n = len(base)
    delta = _tensor(row, ("force_delta_slots",), (9, 3))
    valid = _tensor(row, ("shared_aux_valid",), (9, 1)).bool()
    if len(delta) != n or bool(delta[:, :5].count_nonzero()) or bool(valid[:, :5].any()) or not bool(valid[:, 5:].all()):
        raise ValueError("lag5 delta and validity must be restricted to positions 5..8")
    y = _tensor(row, ("y",), (3,))
    horizon_mask = _tensor(row, ("horizon_mask",), (3,)).bool()
    occurrence = _tensor(row, ("event_occurrence",), (5,)).bool()
    observed = _tensor(row, ("event_observed_mask",), (5,)).bool()
    if any(len(value) != n for value in (y, horizon_mask, occurrence, observed)):
        raise ValueError("row count mismatch")
    if bool((occurrence & ~observed).any()) or bool((occurrence.sum(1) > 1).any()):
        raise ValueError("invalid event occurrence")
    for index in range(n):
        observed_row = observed[index].tolist()
        if observed_row != sorted(observed_row, reverse=True):
            raise ValueError("event observation must be a contiguous prefix")
    event_name = "event_time_bin" if "event_time_bin" in row else "event_bin"
    event_bin = torch.as_tensor(_metadata(row, event_name, n), dtype=torch.int64)
    derived_bin = torch.where(occurrence.any(1), occurrence.to(torch.int64).argmax(1) + 1, 0)
    if not torch.equal(event_bin, derived_bin):
        raise ValueError("event_bin disagrees with event_occurrence")
    if bool(((y[horizon_mask] != 0) & (y[horizon_mask] != 1)).any()):
        raise ValueError("eligible horizon targets must be binary")
    for hindex, horizon in enumerate((1, 3, 5)):
        expected = ((event_bin > 0) & (event_bin <= horizon)).float()
        if bool((y[horizon_mask[:, hindex], hindex] != expected[horizon_mask[:, hindex]]).any()):
            raise ValueError(f"H{horizon} target disagrees with event time")
        known = ((event_bin > 0) & (event_bin <= horizon)) | observed[:, horizon - 1]
        if bool((horizon_mask[:, hindex] & ~known).any()):
            raise ValueError(f"H{horizon} marks an unobserved future as eligible")
    for name in ("episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label"):
        _metadata(row, name, n)
    identities = list(zip((str(x) for x in _metadata(row, "episode_id", n)), (int(x) for x in _metadata(row, "t", n))))
    if len(set(identities)) != n:
        raise ValueError("duplicate episode/time identity")
    current = torch.as_tensor(_metadata(row, "current_slip_label", n))
    if bool(((current != 0) & (current != 1)).any()):
        raise ValueError("current slip labels must be binary")
    if not timeline and bool((current[horizon_mask.any(1)] != 0).any()):
        raise ValueError("eligible training population must be current-static")
    if timeline:
        _metadata(row, "timeline_contiguous", n)
        _tensor(row, ("right_censored",), (3,))
    if "future_force_target" in row or "future_force_observed_mask" in row:
        _tensor(row, ("future_force_target",), (5, 3))
        _tensor(row, ("future_force_observed_mask",), (5,))


def validate_prepared(payload: dict, data_path: Path | None = None) -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    if payload.get("schema") != protocol["prepared_schema"] or payload.get("status") != "complete" or not payload.get("formal"):
        raise ValueError("invalid prepared identity")
    if payload.get("horizons") != protocol["horizons"] or int(payload.get("history", -1)) != protocol["history"]:
        raise ValueError("history or horizons mismatch")
    normalization = payload.get("normalization", {})
    if normalization.get("fit_role") != "fit_train":
        raise ValueError("normalization must be fit-only")
    for name, width in (("base", 772), ("force_delta", 3), ("future_force", 3)):
        spec = normalization.get(name, {})
        mean, std = spec.get("mean"), spec.get("std")
        if not torch.is_tensor(mean) or not torch.is_tensor(std) or tuple(mean.shape) != (width,) or tuple(std.shape) != (width,):
            raise ValueError(f"invalid {name} normalizer")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
            raise ValueError(f"non-finite {name} normalizer")
    roles, timelines = payload.get("roles", {}), payload.get("timelines", {})
    if set(roles) != set(ROLES) or any(role not in timelines for role in PREDICTION_ROLES):
        raise ValueError("missing role or timeline")
    declared = payload.get("role_groups")
    if not isinstance(declared, dict) or set(declared) != set(ROLES):
        raise ValueError("missing role_groups")
    group_sets = {role: set(str(item) for item in declared[role]) for role in ROLES}
    for index, role in enumerate(ROLES):
        validate_row(roles[role], timeline=False)
        if not set(str(item) for item in roles[role]["leakage_group"]).issubset(group_sets[role]):
            raise ValueError(f"eligible group outside {role}")
        for previous in ROLES[:index]:
            if group_sets[role] & group_sets[previous]:
                raise ValueError(f"leakage group crosses {previous}/{role}")
        for hindex, horizon in enumerate(protocol["horizons"]):
            mask = roles[role]["horizon_mask"][:, hindex].bool()
            values = roles[role]["y"][mask, hindex]
            if not len(values) or int(values.unique().numel()) != 2:
                raise ValueError(f"{role} H{horizon} lacks both classes")
    for role in PREDICTION_ROLES:
        validate_row(timelines[role], timeline=True)
        if not set(str(item) for item in timelines[role]["leakage_group"]).issubset(group_sets[role]):
            raise ValueError(f"timeline group outside {role}")
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError("conditions must be a mapping")
    return {"protocol": protocol, "data_sha256": sha256(data_path) if data_path else None}


def active_input_columns(group: str) -> set[int]:
    active = set(range(772)) | {775}
    if group in DELTA_GROUPS:
        active.update(range(772, 775))
    return active


def assemble_inputs(payload: dict, row: dict, group: str) -> torch.Tensor:
    normalizers = payload["normalization"]
    base = (_row_base(row).float() - normalizers["base"]["mean"].float()) / normalizers["base"]["std"].float()
    delta = torch.zeros((len(base), 9, 3), dtype=torch.float32)
    if group in DELTA_GROUPS:
        normalized = (row["force_delta_slots"].float() - normalizers["force_delta"]["mean"].float()) / normalizers["force_delta"]["std"].float()
        delta[:, 5:] = normalized[:, 5:]
    valid = row["shared_aux_valid"].float()
    result = torch.cat((base, delta, valid), dim=2)
    if tuple(result.shape[1:]) != (9, 776) or result.requires_grad or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid assembled input")
    inactive = sorted(set(range(776)) - active_input_columns(group))
    if inactive and bool(result[:, :, inactive].count_nonzero()):
        raise ValueError("inactive input is nonzero")
    return result


def normalized_state_targets(payload: dict, row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    target = row["future_force_target"].float()
    spec = payload["normalization"]["future_force"]
    normalized = (target - spec["mean"].float()) / spec["std"].float()
    return normalized, row["future_force_observed_mask"].bool()


def set_epoch_seed(seed: int, epoch: int) -> None:
    value = seed + 1009 * epoch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def code_bundle_sha() -> tuple[str, dict[str, str]]:
    files = (Path(__file__).resolve(), PROTOCOL.resolve())
    hashes = {str(path): sha256(path) for path in files}
    return identity_sha(hashes), hashes


def _value(row: dict, name: str, index: int) -> Any:
    value = row[name]
    return value[index].item() if torch.is_tensor(value) else value[index]


def _prediction_header(group: str) -> list[str]:
    header = ["episode_id", "leakage_group", "t", "first_current_slip_t", "current_slip_label", "p_slip_current", "timeline_contiguous", "common_population"]
    for horizon in (1, 3, 5):
        header += [f"target_future_H{horizon}", f"eligible_H{horizon}", f"right_censored_H{horizon}", f"common_eligible_H{horizon}", f"p_future_H{horizon}_raw"]
    header += [f"q_future_step{step}_raw" for step in range(1, 6)]
    header += ["predicted_force_x", "predicted_force_y", "predicted_force_z", "force_abs_fz", "force_ft", "force_ft_over_fn", "latest_force_delta_x", "latest_force_delta_y", "latest_force_delta_z", "latest_force_delta_valid"]
    if group in P4_GROUPS:
        for step in range(1, 6):
            for axis in "xyz":
                header += [f"predicted_state_force_tplus{step}_{axis}", f"target_state_force_tplus{step}_{axis}"]
            header += [f"state_target_mask_tplus{step}"]
    return header


def infer(model: nn.Module, x: torch.Tensor, group: str, device: str, batch_size: int) -> dict[str, np.ndarray | None]:
    risks, hazards, states = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            output = model(x[start : start + batch_size].to(device))
            probability, hazard = model_probabilities(output, group, [1, 3, 5])
            risks.append(probability.cpu())
            if hazard is not None:
                hazards.append(hazard.cpu())
            if "state_prediction" in output:
                states.append(output["state_prediction"].cpu())
    result = {"probabilities": torch.cat(risks).numpy(), "hazards": torch.cat(hazards).numpy() if hazards else None, "states": torch.cat(states).numpy() if states else None}
    for value in result.values():
        if value is not None and not np.isfinite(value).all():
            raise FloatingPointError("non-finite inference")
    return result


def atomic_predictions(path: Path, payload: dict, row: dict, predictions: dict, group: str, *, timeline: bool) -> dict:
    probabilities = predictions["probabilities"]
    hazards, states = predictions["hazards"], predictions["states"]
    if group in HAZARD_GROUPS:
        expected = 1.0 - np.cumprod(1.0 - hazards, axis=1)[:, [0, 2, 4]]
        if not np.allclose(probabilities, expected, rtol=0, atol=1e-6):
            raise RuntimeError("hazard cumulative probability identity failed")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    header = _prediction_header(group)
    base = _row_base(row)
    mask, common = row["horizon_mask"].bool(), row.get("common_mask", row["horizon_mask"].bool().all(1)).bool()
    right = row["right_censored"].bool() if timeline else torch.zeros_like(mask)
    contiguous = row["timeline_contiguous"] if timeline else [True] * len(mask)
    state_mean = payload["normalization"]["future_force"]["mean"].numpy()
    state_std = payload["normalization"]["future_force"]["std"].numpy()
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index in range(len(mask)):
            line = [_value(row, "episode_id", index), _value(row, "leakage_group", index), int(_value(row, "t", index)), _value(row, "first_current_slip_t", index), int(_value(row, "current_slip_label", index)), float(base[index, -1, 768]), bool(contiguous[index]), bool(common[index])]
            for hindex, horizon in enumerate((1, 3, 5)):
                eligible = bool(mask[index, hindex])
                line += [int(row["y"][index, hindex]) if eligible else "", eligible, bool(right[index, hindex]), bool(common[index] and eligible), float(probabilities[index, hindex])]
            line += [float(hazards[index, step]) for step in range(5)] if hazards is not None else [""] * 5
            force = base[index, -1, 769:772].float().tolist()
            fn, ft = abs(force[2]), math.sqrt(force[0] ** 2 + force[1] ** 2)
            delta = row["force_delta_slots"][index, -1].float().tolist()
            line += force + [fn, ft, ft / (fn + 1e-6)] + delta + [bool(row["shared_aux_valid"][index, -1, 0])]
            if group in P4_GROUPS:
                predicted = states[index] * state_std + state_mean
                target = row["future_force_target"][index].float().numpy()
                state_mask = row["future_force_observed_mask"][index].bool().tolist()
                for step in range(5):
                    for axis in range(3):
                        line += [float(predicted[step, axis]), float(target[step, axis]) if state_mask[step] else ""]
                    line += [state_mask[step]]
            writer.writerow(line)
    os.replace(temporary, path)
    return {"schema": json.loads(PROTOCOL.read_text())["prediction_schema"], "path": str(path.resolve()), "sha256": sha256(path), "rows": len(mask), "columns": header}


def _group_enabled(payload: dict, group: str) -> bool:
    if group in P3_GROUPS:
        return bool(payload.get("conditions", {}).get("P3_triggered", False))
    if group in P4_GROUPS:
        return bool(payload.get("conditions", {}).get("P4_triggered", False))
    return True


def train(data_path: Path, group: str, seed: int, output: Path, device: str, smoke: bool, execute_formal: bool, resume: bool, interrupt_after_epoch: int | None) -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    if group not in protocol["groups"] or seed not in protocol["seeds"]:
        raise ValueError("unregistered run identity")
    if smoke == execute_formal:
        raise ValueError("select exactly one mode")
    total_epochs = 2 if smoke else protocol["max_epochs"]
    if interrupt_after_epoch is not None and not 0 < interrupt_after_epoch < total_epochs:
        raise ValueError("interrupt epoch must be inside budget")
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
    if not _group_enabled(payload, group):
        raise ValueError(f"{group} condition is not triggered")
    if group in P4_GROUPS:
        for role in ROLES:
            if "future_force_target" not in payload["roles"][role]:
                raise ValueError("P4 requires future force targets")
    code_sha, source_hashes = code_bundle_sha()
    initial, initial_hashes = fixed_initial_state(seed, group, protocol["model"]["gru_hidden_size"])
    run_config = {
        "schema": protocol["schema"], "prediction_schema": protocol["prediction_schema"], "group": group, "head_type": head_type(group), "seed": seed,
        "mode": "smoke" if smoke else "formal", "data_path": str(data_path.resolve()), "prepared_sha": checked["data_sha256"], "protocol_sha": sha256(PROTOCOL), "code_sha": code_sha,
        "source_hashes": source_hashes, "horizons": protocol["horizons"], "history": protocol["history"], "input_dim": protocol["input_dim"],
        "hidden_size": protocol["model"]["gru_hidden_size"], "effective_epoch_budget": total_epochs, "formal_max_epochs": protocol["max_epochs"], "patience": protocol["patience"],
        "batch_size": protocol["batch_size"], "optimizer": protocol["optimizer"], "selection": protocol["selection"], "state_loss_weight": protocol["state_loss_weight"],
        "active_input_columns": sorted(active_input_columns(group)), "initialization_tensor_sha256": initial_hashes,
    }
    run_identity = identity_sha(run_config)
    config_path, latest_path, best_path = output / "config.json", output / "latest.pth", output / "best.pth"
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output without --resume: {output}")
    if resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != run_config:
            raise ValueError("resume config mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, run_config)
    fit, selection = payload["roles"]["fit_train"], payload["roles"]["selection"]
    x_fit, x_selection = assemble_inputs(payload, fit, group), assemble_inputs(payload, selection, group)
    y_fit, mask_fit = fit["y"].float(), fit["horizon_mask"].bool()
    y_selection, mask_selection = selection["y"].float(), selection["horizon_mask"].bool()
    occurrence_fit, observed_fit = fit["event_occurrence"].float(), fit["event_observed_mask"].bool()
    if group in P4_GROUPS:
        state_fit, state_mask_fit = normalized_state_targets(payload, fit)
    else:
        state_fit, state_mask_fit = torch.zeros(len(x_fit), 5, 3), torch.zeros(len(x_fit), 5, dtype=torch.bool)
    weights = []
    for index in range(3):
        values = y_fit[mask_fit[:, index], index]
        positives, negatives = float(values.sum()), float(len(values) - values.sum())
        if positives <= 0 or negatives <= 0:
            raise ValueError("fit horizon lacks both classes")
        weights.append(negatives / positives)
    pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    model = make_model(group, protocol["model"]["gru_hidden_size"]).to(device)
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["optimizer"]["lr"], weight_decay=protocol["optimizer"]["weight_decay"])
    start_epoch, history, best_loss, best_epoch, best_state, stale = 1, [], math.inf, None, None, 0
    updated = {key: False for key in model.state_dict()}
    gradient_columns = torch.zeros(776, dtype=torch.bool)
    finite_gradients = True
    if resume:
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state.get("run_identity_sha256") != run_identity:
            raise ValueError("resume identity mismatch")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch, history = state["epoch"] + 1, state["history"]
        best_loss, best_epoch, best_state, stale = state["best_selection_loss"], state["best_epoch"], state["best_model_state"], state["stale"]
        updated, finite_gradients = dict(state["audit"]["parameter_tensor_updated"]), bool(state["audit"]["finite_all_parameter_gradients"])
        gradient_columns = state["audit"]["input_column_received_nonzero_data_gradient"].bool()
    dataset = TensorDataset(x_fit, y_fit, mask_fit, occurrence_fit, observed_fit, state_fit, state_mask_fit)
    for epoch in range(start_epoch, total_epochs + 1):
        if stale >= protocol["patience"]:
            break
        set_epoch_seed(seed, epoch)
        generator = torch.Generator().manual_seed(seed + 7919 * epoch)
        loader = DataLoader(dataset, batch_size=protocol["batch_size"], shuffle=True, generator=generator, num_workers=0)
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.train()
        fit_losses = []
        for x_batch, y_batch, m_batch, event_batch, observed_batch, state_batch, state_m_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            x_device = x_batch.to(device).detach().requires_grad_(True)
            output_model = model(x_device)
            if group in HAZARD_GROUPS:
                risk_loss = censored_hazard_nll(output_model["risk_logits"], event_batch.to(device), observed_batch.to(device))
            else:
                risk_loss = weighted_independent_loss(output_model["risk_logits"], y_batch.to(device), m_batch.to(device), pos_weight)
            state_loss = torch.zeros((), device=device)
            if group == "P4_state":
                state_loss = masked_state_mse(output_model["state_prediction"], state_batch.to(device), state_m_batch.to(device))
            loss = risk_loss + protocol["state_loss_weight"] * state_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite fit loss")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            finite_now = all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients)
            finite_gradients = finite_gradients and finite_now
            if not finite_now or x_device.grad is None:
                raise FloatingPointError("missing or non-finite gradient")
            # Audit gradient carried by the actual data, rather than sensitivity to a
            # hypothetical nonzero value in a deliberately zeroed inactive slot.
            gradient_columns |= ((x_device.grad.detach() * x_device.detach()).abs().sum((0, 1)).cpu() > 0)
            optimizer.step()
            fit_losses.append((float(loss.detach().cpu()), float(risk_loss.detach().cpu()), float(state_loss.detach().cpu())))
        for key, value in model.state_dict().items():
            updated[key] |= not torch.equal(before[key], value.detach().cpu())
        model.eval()
        with torch.no_grad():
            selection_output = model(x_selection.to(device))
            selection_probability, _ = model_probabilities(selection_output, group, protocol["horizons"])
            selection_loss = float(masked_horizon_bce(selection_probability, y_selection.to(device), mask_selection.to(device)).cpu())
        row_history = {"epoch": epoch, "fit_total_loss": float(np.mean([x[0] for x in fit_losses])), "fit_risk_loss": float(np.mean([x[1] for x in fit_losses])), "selection_unweighted_cumulative_horizon_bce": selection_loss}
        if group == "P4_state":
            row_history["fit_state_mse"] = float(np.mean([x[2] for x in fit_losses]))
        history.append(row_history)
        if selection_loss < best_loss:
            best_loss, best_epoch, stale = selection_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "history": history, "best_selection_loss": best_loss, "best_epoch": best_epoch, "best_model_state": best_state, "stale": stale, "run_config": run_config, "run_identity_sha256": run_identity, "positive_weights": weights, "audit": {"finite_all_parameter_gradients": finite_gradients, "parameter_tensor_updated": updated, "input_column_received_nonzero_data_gradient": gradient_columns}}
        atomic_torch(latest_path, checkpoint)
        if interrupt_after_epoch == epoch:
            audit = dict(checkpoint["audit"])
            audit["input_column_received_nonzero_data_gradient"] = gradient_columns.tolist()
            summary = {"schema": protocol["prediction_schema"], "status": "interrupted", "formal": execute_formal, "smoke": smoke, "group": group, "head_type": head_type(group), "seed": seed, "run_config": run_config, "run_identity_sha256": run_identity, "history": history, "audit": audit, "artifacts": {"latest": {"path": str(latest_path.resolve()), "sha256": sha256(latest_path)}}}
            atomic_json(output / "summary.json", summary)
            return summary
    if best_state is None:
        raise RuntimeError("no best checkpoint")
    active, inactive = active_input_columns(group), set(range(776)) - active_input_columns(group)
    if not finite_gradients or not all(bool(gradient_columns[index]) for index in active):
        raise RuntimeError("active input column lacks finite nonzero gradient")
    if any(bool(gradient_columns[index]) for index in inactive):
        raise RuntimeError("inactive input column received gradient")
    if not all(updated.values()):
        raise RuntimeError("trainable parameter tensor did not update")
    atomic_torch(best_path, {"model_state": best_state, "best_epoch": best_epoch, "best_selection_loss": best_loss, "run_config": run_config, "run_identity_sha256": run_identity, "positive_weights": weights})
    restored = torch.load(best_path, map_location="cpu", weights_only=False)
    if restored.get("run_identity_sha256") != run_identity or not nested_equal(restored["model_state"], best_state):
        raise RuntimeError("best checkpoint round-trip failure")
    model.load_state_dict(best_state, strict=True)
    artifacts: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    if execute_formal:
        for role in PREDICTION_ROLES:
            artifacts[role] = {}
            for population, row in (("eligible", payload["roles"][role]), ("timeline", payload["timelines"][role])):
                x = assemble_inputs(payload, row, group)
                prediction = infer(model, x, group, device, protocol["batch_size"])
                path = output / f"predictions_{population}_{role}.csv"
                artifacts[role][population] = atomic_predictions(path, payload, row, prediction, group, timeline=population == "timeline")
                probabilities = prediction["probabilities"]
                diagnostics[f"{population}_{role}"] = {f"H{horizon}": {"mean": float(probabilities[:, index].mean()), "std": float(probabilities[:, index].std()), "unique": int(np.unique(probabilities[:, index]).size)} for index, horizon in enumerate(protocol["horizons"])}
    audit = {"finite_all_parameter_gradients": finite_gradients, "all_parameter_tensors_updated": all(updated.values()), "parameter_tensor_updated": updated, "active_input_columns_all_received_nonzero_data_gradient": all(bool(gradient_columns[index]) for index in active), "inactive_input_columns_received_no_data_gradient": not any(bool(gradient_columns[index]) for index in inactive), "input_column_received_nonzero_data_gradient": gradient_columns.tolist(), "checkpoint_roundtrip_exact": True, "selection_only_checkpoint_choice": True, "outer_scored_by_trainer": False, "cached_inputs_have_no_upstream_trainable_parameters": True, "future_output_not_multiplied_by_current_slip_probability": True}
    summary = {"schema": protocol["prediction_schema"], "status": "complete", "formal": execute_formal, "smoke": smoke, "group": group, "head_type": head_type(group), "seed": seed, "run_config": run_config, "run_identity_sha256": run_identity, "best_epoch": best_epoch, "best_selection_loss": best_loss, "history": history, "positive_weights": weights, "model_parameters": sum(parameter.numel() for parameter in model.parameters()), "audit": audit, "probability_diagnostics": diagnostics, "artifacts": {"config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)}, "best": {"path": str(best_path.resolve()), "sha256": sha256(best_path)}, "latest": {"path": str(latest_path.resolve()), "sha256": sha256(latest_path)}, "predictions": artifacts}}
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--group", choices=protocol["groups"])
    parser.add_argument("--seed", type=int, choices=protocol["seeds"])
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
        checked = validate_prepared(torch.load(data_path, map_location="cpu", weights_only=False), data_path)
        print(json.dumps({"status": "pass", "data_sha256": checked["data_sha256"]}))
        return
    if args.group is None or args.seed is None or args.output is None:
        parser.error("--group, --seed, and --output are required")
    result = train(data_path, args.group, args.seed, args.output.resolve(), args.device, args.smoke, args.execute_formal, args.resume, args.interrupt_after_epoch)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
