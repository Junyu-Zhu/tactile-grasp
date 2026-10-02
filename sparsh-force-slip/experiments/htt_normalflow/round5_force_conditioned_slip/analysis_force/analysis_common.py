from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_receipt_inventory(summary_path: Path, inventory_path: Path) -> dict:
    """Bind a formal result to exactly one frozen scheduler job."""
    summary_path=summary_path.resolve();inventory_path=inventory_path.resolve()
    receipt_path=summary_path.with_name(summary_path.name+".scheduler_receipt.json")
    if not receipt_path.is_file():raise ValueError("formal result lacks scheduler receipt")
    receipt=json.loads(receipt_path.read_text());inventory=json.loads(inventory_path.read_text())
    matches=[job for job in inventory.get("jobs",[]) if Path(job.get("acceptance_path","")).resolve()==summary_path]
    if len(matches)!=1 or receipt.get("argv")!=matches[0].get("argv"):
        raise ValueError("scheduler receipt is not bound to exactly one frozen inventory job")
    for raw,digest in receipt.get("input_sha256",{}).items():
        path=Path(raw)
        if not path.is_file() or sha256(path)!=digest:raise ValueError("scheduler receipt input drift")
    return {"scheduler_receipt_sha256":sha256(receipt_path),"inventory_sha256":sha256(inventory_path)}


def protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def invariant_force(values: np.ndarray) -> dict[str, np.ndarray]:
    force = np.asarray(values, dtype=np.float64)
    if force.ndim != 2 or force.shape[1] != 3 or not np.isfinite(force).all():
        raise ValueError("force must be finite [N,3]")
    return {
        "Fn": np.abs(force[:, 2]),
        "Ft": np.linalg.norm(force[:, :2], axis=1),
        "Fmag": np.linalg.norm(force, axis=1),
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64); p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 3 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("target and prediction must be aligned finite [N,3]")
    error = p - y
    axis = {
        "bias_n": error.mean(0).tolist(),
        "mae_n": np.abs(error).mean(0).tolist(),
        "rmse_n": np.sqrt(np.square(error).mean(0)).tolist(),
    }
    invariant = {}
    yi, pi = invariant_force(y), invariant_force(p)
    for key in yi:
        delta = pi[key] - yi[key]
        invariant[key] = {"bias_n": float(delta.mean()), "mae_n": float(np.abs(delta).mean()),
                          "rmse_n": float(np.sqrt(np.square(delta).mean()))}
    return {"count": len(y), "axis": axis, "invariant": invariant}


def lagged_change_metrics(episode: np.ndarray, frames: np.ndarray, target: np.ndarray,
                          prediction: np.ndarray, lag: int = 5) -> dict[str, Any]:
    lookup = {(str(ep), int(t)): i for i, (ep, t) in enumerate(zip(episode, frames))}
    current, previous = [], []
    for i, (ep, t) in enumerate(zip(episode, frames)):
        j = lookup.get((str(ep), int(t) - lag))
        if j is not None:
            current.append(i); previous.append(j)
    if not current:
        raise ValueError("no exact lagged pairs")
    current = np.asarray(current); previous = np.asarray(previous)
    return regression_metrics(target[current] - target[previous], prediction[current] - prediction[previous])


def circular_mismatch(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if len(array) < 2:
        raise ValueError("mismatch requires at least two rows")
    return np.roll(array, len(array) // 2, axis=0)


def cluster_force_bootstrap(groups: np.ndarray, target: np.ndarray, prediction: np.ndarray,
                            train_mean: np.ndarray, repetitions: int = 200,
                            seed: int = 20260915) -> dict[str, Any]:
    """Complete leakage-group bootstrap for aggregate force RMSE contrasts."""
    group_array=np.asarray(groups);names=sorted(set(group_array.tolist()))
    if len(names)<2:raise ValueError("force bootstrap requires at least two leakage groups")
    indexes={name:np.flatnonzero(group_array==name) for name in names};rng=np.random.default_rng(seed);rows=[]
    for _ in range(repetitions):
        chosen=rng.choice(names,size=len(names),replace=True);idx=np.concatenate([indexes[name] for name in chosen])
        model=regression_metrics(target[idx],prediction[idx]);constant=regression_metrics(target[idx],np.broadcast_to(train_mean,target.shape)[idx]);zero=regression_metrics(target[idx],np.zeros_like(target)[idx])
        row={}
        for j,name in enumerate(("shear_x","shear_y","normal")):
            row[f"model_axis_{name}_rmse_n"]=model["axis"]["rmse_n"][j]
            row[f"improvement_over_train_mean_axis_{name}_rmse_n"]=constant["axis"]["rmse_n"][j]-model["axis"]["rmse_n"][j]
            row[f"improvement_over_zero_axis_{name}_rmse_n"]=zero["axis"]["rmse_n"][j]-model["axis"]["rmse_n"][j]
        for name in ("Fn","Ft","Fmag"):
            row[f"model_{name}_rmse_n"]=model["invariant"][name]["rmse_n"]
            row[f"improvement_over_train_mean_{name}_rmse_n"]=constant["invariant"][name]["rmse_n"]-model["invariant"][name]["rmse_n"]
            row[f"improvement_over_zero_{name}_rmse_n"]=zero["invariant"][name]["rmse_n"]-model["invariant"][name]["rmse_n"]
        rows.append(row)
    return {"unit":"complete schema-2 leakage_group","groups":len(names),"repetitions":repetitions,
            "intervals_95":{key:{"lower":float(np.quantile([r[key] for r in rows],.025)),"upper":float(np.quantile([r[key] for r in rows],.975))} for key in rows[0]}}
