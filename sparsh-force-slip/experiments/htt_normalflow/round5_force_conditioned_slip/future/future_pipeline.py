#!/usr/bin/env python3
"""Round-5 future cache and training entry points.

This module is deliberately self-contained so source-cache conversion can run on
CPU without importing the Sparsh stack. Formal training is guarded by protocol
hash and upstream provenance checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
SOURCE_VARIANTS = ("z_p_slip", "z_p_slip_force", "z_p_slip_force_pred_delta")
HTT_VARIANTS = ("risk", "base", "full_state")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def protocol_sha256() -> str:
    return sha256(PROTOCOL_PATH)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def identity_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def code_bundle_sha256() -> str:
    return identity_hash({"future_pipeline.py": sha256(Path(__file__)), "protocol.json": protocol_sha256()})


def episode_key(meta: dict[str, Any]) -> tuple[str, str]:
    return str(meta["dataset"]), str(meta["trajectory"])


def build_source_cache(input_path: Path, output_path: Path, split: str) -> dict[str, Any]:
    """Convert the legacy cache to exact predicted-force-delta common samples."""
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if split not in {"train", "val", "validation"}:
        raise ValueError("source bridge permits train/validation only")
    expected = {"z", "future_slip", "force_pred_n", "slip_probs", "metadata", "horizons"}
    missing = expected - payload.keys()
    if missing:
        raise KeyError(f"legacy cache missing {sorted(missing)}")
    n = len(payload["metadata"])
    if any(len(payload[k]) != n for k in ("z", "future_slip", "force_pred_n", "slip_probs")):
        raise ValueError("legacy cache arrays and metadata have inconsistent lengths")
    lookup: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for i, m in enumerate(payload["metadata"]):
        sample = int(m["sample"])
        key = episode_key(m)
        if sample in lookup[key]:
            raise ValueError(f"duplicate source sample {key}/{sample}")
        lookup[key][sample] = i
    keep: list[int] = []
    previous: list[int] = []
    rejected_missing_exact_lag = 0
    for i, m in enumerate(payload["metadata"]):
        t = int(m["sample"])
        if t < 10:
            continue
        episode = lookup[episode_key(m)]
        j = episode.get(t - 5)
        if j is None or (t - 10) not in episode:
            rejected_missing_exact_lag += 1
            continue
        keep.append(i)
        previous.append(j)
    if not keep:
        raise ValueError("no source samples satisfy exact t,t-5 and t>=10")
    idx = torch.tensor(keep, dtype=torch.long)
    prev = torch.tensor(previous, dtype=torch.long)
    force = payload["force_pred_n"].float()
    pred_delta = force[idx] - force[prev]
    if not torch.isfinite(pred_delta).all():
        raise ValueError("non-finite predicted force delta")
    z = payload["z"][idx]
    fp = force[idx]
    fn = fp[:, 2].abs()
    ft = torch.linalg.vector_norm(fp[:, :2], dim=1)
    ratio = ft / (fn + 1.0e-6)
    p_slip = payload["slip_probs"][idx, 1].float()
    out = {
        "schema": "round5_source_pred_delta_v1",
        "split": "val" if split in {"val", "validation"} else split,
        "source_path": str(input_path),
        "source_sha256": sha256(input_path),
        "protocol_sha256": protocol_sha256(),
        "horizons": [int(x) for x in payload["horizons"]],
        "z": z,
        "p_slip": p_slip,
        "force_features": torch.stack((fn, ft, ratio), dim=1),
        "predicted_delta_force_xyz": pred_delta,
        "future_slip": payload["future_slip"][idx].float(),
        "metadata": [payload["metadata"][i] for i in keep],
        "audit": {
            "input_samples": n,
            "common_samples": len(keep),
            "minimum_sample": min(int(payload["metadata"][i]["sample"]) for i in keep),
            "rejected_missing_exact_lag": rejected_missing_exact_lag,
            "old_delta_source": "ground_truth batch delta_force; intentionally unused",
            "new_delta_source": "force_pred_n[t]-force_pred_n[t-5]",
            "complete_recursive_input": "t>=10 and exact t-5 and t-10 rows; no interpolation",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(out, tmp)
    os.replace(tmp, output_path)
    summary = {k: out[k] for k in ("schema", "split", "source_path", "source_sha256", "protocol_sha256", "horizons", "audit")}
    summary["output_path"] = str(output_path)
    summary["output_sha256"] = sha256(output_path)
    atomic_json(output_path.with_suffix(".manifest.json"), summary)
    return summary


def source_input(payload: dict[str, Any], variant: str) -> torch.Tensor:
    if variant not in SOURCE_VARIANTS:
        raise ValueError(variant)
    parts = [payload["z"].float(), payload["p_slip"].float()[:, None]]
    if variant != "z_p_slip":
        parts.append(payload["force_features"].float())
    if variant == "z_p_slip_force_pred_delta":
        parts.append(payload["predicted_delta_force_xyz"].float())
    return torch.cat(parts, dim=1)


class LegacyFutureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, out_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def canonical_htt_payload(path: Path, require_formal: bool = True) -> dict[str, Any]:
    p = torch.load(path, map_location="cpu", weights_only=False)
    required = {"schema", "fold", "upstream_seed", "upstream_force", "upstream_slip", "condition_normalization", "formal", "episodes"}
    missing = required - p.keys()
    if missing:
        raise KeyError(f"HTT upstream missing {sorted(missing)}")
    if p["schema"] != "round5_future_upstream_v1":
        raise ValueError("wrong HTT upstream schema")
    if p["fold"] != "htt_leave_p1" or int(p["upstream_seed"]) != 20260914:
        raise ValueError("HTT upstream fold/seed violates preregistration")
    if p["upstream_force"] != "F-adapt" or p["upstream_slip"] != "R5-F-adapt-conditioned-slip":
        raise ValueError("HTT upstream model identities violate preregistration")
    if require_formal and not bool(p["formal"]):
        raise ValueError("smoke/non-formal upstream cannot feed a formal run")
    epsilon = float(p["condition_normalization"].get("ratio_epsilon_n", float("nan")))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid upstream ratio_epsilon_n")
    return p


def _episode_arrays(ep: dict[str, Any]) -> dict[str, Any]:
    required = {"episode_id", "role", "t", "z", "p_slip", "base_p_slip", "force_pred_n", "stage"}
    missing = required - ep.keys()
    if missing:
        raise KeyError(f"episode missing {sorted(missing)}")
    t = np.asarray(ep["t"], dtype=np.int64)
    if len(set(t.tolist())) != len(t):
        raise ValueError(f"duplicate t in {ep['episode_id']}")
    order = np.argsort(t)
    out = {"episode_id": str(ep["episode_id"]), "role": str(ep["role"]), "t": t[order]}
    for k in ("z", "p_slip", "base_p_slip", "force_pred_n", "stage"):
        a = torch.as_tensor(ep[k])
        if len(a) != len(t):
            raise ValueError(f"{ep['episode_id']} length mismatch for {k}")
        out[k] = a[torch.as_tensor(order, dtype=torch.long)].float()
    return out


def support_payload_from_contract(path: Path, fold: str = "htt_leave_p1") -> dict[str, Any]:
    """Load only authoritative slip labels/roles needed for the train support gate."""
    c = json.loads(path.read_text(encoding="utf-8"))
    if c.get("format") != "round5_htt_mae_tokens_targets_v1":
        raise ValueError("wrong HTT data contract format")
    sp = c.get("slip_cache_provenance", {})
    slip_entries = [e for e in c.get("entries", []) if e.get("task") == "slip"]
    if sp.get("status") != "complete" or int(sp.get("entries", -1)) != len(slip_entries):
        raise ValueError("slip portion of data contract is incomplete")
    episodes = []
    for e in slip_entries:
        role = e.get("roles_by_fold", {}).get(fold)
        if role != "train":
            continue
        lp = Path(e["label_path"])
        if sha256(lp) != e.get("label_sha256"):
            raise ValueError(f"label SHA mismatch: {lp}")
        labels = np.load(lp, allow_pickle=False)
        frames = int(e["frames"])
        if labels.shape[0] != frames:
            raise ValueError(f"label/frame mismatch: {e['episode_id']}")
        episodes.append({"episode_id": e["episode_id"], "role": "train", "t": np.arange(frames), "stage": labels})
    return {"episodes": episodes, "contract_path": str(path), "contract_sha256": sha256(path), "fold": fold}


def assemble_htt_upstream(contract_path: Path, prediction_manifest_path: Path, output_path: Path, allow_smoke: bool = False) -> dict[str, Any]:
    """Join immutable token/label evidence with the current module's predictions."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    pred = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    if contract.get("format") != "round5_htt_mae_tokens_targets_v1" or contract.get("status") != "complete":
        raise ValueError("assembler requires the complete Round5 HTT contract")
    if pred.get("format") != "round5_fadapt_contact_predictions_v1" or pred.get("status") != "complete":
        raise ValueError("wrong or incomplete contact prediction manifest")
    if pred.get("fold") != "htt_leave_p1" or int(pred.get("seed", -1)) != 20260914:
        raise ValueError("prediction fold/seed violates preregistration")
    if not bool(pred.get("formal")) and not allow_smoke:
        raise ValueError("formal assembler refuses smoke predictions")
    contract_sha = sha256(contract_path)
    if pred.get("token_cache_sha256") != contract_sha:
        raise ValueError("prediction token-cache SHA does not match contract")
    parent_specs = (
        ("force_prediction_manifest", "force_prediction_manifest_sha256"),
        ("fused_slip_checkpoint", "fused_slip_checkpoint_sha256"),
        ("base_slip_checkpoint", "base_slip_checkpoint_sha256"),
    )
    parents = {}
    for path_key, hash_key in parent_specs:
        parent_path = Path(pred.get(path_key, ""))
        if not parent_path.is_file() or sha256(parent_path) != pred.get(hash_key):
            raise ValueError(f"missing or mismatched upstream parent: {path_key}")
        parents[path_key] = parent_path
    force_parent = json.loads(parents["force_prediction_manifest"].read_text())
    if (force_parent.get("status") != "complete" or force_parent.get("format") != "round5_force_predictions_v1"
            or force_parent.get("variant") != "adapt" or force_parent.get("fold") != "htt_leave_p1"
            or int(force_parent.get("seed", -1)) != 20260914
            or bool(force_parent.get("formal")) != bool(pred.get("formal"))):
        raise ValueError("force-prediction parent identity/formality mismatch")
    fused_parent = torch.load(parents["fused_slip_checkpoint"], map_location="cpu", weights_only=False)
    fused_config = fused_parent.get("config", {})
    if (fused_parent.get("format") != "round5_force_conditioned_slip_v1"
            or fused_config.get("variant") != "F-adapt" or fused_config.get("fold") != "htt_leave_p1"
            or int(fused_config.get("seed", -1)) != 20260914
            or identity_hash(fused_config) != pred.get("fused_slip_config_sha256")
            or bool(fused_config.get("smoke")) == bool(pred.get("formal"))):
        raise ValueError("fused-slip parent identity/config/formality mismatch")
    base_parent = torch.load(parents["base_slip_checkpoint"], map_location="cpu", weights_only=False)
    base_config = base_parent.get("config", {})
    if (base_parent.get("format") != "round3_mae_slip_branch_v1" or base_config.get("init") != "fresh"
            or base_config.get("fold") != "htt_leave_p1" or int(base_config.get("seed", -1)) != 20260914
            or (bool(base_config.get("smoke")) and not allow_smoke)):
        raise ValueError("base Round3-B parent identity/formality mismatch")
    by_id = {e["episode_id"]: e for e in contract.get("entries", []) if e.get("task") == "slip"}
    expected = {eid for eid,e in by_id.items() if e["roles_by_fold"]["htt_leave_p1"] != "test"}
    pred_entries = {e["episode_id"]: e for e in pred.get("entries", [])}
    if set(pred_entries) != expected:
        raise ValueError(f"prediction episode identity mismatch: missing={sorted(expected-set(pred_entries))[:3]}, extra={sorted(set(pred_entries)-expected)[:3]}")
    episodes=[]
    for eid in sorted(expected):
        c, pe = by_id[eid], pred_entries[eid]
        role = c["roles_by_fold"]["htt_leave_p1"]
        if role == "test":
            raise AssertionError("test role reached assembler")
        if pe.get("roles_by_fold") != c.get("roles_by_fold") or int(pe["frames"]) != int(c["frames"]):
            raise ValueError(f"role/frame mismatch: {eid}")
        token_path, label_path, pp = Path(c["token_path"]), Path(c["label_path"]), Path(pe["path"])
        for path, expected_sha in ((token_path,c["token_sha256"]),(label_path,c["label_sha256"]),(pp,pe["sha256"])):
            if sha256(path) != expected_sha: raise ValueError(f"artifact SHA mismatch: {path}")
        tokens=np.load(token_path,mmap_mode="r",allow_pickle=False); labels=np.load(label_path,allow_pickle=False)
        with np.load(pp,allow_pickle=False) as q:
            needed={"frame_index","force_pred_n","p_slip","base_p_slip"}
            if needed-set(q.files): raise KeyError(f"prediction episode missing {sorted(needed-set(q.files))}")
            t=np.asarray(q["frame_index"],dtype=np.int64); force=np.asarray(q["force_pred_n"],dtype=np.float32); p_slip=np.asarray(q["p_slip"],dtype=np.float32); base=np.asarray(q["base_p_slip"],dtype=np.float32)
        frames=int(c["frames"])
        if tokens.shape[0]!=frames or labels.shape[0]!=frames or force.shape!=(frames,3) or p_slip.shape!=(frames,) or base.shape!=(frames,) or not np.array_equal(t,np.arange(frames)):
            raise ValueError(f"assembled array shape/time mismatch: {eid}")
        if not all(np.isfinite(a).all() for a in (tokens,force,p_slip,base)) or np.any((p_slip<0)|(p_slip>1)) or np.any((base<0)|(base>1)):
            raise ValueError(f"nonfinite/out-of-range upstream values: {eid}")
        z=np.asarray(tokens,dtype=np.float32).mean(axis=1).astype(np.float16)
        episodes.append({"episode_id":eid,"role":role,"t":torch.from_numpy(t),"z":torch.from_numpy(z),"p_slip":torch.from_numpy(p_slip),"base_p_slip":torch.from_numpy(base),"force_pred_n":torch.from_numpy(force),"stage":torch.from_numpy(np.asarray(labels))})
    condition_normalization = pred.get("condition_normalization")
    epsilon = float((condition_normalization or {}).get("ratio_epsilon_n", float("nan")))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("prediction manifest lacks valid train-only ratio_epsilon_n")
    out={"schema":"round5_future_upstream_v1","formal":bool(pred["formal"]),"fold":"htt_leave_p1","upstream_seed":20260914,"upstream_force":"F-adapt","upstream_slip":"R5-F-adapt-conditioned-slip","condition_normalization":condition_normalization,"contract_path":str(contract_path),"contract_sha256":contract_sha,"prediction_manifest_path":str(prediction_manifest_path),"prediction_manifest_sha256":sha256(prediction_manifest_path),"token_pooling":"mean over complete frozen MAE tokens; stored float16","roles_included":sorted({e["role"] for e in episodes}),"test_role_consumed":False,"episodes":episodes}
    output_path.parent.mkdir(parents=True,exist_ok=True);tmp=output_path.with_suffix(output_path.suffix+".tmp");torch.save(out,tmp);os.replace(tmp,output_path)
    manifest={k:v for k,v in out.items() if k!="episodes"};manifest.update({"status":"complete","smoke":not bool(out["formal"]),"episode_count":len(episodes),"output_path":str(output_path),"output_sha256":sha256(output_path)})
    atomic_json(output_path.with_suffix(".manifest.json"),manifest);return manifest


def htt_support_audit(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = protocol()["htt"]
    candidates = [int(x) for x in cfg["candidate_horizons"]]
    counts = {h: {"positive_frames": 0, "negative_frames": 0, "positive_onset_episodes": set(), "negative_episodes": set()} for h in candidates}
    for raw in payload["episodes"]:
        required = {"episode_id", "role", "t", "stage"}
        if required - raw.keys():
            raise KeyError(f"support episode missing {sorted(required - raw.keys())}")
        order = np.argsort(np.asarray(raw["t"], dtype=np.int64))
        ep = {"episode_id": str(raw["episode_id"]), "role": str(raw["role"]), "t": np.asarray(raw["t"], dtype=np.int64)[order], "stage": torch.as_tensor(raw["stage"])[torch.as_tensor(order)].float()}
        if ep["role"] != "train":
            continue
        ts = ep["t"].astype(int)
        stage = ep["stage"].numpy().astype(int)
        pos = {int(t): i for i, t in enumerate(ts)}
        eid = ep["episode_id"]
        gross_times = ts[stage == 2]
        first_gross = int(gross_times.min()) if len(gross_times) else None
        for h in candidates:
            for t in ts:
                t = int(t)
                # Four exact history endpoints; each endpoint has recursive force input.
                if t < 13 or any(any((t - j - lag) not in pos for lag in (0, 5, 10)) for j in range(4)):
                    continue
                if any((t + j) not in pos for j in range(1, h + 1)):
                    continue
                if stage[pos[t]] != 0:
                    continue
                # The task ends at the first gross onset. Static-looking recovery
                # frames after it are never valid negatives.
                if first_gross is not None and t >= first_gross:
                    continue
                positive = first_gross is not None and t < first_gross <= t + h
                key = "positive_frames" if positive else "negative_frames"
                counts[h][key] += 1
                if positive:
                    counts[h]["positive_onset_episodes"].add(eid)
                else:
                    counts[h]["negative_episodes"].add(eid)
    rows = []
    gate = cfg["support_gate"]
    chosen = None
    for h in candidates:
        c = counts[h]
        row = {
            "horizon": h,
            "positive_frames": c["positive_frames"],
            "positive_onset_episodes": len(c["positive_onset_episodes"]),
            "negative_frames": c["negative_frames"],
            "negative_episodes": len(c["negative_episodes"]),
        }
        row["supported"] = all(row[k] >= int(gate[k]) for k in ("positive_frames", "positive_onset_episodes", "negative_frames", "negative_episodes"))
        rows.append(row)
        if chosen is None and row["supported"]:
            chosen = h
    return {
        "schema": "round5_htt_future_support_v1",
        "protocol_sha256": protocol_sha256(),
        "train_only": True,
        "minimum_history_endpoint": 13,
        "no_padding_or_interpolation": True,
        "candidates": rows,
        "selected_horizon": chosen,
        "status": "supported" if chosen is not None else "unsupported_no_training",
    }


def _htt_examples(payload: dict[str, Any], horizon: int, role: str, variant: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys, residuals = [], [], []
    epsilon = float(payload["condition_normalization"]["ratio_epsilon_n"])
    for raw in payload["episodes"]:
        ep = _episode_arrays(raw)
        if ep["role"] != role:
            continue
        pos = {int(t): i for i, t in enumerate(ep["t"].tolist())}
        gross_times = [int(t) for t in ep["t"].tolist() if int(ep["stage"][pos[int(t)]].item()) == 2]
        first_gross = min(gross_times) if gross_times else None
        for t0 in ep["t"].tolist():
            t = int(t0)
            if t < 13 or any(any((t-j-lag) not in pos for lag in (0,5,10)) for j in range(4)) or any((t+j) not in pos for j in range(1, horizon+1)):
                continue
            if int(ep["stage"][pos[t]].item()) != 0:
                continue
            if first_gross is not None and t >= first_gross:
                continue
            seq=[]
            for cur in range(t-3, t+1):
                i, j = pos[cur], pos[cur-5]
                z = ep["z"][i].flatten()
                ps = ep["p_slip"][i].reshape(1)
                step=[z,ps]
                if variant in {"base","full_state"}:
                    f, fp = ep["force_pred_n"][i], ep["force_pred_n"][j]
                    fn, ft = f[2].abs(), torch.linalg.vector_norm(f[:2])
                    fnp, ftp = fp[2].abs(), torch.linalg.vector_norm(fp[:2])
                    step.append(torch.stack((fn,ft,ft/(fn+epsilon),fn-fnp,ft-ftp)))
                seq.append(torch.cat(step))
            xs.append(torch.stack(seq))
            ys.append(float(first_gross is not None and t < first_gross <= t+horizon))
            current_z = ep["z"][pos[t]].flatten()
            residuals.append(torch.cat([ep["z"][pos[t+delta]].flatten()-current_z for delta in (1,3,horizon)]))
    if not xs:
        raise ValueError(f"no HTT examples for role={role}, horizon={horizon}")
    return torch.stack(xs), torch.tensor(ys), torch.stack(residuals)


class GRURisk(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, full_state: bool, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.risk = nn.Linear(hidden_dim, 1)
        self.state = nn.Linear(hidden_dim, z_dim) if full_state else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.gru(x)[0][:, -1]
        return self.risk(h).squeeze(1), None if self.state is None else self.state(h)


class TensorTriplet(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, r: torch.Tensor): self.x,self.y,self.r=x,y,r
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i],self.y[i],self.r[i]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def set_epoch_seed(seed: int, epoch: int) -> None:
    """Make an epoch independent of prior process/RNG history for exact resume."""
    value = int(seed) + 1009 * int(epoch)
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def _average_precision(y: torch.Tensor, p: torch.Tensor) -> float:
    yy=y.detach().cpu().numpy(); pp=p.detach().cpu().numpy()
    if int(yy.sum()) == 0: return 0.0
    return float(average_precision_score(yy,pp))


def _finite_step(loss: torch.Tensor, model: nn.Module) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("missing or non-finite training gradient")


def _parameters_changed(before: dict[str, torch.Tensor], model: nn.Module) -> bool:
    return any(not torch.equal(before[k], v.detach().cpu()) for k,v in model.state_dict().items())


def train_source(train_path: Path, val_path: Path, out: Path, variant: str, seed: int, device: str, epochs: int | None = None, resume: bool = False, interrupt_after_epoch: int | None = None) -> dict[str, Any]:
    cfg=protocol()["source_bridge"]; set_seed(seed)
    tr=torch.load(train_path,map_location="cpu",weights_only=False); va=torch.load(val_path,map_location="cpu",weights_only=False)
    if tr.get("schema")!="round5_source_pred_delta_v1" or va.get("schema")!="round5_source_pred_delta_v1": raise ValueError("wrong source cache schema")
    if tr.get("protocol_sha256")!=protocol_sha256() or va.get("protocol_sha256")!=protocol_sha256(): raise ValueError("source cache protocol SHA mismatch")
    total_epochs=epochs or cfg["max_epochs"]
    execution_mode="smoke" if epochs is not None else "formal"
    run_config={"domain":"source","variant":variant,"seed":seed,"train_cache_sha256":sha256(train_path),"val_cache_sha256":sha256(val_path),"protocol_sha256":protocol_sha256(),"code_bundle_sha256":code_bundle_sha256(),"model":"legacy_mlp","model_logit_semantics":"stable_no_future_slip","reported_probability_semantics":"future_any_slip=1-sigmoid(stable_logit)","hidden_dim":cfg["hidden_dim"],"dropout":cfg["dropout"],"optimizer":"AdamW","learning_rate":cfg["learning_rate"],"weight_decay":cfg["weight_decay"],"batch_size":cfg["batch_size"],"max_epochs":cfg["max_epochs"],"execution_mode":execution_mode,"effective_epoch_budget":total_epochs}
    run_id=identity_hash(run_config)
    xtr,xva=source_input(tr,variant),source_input(va,variant); ytr=1-tr["future_slip"].float(); yva=va["future_slip"].float()
    model=LegacyFutureMLP(xtr.shape[1],cfg["hidden_dim"],len(cfg["horizons"]),cfg["dropout"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    dataset=torch.utils.data.TensorDataset(xtr,ytr)
    best=(-math.inf,None); history=[]; start_epoch=1
    latest=out/"latest.pth"
    if not resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {out}")
    if resume:
        if not latest.exists(): raise ValueError(f"resume requested but missing {latest}")
        state=torch.load(latest,map_location="cpu",weights_only=False)
        if state.get("run_identity_sha256")!=run_id: raise ValueError("resume checkpoint cache/config identity mismatch")
        model.load_state_dict(state["model_state"]);opt.load_state_dict(state["optimizer_state"])
        best=(float(state["best_score"]),state["best_model_state"]);history=list(state["history"]);start_epoch=int(state["epoch"])+1
    out.mkdir(parents=True,exist_ok=True)
    update_observed=False
    for epoch in range(start_epoch,total_epochs+1):
        set_epoch_seed(seed,epoch)
        generator=torch.Generator().manual_seed(seed+7919*epoch)
        loader=DataLoader(dataset,batch_size=cfg["batch_size"],shuffle=True,generator=generator)
        model.train(); losses=[];before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for x,y_stable in loader:
            x,y_stable=x.to(device),y_stable.to(device); loss=nn.functional.binary_cross_entropy_with_logits(model(x),y_stable)
            opt.zero_grad();_finite_step(loss,model);opt.step();losses.append(float(loss))
        update_observed=update_observed or _parameters_changed(before,model)
        if not update_observed: raise RuntimeError("source parameters did not update")
        model.eval()
        with torch.no_grad(): p=1-torch.sigmoid(model(xva.to(device))).cpu()
        if not torch.isfinite(p).all(): raise FloatingPointError("non-finite source validation prediction")
        ap=_average_precision(yva[:,0],p[:,0]); history.append({"epoch":epoch,"train_loss":sum(losses)/len(losses),"validation_H1_AP":ap})
        if ap>best[0]: best=(ap,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        atomic_torch_save(latest,{"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"epoch":epoch,"variant":variant,"seed":seed,"protocol_sha256":protocol_sha256(),"run_identity_sha256":run_id,"run_config":run_config,"best_score":best[0],"best_model_state":best[1],"history":history,"checks":{"finite_loss_and_gradients":True,"parameter_update_observed":update_observed}})
        if interrupt_after_epoch is not None and epoch == interrupt_after_epoch:
            summary={"status":"interrupted","formal":False,"smoke":epochs is not None,"variant":variant,"seed":seed,"run_config":run_config,"run_identity_sha256":run_id,"interrupted_after_epoch":epoch,"target_epochs":total_epochs,"checks":{"finite_loss_and_gradients":True,"parameter_update_observed":update_observed},"artifacts":{"latest":{"path":str(latest),"sha256":sha256(latest)}},"history":history}
            atomic_json(out/"summary.json",summary);return summary
    best_path=out/"best.pth";atomic_torch_save(best_path,{"model_state":best[1],"variant":variant,"seed":seed,"protocol_sha256":protocol_sha256(),"run_identity_sha256":run_id,"run_config":run_config,"best_validation_H1_AP":best[0]})
    restored=torch.load(latest,map_location="cpu",weights_only=False);roundtrip=restored["run_identity_sha256"]==run_id and all(torch.equal(restored["model_state"][k].cpu(),v.detach().cpu()) for k,v in model.state_dict().items())
    if not roundtrip: raise RuntimeError("source checkpoint round-trip mismatch")
    atomic_json(out/"summary.json",{"status":"complete","formal":epochs is None,"smoke":epochs is not None,"variant":variant,"seed":seed,"run_config":run_config,"run_identity_sha256":run_id,"checks":{"finite_loss_and_gradients":True,"parameter_update_observed":update_observed,"checkpoint_roundtrip_exact":roundtrip},"artifacts":{"best":{"path":str(best_path),"sha256":sha256(best_path)},"latest":{"path":str(latest),"sha256":sha256(latest)}},"best_validation_H1_AP":best[0],"history":history,"common_train_n":len(xtr),"common_val_n":len(xva)})
    return json.loads((out/"summary.json").read_text())


def train_htt(upstream: Path, support_path: Path, out: Path, variant: str, seed: int, device: str, epochs: int | None = None, require_formal: bool = True, resume: bool = False, interrupt_after_epoch: int | None = None) -> dict[str, Any]:
    cfg=protocol()["htt"]; set_seed(seed); p=canonical_htt_payload(upstream,require_formal=require_formal)
    audit=json.loads(support_path.read_text()); h=audit.get("selected_horizon")
    if audit.get("protocol_sha256")!=protocol_sha256() or h is None: raise ValueError("support audit is stale or unsupported")
    xtr,ytr,rtr=_htt_examples(p,h,"train",variant); xva,yva,rva=_htt_examples(p,h,"validation",variant)
    mean=xtr.mean((0,1),keepdim=True); std=xtr.std((0,1),keepdim=True).clamp_min(1e-6); xtr=(xtr-mean)/std; xva=(xva-mean)/std
    residual_mean=rtr.mean(0,keepdim=True); residual_std=rtr.std(0,keepdim=True).clamp_min(1e-6)
    rtr=(rtr-residual_mean)/residual_std; rva=(rva-residual_mean)/residual_std
    model=GRURisk(xtr.shape[2],rtr.shape[1],variant=="full_state",cfg["hidden_dim"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    pos=max(float(ytr.sum()),1.); neg=max(float(len(ytr)-ytr.sum()),1.); pos_weight=torch.tensor(neg/pos,device=device)
    dataset=TensorTriplet(xtr,ytr,rtr)
    total_epochs=epochs or cfg["max_epochs"]
    execution_mode="formal" if epochs is None and require_formal else "smoke"
    run_config={"domain":"htt","variant":variant,"seed":seed,"upstream_sha256":sha256(upstream),"support_audit_sha256":sha256(support_path),"protocol_sha256":protocol_sha256(),"code_bundle_sha256":code_bundle_sha256(),"horizon":h,"model":"single_layer_gru","hidden_dim":cfg["hidden_dim"],"state_auxiliary_weight":cfg["state_auxiliary_weight"],"state_prediction_horizons":[1,3,h],"state_output_dim":int(rtr.shape[1]),"force_ratio_epsilon_n":float(p["condition_normalization"]["ratio_epsilon_n"]),"optimizer":"AdamW","learning_rate":cfg["learning_rate"],"weight_decay":cfg["weight_decay"],"batch_size":cfg["batch_size"],"max_epochs":cfg["max_epochs"],"patience":cfg["patience"],"execution_mode":execution_mode,"effective_epoch_budget":total_epochs}
    run_id=identity_hash(run_config);best=(math.inf,None,0);stale=0;history=[];start_epoch=1;latest=out/"latest.pth";out.mkdir(parents=True,exist_ok=True)
    if not resume and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {out}")
    if resume:
        if not latest.exists(): raise ValueError(f"resume requested but missing {latest}")
        state=torch.load(latest,map_location="cpu",weights_only=False)
        if state.get("run_identity_sha256")!=run_id: raise ValueError("resume checkpoint cache/config identity mismatch")
        model.load_state_dict(state["model_state"]);opt.load_state_dict(state["optimizer_state"]);best=(float(state["best_score"]),state["best_model_state"],int(state["best_epoch"]));stale=int(state["stale"]);history=list(state["history"]);start_epoch=int(state["epoch"])+1
    update_observed=False
    for epoch in range(start_epoch,total_epochs+1):
        set_epoch_seed(seed,epoch);loader=DataLoader(dataset,batch_size=cfg["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(seed+7919*epoch))
        model.train(); losses=[];before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for x,y,r in loader:
            x,y,r=x.to(device),y.to(device),r.to(device); logit,rhat=model(x)
            risk=nn.functional.binary_cross_entropy_with_logits(logit,y,pos_weight=pos_weight); loss=risk
            if rhat is not None: loss=loss+cfg["state_auxiliary_weight"]*nn.functional.mse_loss(rhat,r)
            opt.zero_grad();_finite_step(loss,model);opt.step();losses.append(float(loss))
        update_observed=update_observed or _parameters_changed(before,model)
        if not update_observed: raise RuntimeError("HTT parameters did not update")
        model.eval()
        with torch.no_grad(): lv,_=model(xva.to(device)); vbce=float(nn.functional.binary_cross_entropy_with_logits(lv,yva.to(device)))
        if not math.isfinite(vbce): raise FloatingPointError("non-finite HTT validation loss")
        history.append({"epoch":epoch,"train_loss":sum(losses)/len(losses),"validation_risk_BCE":vbce})
        if vbce<best[0]: best=(vbce,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch); stale=0
        else: stale+=1
        atomic_torch_save(latest,{"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),"epoch":epoch,"variant":variant,"seed":seed,"protocol_sha256":protocol_sha256(),"run_identity_sha256":run_id,"run_config":run_config,"best_score":best[0],"best_model_state":best[1],"best_epoch":best[2],"stale":stale,"history":history,"normalization":{"mean":mean,"std":std,"state_residual_mean":residual_mean,"state_residual_std":residual_std},"checks":{"finite_loss_and_gradients":True,"parameter_update_observed":update_observed}})
        if interrupt_after_epoch is not None and epoch == interrupt_after_epoch:
            summary={"status":"interrupted","formal":False,"smoke":True,"variant":variant,"seed":seed,"horizon":h,"run_config":run_config,"run_identity_sha256":run_id,"interrupted_after_epoch":epoch,"target_epochs":total_epochs,"checks":{"finite_loss_and_gradients":True,"parameter_update_observed":update_observed},"artifacts":{"latest":{"path":str(latest),"sha256":sha256(latest)}},"history":history}
            atomic_json(out/"summary.json",summary);return summary
        if stale>=cfg["patience"]: break
    best_path=out/"best.pth";atomic_torch_save(best_path,{"model_state":best[1],"variant":variant,"seed":seed,"horizon":h,"normalization":{"mean":mean,"std":std,"state_residual_mean":residual_mean,"state_residual_std":residual_std},"protocol_sha256":protocol_sha256(),"run_identity_sha256":run_id,"run_config":run_config})
    restored=torch.load(latest,map_location="cpu",weights_only=False);roundtrip=restored["run_identity_sha256"]==run_id and all(torch.equal(restored["model_state"][k].cpu(),v.detach().cpu()) for k,v in model.state_dict().items())
    if not roundtrip: raise RuntimeError("HTT checkpoint round-trip mismatch")
    atomic_json(out/"summary.json",{"status":"complete","formal":epochs is None and require_formal,"smoke":epochs is not None or not require_formal,"variant":variant,"seed":seed,"horizon":h,"run_config":run_config,"run_identity_sha256":run_id,"checks":{"finite_loss_and_gradients":True,"parameter_update_observed":update_observed,"checkpoint_roundtrip_exact":roundtrip},"artifacts":{"best":{"path":str(best_path),"sha256":sha256(best_path)},"latest":{"path":str(latest),"sha256":sha256(latest)}},"best_epoch":best[2],"best_validation_risk_BCE":best[0],"train_n":len(ytr),"validation_n":len(yva),"history":history})
    return json.loads((out/"summary.json").read_text())


def inventory(output: Path) -> dict[str, Any]:
    runs=[]
    for domain,variants in (("source",SOURCE_VARIANTS),("htt",HTT_VARIANTS)):
        for variant in variants:
            for seed in (20260914,20260915,20260916):
                runs.append({"domain":domain,"variant":variant,"seed":seed,"status":"planned" if domain=="source" else "pending_dependency","formal":True})
    p={"schema":"round5_future_run_inventory_v1","protocol_sha256":protocol_sha256(),"formal_runs":len(runs),"runs":runs}
    atomic_json(output,p);return p


def main(argv: Iterable[str] | None=None) -> int:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("protocol-hash")
    a=sub.add_parser("inventory");a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("build-source-cache");a.add_argument("--input",type=Path,required=True);a.add_argument("--output",type=Path,required=True);a.add_argument("--split",choices=["train","val","validation"],required=True)
    a=sub.add_parser("assemble-htt-upstream");a.add_argument("--contract-json",type=Path,required=True);a.add_argument("--prediction-manifest",type=Path,required=True);a.add_argument("--output",type=Path,required=True);a.add_argument("--allow-smoke",action="store_true")
    a=sub.add_parser("audit-htt-support");g=a.add_mutually_exclusive_group(required=True);g.add_argument("--upstream",type=Path);g.add_argument("--contract-json",type=Path);a.add_argument("--output",type=Path,required=True);a.add_argument("--allow-smoke",action="store_true")
    for name,variants in (("train-source",SOURCE_VARIANTS),("train-htt",HTT_VARIANTS)):
        a=sub.add_parser(name);a.add_argument("--variant",choices=variants,required=True);a.add_argument("--seed",type=int,choices=[20260914,20260915,20260916],required=True);a.add_argument("--output",type=Path,required=True);a.add_argument("--device",default="cuda:0");a.add_argument("--protocol-sha256",required=True);a.add_argument("--smoke-epochs",type=int);a.add_argument("--resume",action="store_true");a.add_argument("--interrupt-after-epoch",type=int)
        if name=="train-source":a.add_argument("--train-cache",type=Path,required=True);a.add_argument("--validation-cache",type=Path,required=True)
        else:a.add_argument("--upstream",type=Path,required=True);a.add_argument("--support-audit",type=Path,required=True);a.add_argument("--allow-smoke",action="store_true")
    args=ap.parse_args(argv)
    if args.cmd=="protocol-hash": print(protocol_sha256());return 0
    if args.cmd=="inventory": result=inventory(args.output)
    elif args.cmd=="build-source-cache": result=build_source_cache(args.input,args.output,args.split)
    elif args.cmd=="assemble-htt-upstream": result=assemble_htt_upstream(args.contract_json,args.prediction_manifest,args.output,args.allow_smoke)
    elif args.cmd=="audit-htt-support":
        p=support_payload_from_contract(args.contract_json) if args.contract_json else canonical_htt_payload(args.upstream,require_formal=not args.allow_smoke)
        result=htt_support_audit(p);result["support_source"]={k:p[k] for k in ("contract_path","contract_sha256","fold") if k in p};atomic_json(args.output,result)
    else:
        if args.protocol_sha256!=protocol_sha256(): raise SystemExit("protocol SHA mismatch")
        if args.smoke_epochs is not None and "smoke" not in str(args.output).lower(): raise SystemExit("smoke outputs must contain 'smoke'")
        if args.interrupt_after_epoch is not None and (args.smoke_epochs is None or not 0 < args.interrupt_after_epoch < args.smoke_epochs): raise SystemExit("interrupt requires smoke budget and 0 < interrupt-after-epoch < smoke-epochs")
        if args.resume and args.interrupt_after_epoch is not None: raise SystemExit("resume and interrupt cannot be requested together")
        if args.cmd=="train-source":result=train_source(args.train_cache,args.validation_cache,args.output,args.variant,args.seed,args.device,args.smoke_epochs,args.resume,args.interrupt_after_epoch)
        else:result=train_htt(args.upstream,args.support_audit,args.output,args.variant,args.seed,args.device,args.smoke_epochs,not args.allow_smoke,args.resume,args.interrupt_after_epoch)
    print(json.dumps(result,indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
