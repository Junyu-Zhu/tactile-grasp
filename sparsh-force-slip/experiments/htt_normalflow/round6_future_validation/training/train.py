#!/usr/bin/env python3
"""Train the pre-registered Round-6 conditional GRU and evaluate held-out roles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
GROUPS = ("A_visual", "B_force", "C_force_delta")
SEEDS = (20260914, 20260915, 20260916)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)); os.replace(tmp, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, tmp); os.replace(tmp, path)


def atomic_predictions(path: Path, role_payload: dict, y: np.ndarray, p: np.ndarray, p_slip: np.ndarray) -> None:
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("w",newline="") as stream:
        writer=csv.writer(stream); writer.writerow(("episode_id","leakage_group","t","first_current_slip_t","target_future_H1","p_future_raw","p_slip_current","p_future_times_p_slip"))
        for i in range(len(y)):
            writer.writerow((role_payload["episode_id"][i],role_payload["leakage_group"][i],int(role_payload["t"][i]),role_payload["first_current_slip_t"][i],int(y[i]),float(p[i]),float(p_slip[i]),float(p[i]*p_slip[i])))
    os.replace(tmp,path)


class GRURisk(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__(); self.gru = nn.GRU(input_dim, hidden, batch_first=True); self.risk = nn.Linear(hidden, 1)
    def forward(self, x): return self.risk(self.gru(x)[0][:, -1]).squeeze(1)


def canonical_initial_state(group: str, seed: int, dims: dict[str, int], hidden: int) -> tuple[dict, dict]:
    torch.manual_seed(seed); full = GRURisk(dims["C_force_delta"], hidden)
    full_state = full.state_dict(); dim = dims[group]
    target = GRURisk(dim, hidden)
    state = {k: v.detach().clone() for k, v in full_state.items()}
    state["gru.weight_ih_l0"] = state["gru.weight_ih_l0"][:, :dim].clone()
    target.load_state_dict(state, strict=True)
    evidence = {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in target.state_dict().items()}
    evidence["shared_origin_C_seed"] = seed; evidence["input_columns"] = [0, dim]
    return target.state_dict(), evidence


def set_epoch_seed(seed: int, epoch: int) -> None:
    value = seed + 1009 * epoch; random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def code_bundle_sha() -> str:
    return identity_sha({p.name: sha256(p) for p in (Path(__file__).resolve(), PROTOCOL.resolve())})


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    if positives == 0: return 0.0
    order = np.argsort(-score, kind="mergesort"); ys = y[order]; ss = score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    tp = np.cumsum(ys)[ends]; fp = (ends + 1) - tp
    recall = tp / positives; precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def confusion(y, score, threshold):
    pred = score >= threshold; y = y.astype(bool)
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); tn = int(np.sum(~pred & ~y)); fn = int(np.sum(~pred & y))
    recall = tp / (tp + fn) if tp + fn else math.nan; fpr = fp / (fp + tn) if fp + tn else math.nan
    ba = .5 * (recall + (tn / (tn + fp))) if tp + fn and tn + fp else math.nan
    return {"threshold": float(threshold), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "balanced_accuracy": ba, "fpr": fpr, "recall": recall, "observed_no_alarm": tp + fp == 0}


def calibrate(y, score, kind: str, limit: float | None = None):
    y = np.asarray(y, dtype=bool); score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort"); ys = y[order]; ss = score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    tp = np.cumsum(ys)[ends]; fp = (ends+1)-tp; positives=int(y.sum()); negatives=len(y)-positives
    rows = [confusion(y, score, np.nextafter(1.0, math.inf))]
    for i,end in enumerate(ends):
        tpi,fpi=int(tp[i]),int(fp[i]); fni, tni=positives-tpi, negatives-fpi
        recall=tpi/positives if positives else math.nan; fpr=fpi/negatives if negatives else math.nan
        ba=.5*(recall+tni/negatives) if positives and negatives else math.nan
        rows.append({"threshold":float(ss[end]),"tn":tni,"fp":fpi,"fn":fni,"tp":tpi,"balanced_accuracy":ba,"fpr":fpr,"recall":recall,"observed_no_alarm":tpi+fpi==0})
    if kind == "maxBA":
        return max(rows, key=lambda r: (r["balanced_accuracy"], -r["fpr"], r["threshold"]))
    valid = [r for r in rows if r["fpr"] <= float(limit) + 1e-15]
    return max(valid, key=lambda r: (r["recall"], -r["fpr"], r["threshold"]))


def evaluate_role(y, p, p_slip, thresholds=None):
    result = {"n": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()), "average_precision": average_precision(y, p), "brier": float(np.mean((p-y)**2)), "probability_std": float(np.std(p)), "probability_unique": int(len(np.unique(p)))}
    if thresholds is None:
        result["thresholds"] = {"fixed_0.5": confusion(y, p, .5), "calibration_maxBA": calibrate(y, p, "maxBA")}
        for limit in (.01, .05, .10): result["thresholds"][f"calibration_FPR_{limit:.2f}"] = calibrate(y, p, "fpr", limit)
    else:
        result["thresholds"] = {name: confusion(y, p, row["threshold"]) for name, row in thresholds.items()}
    gated = p * p_slip
    result["fixed_multiplication_ablation"] = {"average_precision": average_precision(y, gated), "brier": float(np.mean((gated-y)**2)), "fixed_0.5": confusion(y, gated, .5)}
    return result


def _role(payload, role, dim, mean, std):
    if int(payload["feature_layout_C"]["z"][1]) != 768 or int(payload["group_input_dims"]["A_visual"]) != 769:
        raise ValueError("prepared feature layout is not the frozen z768+pSlip prefix")
    row = payload["roles"][role]
    if row["x_C"].requires_grad or row["y"].requires_grad: raise ValueError("prepared tensors must be detached")
    if row["x_C"].ndim != 3 or row["x_C"].shape[1:] != (4,775): raise ValueError(f"invalid prepared shape for {role}")
    x = (row["x_C"].float()[:, :, :dim] - mean[:dim]) / std[:dim]
    return x, row["y"].float(), row["x_C"].float()[:, -1, 768]


def train(data_path: Path, group: str, seed: int, output: Path, device: str, smoke: bool, execute_formal: bool, resume: bool, interrupt_after_epoch: int | None):
    cfg = json.loads(PROTOCOL.read_text())
    if group not in GROUPS or seed not in SEEDS: raise ValueError("unregistered run identity")
    if smoke == execute_formal: raise ValueError("select exactly one execution mode: --smoke or --execute-formal")
    total_epochs = 2 if smoke else int(cfg["max_epochs"])
    if interrupt_after_epoch is not None and not (0 < interrupt_after_epoch < total_epochs): raise ValueError("interrupt epoch must be within the unchanged execution budget")
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True)
    if not device.startswith("cpu"):
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    if data.get("schema") != cfg["schema"] + "_prepared_v1" or data.get("status") != "complete" or not data.get("formal"):
        raise ValueError("invalid or non-formal prepared data")
    if data["provenance"]["protocol"]["sha256"] != sha256(PROTOCOL): raise ValueError("prepared data protocol drift")
    dims = {k: int(v) for k, v in data["group_input_dims"].items()}; dim = dims[group]
    mean, std = data["normalization_C"]["mean"].float(), data["normalization_C"]["std"].float()
    xtr, ytr, _ = _role(data, "fit_train", dim, mean, std); xsel, ysel, _ = _role(data, "selection", dim, mean, std)
    initial, init_evidence = canonical_initial_state(group, seed, dims, int(cfg["model"]["hidden_size"]))
    model = GRURisk(dim, int(cfg["model"]["hidden_size"])).to(device); model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["lr"]), weight_decay=float(cfg["optimizer"]["weight_decay"]))
    mode = "smoke" if smoke else "formal"
    run_config = {"schema": cfg["schema"], "group": group, "seed": seed, "data_path": str(data_path.resolve()), "data_sha256": sha256(data_path), "protocol_sha256": sha256(PROTOCOL), "code_bundle_sha256": code_bundle_sha(), "input_dim": dim, "history": 4, "horizon": 1, "mode": mode, "effective_epoch_budget": total_epochs, "max_epochs": int(cfg["max_epochs"]), "patience": int(cfg["early_stopping_patience"]), "batch_size": int(cfg["batch_size"]), "optimizer": cfg["optimizer"], "selection": cfg["selection"], "initialization": init_evidence}
    run_id = identity_sha(run_config); latest = output / "latest.pth"; best_path = output / "best.pth"
    if not resume and output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing non-empty output without --resume: {output}")
    start, history, best_loss, best_epoch, best_state, stale = 1, [], math.inf, None, None, 0
    changed = {k: False for k in model.state_dict()}; finite_all_gradients = True
    if resume:
        if not latest.exists(): raise FileNotFoundError(latest)
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state.get("run_identity_sha256") != run_id: raise ValueError("resume identity mismatch")
        model.load_state_dict(state["model_state"], strict=True); optimizer.load_state_dict(state["optimizer_state"])
        history = state["history"]; best_loss = float(state["best_selection_loss"]); best_epoch = state["best_epoch"]; best_state = state["best_model_state"]; stale = int(state["stale"]); start = int(state["epoch"])+1
        changed = dict(state["checks"]["parameter_updates_by_tensor"]); finite_all_gradients = bool(state["checks"]["finite_loss_and_all_trainable_gradients"])
    output.mkdir(parents=True, exist_ok=True)
    pos = float(ytr.sum()); neg = float(len(ytr)-pos)
    if pos <= 0 or neg <= 0: raise ValueError("fit role must contain both classes")
    pos_weight = torch.tensor(neg/pos, device=device); dataset = TensorDataset(xtr, ytr)
    for epoch in range(start, total_epochs+1):
        if stale >= int(cfg["early_stopping_patience"]): break
        set_epoch_seed(seed, epoch); generator = torch.Generator().manual_seed(seed + 7919*epoch)
        loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
        before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; model.train(); losses=[]
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True); loss = nn.functional.binary_cross_entropy_with_logits(model(x.to(device)), y.to(device), pos_weight=pos_weight)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            finite_all_gradients = finite_all_gradients and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
            if not finite_all_gradients: raise FloatingPointError("missing or non-finite trainable gradient")
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        for k,v in model.state_dict().items(): changed[k] |= not torch.equal(before[k],v.detach().cpu())
        model.eval()
        with torch.no_grad(): selection_loss = float(nn.functional.binary_cross_entropy_with_logits(model(xsel.to(device)), ysel.to(device)).cpu())
        if not math.isfinite(selection_loss): raise FloatingPointError("non-finite selection loss")
        history.append({"epoch": epoch, "fit_weighted_bce": float(np.mean(losses)), "selection_unweighted_bce": selection_loss})
        if selection_loss < best_loss:
            best_loss, best_epoch, best_state, stale = selection_loss, epoch, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else: stale += 1
        checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch, "history": history, "best_selection_loss": best_loss, "best_epoch": best_epoch, "best_model_state": best_state, "stale": stale, "run_config": run_config, "run_identity_sha256": run_id, "normalization": {"mean": mean[:dim], "std": std[:dim]}, "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, "checks":{"finite_loss_and_all_trainable_gradients":finite_all_gradients,"parameter_updates_by_tensor":changed}}
        atomic_torch(latest, checkpoint)
        if interrupt_after_epoch == epoch:
            summary = {"status":"interrupted", "formal":False, "smoke":smoke, "group":group, "seed":seed, "run_config":run_config, "run_identity_sha256":run_id, "history":history, "checks":{"finite_loss_and_all_trainable_gradients":finite_all_gradients,"parameter_updates_by_tensor":changed}, "artifacts":{"latest":{"path":str(latest.resolve()),"sha256":sha256(latest)}}}
            atomic_json(output/"summary.json",summary); return summary
    if best_state is None: raise RuntimeError("no best checkpoint selected")
    if not finite_all_gradients or not all(changed.values()): raise RuntimeError("training connectivity/update acceptance failed")
    atomic_torch(best_path, {"model_state":best_state,"best_epoch":best_epoch,"best_selection_loss":best_loss,"run_config":run_config,"run_identity_sha256":run_id,"normalization":{"mean":mean[:dim],"std":std[:dim]}})
    model.load_state_dict(best_state, strict=True); model.eval(); evaluations={}; calibration_thresholds={}; prediction_artifacts={}
    if not smoke:
        for role in ("calibration","outer"):
            x,y,p_slip = _role(data, role, dim, mean, std)
            with torch.no_grad(): p = torch.sigmoid(model(x.to(device))).cpu().numpy()
            if not np.isfinite(p).all(): raise FloatingPointError("non-finite evaluation probability")
            if role == "calibration":
                evaluations[role] = evaluate_role(y.numpy(),p,p_slip.numpy()); calibration_thresholds = evaluations[role]["thresholds"]
            else: evaluations[role] = evaluate_role(y.numpy(),p,p_slip.numpy(),calibration_thresholds)
            prediction_path=output/f"predictions_{role}.csv"; atomic_predictions(prediction_path,data["roles"][role],y.numpy(),p,p_slip.numpy()); prediction_artifacts[role]={"path":str(prediction_path.resolve()),"sha256":sha256(prediction_path)}
    restored = torch.load(best_path,map_location="cpu",weights_only=False)
    roundtrip = restored["run_identity_sha256"] == run_id and all(torch.equal(restored["model_state"][k],best_state[k]) for k in best_state)
    if not roundtrip: raise RuntimeError("best checkpoint round-trip failure")
    summary={"status":"complete","formal":execute_formal,"smoke":smoke,"group":group,"seed":seed,"run_config":run_config,"run_identity_sha256":run_id,"best_epoch":best_epoch,"best_selection_loss":best_loss,"history":history,"checks":{"finite_loss_and_all_trainable_gradients":finite_all_gradients,"all_parameter_tensors_updated":all(changed.values()),"parameter_updates_by_tensor":changed,"checkpoint_roundtrip_exact":roundtrip,"calibration_and_outer_not_used_for_training_or_selection":True},"evaluations":evaluations,"artifacts":{"best":{"path":str(best_path.resolve()),"sha256":sha256(best_path)},"latest":{"path":str(latest.resolve()),"sha256":sha256(latest)},"predictions":prediction_artifacts}}
    atomic_json(output/"summary.json",summary); return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--group",choices=GROUPS,required=True); p.add_argument("--seed",type=int,choices=SEEDS,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--resume",action="store_true"); p.add_argument("--smoke",action="store_true"); p.add_argument("--execute-formal",action="store_true"); p.add_argument("--interrupt-after-epoch",type=int)
    a=p.parse_args(); result=train(a.data.resolve(),a.group,a.seed,a.output.resolve(),a.device,a.smoke,a.execute_formal,a.resume,a.interrupt_after_epoch); print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
