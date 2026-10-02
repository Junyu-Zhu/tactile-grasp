"""Frozen inference only, role-bound selection, resumable atomic episode caches."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XFORMERS_DISABLED", "1")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "scripts"))
from adapters import load_htt, load_normalflow, preprocess
from prepare_splits import file_hash, verify
import phase2_b_multitask as p2
import phase3_2_world_model as wm
from evaluate_real_force_slip_model_test import load_future_head, build_future_input

STAGE1 = "/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth"
STAGE2 = "/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth"


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def allowed_rows(manifest):
    roles = {}
    for fold, parts in manifest["splits"].items():
        for part in ("train", "validation", "calibration"):
            for episode_id in parts.get(part, []):
                roles.setdefault(episode_id, []).append({"fold": fold, "partition": part})
    # No selection from any test list. CV development union contains all HTT
    # probes; downstream evaluation must preserve per-fold roles.
    return [{**row, "development_roles": roles[row["id"]]} for row in manifest["episodes"] if row["id"] in roles]


def future_predictions(z, force, pslip, head, payload):
    previous = np.maximum(np.arange(len(z)) - 5, 0)
    delta = force - force[previous]
    fn = np.abs(force[:, 2])
    ft = np.linalg.norm(force[:, :2], axis=1)
    values = {"Fn_pred_N": fn, "Ft_pred_N": ft, "Ft_over_Fn_pred": ft/(fn+1e-6),
              "p_slip_current": pslip, **{f"dF{axis}_causal_N": delta[:,i] for i,axis in enumerate("xyz")}}
    aux = torch.tensor(np.stack([values[k] for k in payload["aux_names"]], axis=1))
    x = build_future_input(torch.tensor(z), torch.tensor(z[previous]), aux, payload)
    with torch.inference_mode():
        return (1 - torch.sigmoid(head(x))).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json")
    parser.add_argument("--output", default="/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round2/cache")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads(Path(args.manifest).read_text())
    verify(manifest)
    selected = allowed_rows(manifest)
    if len(selected) != 272 or any(r["domain"] == "normalflow" and r["group"] in ("seed", "ball") for r in selected):
        raise ValueError("Unexpected development inventory")
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    shard_rows = selected[args.shard::args.shards]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, model_payload = p2.load_b_checkpoint(Path(STAGE1), torch.device(args.device))
    head, payload = load_future_head(Path(STAGE2), torch.device("cpu"))
    if model.encoder_name != "mae" or model.decoder_variant != "decoupled" or payload["horizons"] != [1,3,5]:
        raise ValueError("Unexpected frozen model identity")
    sources = [Path(__file__), ROOT / "adapters.py", ROOT / "prepare_splits.py",
               ROOT.parents[1] / "scripts/phase2_b_multitask.py", ROOT.parents[1] / "scripts/phase3_2_world_model.py",
               ROOT.parents[1] / "scripts/evaluate_real_force_slip_model_test.py"]
    identity = {"split_manifest_sha256": file_hash(args.manifest), "checkpoint_sha256": file_hash(STAGE1),
                "future_checkpoint_sha256": file_hash(STAGE2), "code_hashes": {str(p): file_hash(p) for p in sources},
                "preprocessing": "legacy load_sample_from_buf; raw reference; resize320x240; RGB; current,t-5; no NF background",
                "precision": "float32 TF32 off", "device": args.device,
                "batch_size": args.batch_size, "threads": args.threads, "python": sys.version,
                "python_executable": sys.executable,
                "runtime_versions": {name: importlib.metadata.version(name) for name in
                                     ("torch", "torchvision", "numpy", "Pillow", "omegaconf", "hydra-core")},
                "torch_cuda_version": torch.version.cuda,
                "xformers_disabled": os.environ.get("XFORMERS_DISABLED"),
                "sparsh_source_hashes": {str(p): file_hash(p) for p in sorted((p2.SPARSH_REPO / "tactile_ssl").rglob("*.py"))}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    index = {"schema_version": 1, **identity, "preprocessing_fingerprint": fingerprint,
             "future_head": {"path": STAGE2, "epoch": payload["epoch"], "horizons": [1,3,5],
                             "input_schema": payload.get("input_schema", ["z", "aux"]), "aux_names": payload["aux_names"],
                             "target": "original future any-slip; deployed input uses predicted-force delta(t,t-5)",
                             "training_source_caveat": "historical feature generator uses dataset GT delta_force; deployment uses predictions"},
             "checkpoint_epoch": model_payload.get("epoch"), "expected_episodes": len(selected),
             "shard": args.shard, "shards": args.shards, "expected_shard_episodes": len(shard_rows),
             "role_policy": "per-fold development only; union covers HTT probes, not a global blind holdout; NF test objects excluded",
             "episodes": [], "status": "running", "pid": os.getpid()}
    start = time.monotonic()
    index_path = output / ("index.json" if args.shards == 1 else f"index_shard{args.shard}.json")
    for row in shard_rows[:args.limit_episodes] if args.limit_episodes else shard_rows:
        cache = output / (row["id"].replace("/", "__") + ".npz")
        sidecar = cache.with_suffix(".json")
        for path, expected in row["source_files"].items():
            if file_hash(path) != expected:
                raise ValueError(f"Source changed: {path}")
        record = {**row, "cache_path": str(cache), "fingerprint": fingerprint}
        if cache.exists() and sidecar.exists():
            saved = json.loads(sidecar.read_text())
            if saved["fingerprint"] != fingerprint or saved["source_files"] != row["source_files"] or file_hash(cache) != saved["cache_sha256"]:
                raise ValueError(f"Stale cache: {cache}")
            record = saved
        else:
            episode = load_htt(row["path"]) if row["domain"] == "htt" else load_normalflow(row["path"])
            if len(episode.images) != row["frames"]:
                raise ValueError("Frame count changed")
            frames = torch.stack([preprocess(im, episode.reference) for im in episode.images])
            zs, forces, ps = [], [], []
            before = time.monotonic()
            with torch.inference_mode():
                for offset in range(0, len(frames), args.batch_size):
                    idx = torch.arange(offset, min(offset+args.batch_size,len(frames)))
                    x = torch.cat([frames[idx], frames[torch.clamp(idx-5,min=0)]], dim=1).to(args.device)
                    tokens = model.encoder(x)
                    out = model.decoder(tokens)
                    zs.append(wm.pool_latent(tokens).float().cpu().numpy())
                    forces.append((out["force"] * torch.tensor([1.5,1.5,2.0],device=args.device)).float().cpu().numpy())
                    ps.append(torch.softmax(out["slip"],dim=1)[:,1].cpu().numpy())
            z, force, pslip = np.concatenate(zs), np.concatenate(forces), np.concatenate(ps)
            risk = future_predictions(z, force, pslip, head, payload)
            if not all(np.isfinite(a).all() for a in (z,force,pslip,risk)):
                raise ValueError("Nonfinite model output")
            n = len(frames)
            image_delta = np.full(n, np.nan, np.float32)
            image_delta[1:] = np.abs(np.diff(episode.images.astype(np.float32)/255,axis=0)).mean(axis=(1,2,3))
            arrays = {"z":z,"force_pred":force,"p_slip":pslip,"p_future":risk,
                      "labels":episode.labels if episode.labels is not None else np.full(n,-1,np.int64),
                      "pose":episode.pose if episode.pose is not None else np.full((n,4,4),np.nan,np.float32),
                      "time":episode.time if episode.time is not None else np.full(n,np.nan),
                      "image_delta_l1":image_delta,"frame_index":np.arange(n)}
            temporary = cache.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(cache)
            record.update(cache_sha256=file_hash(cache), elapsed_seconds=time.monotonic()-before)
            atomic_json(sidecar, record)
        index["episodes"].append(record)
        index["elapsed_seconds"] = time.monotonic()-start
        atomic_json(index_path, index)
        print(json.dumps({"shard":args.shard,"completed":len(index["episodes"]),"total":len(shard_rows),"episode":row["id"],"elapsed":index["elapsed_seconds"]}),flush=True)
    index["status"] = "complete" if len(index["episodes"]) == len(shard_rows) else "partial_smoke"
    atomic_json(index_path, index)


if __name__ == "__main__":
    main()
