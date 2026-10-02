#!/usr/bin/env python3
"""Build resumable episode-sharded caches of complete frozen Sparsh tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(EXP_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import adapters  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402
from prepare_splits import file_hash, verify  # noqa: E402


def sha256_file(path: Path) -> str:
    return file_hash(path)


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(tmp, path)


def episode_key(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode()).hexdigest()[:20]


def canonical_encoder(name: str) -> str:
    return "mae" if name == "mae_letterbox" else name


def encoder_input(episode: adapters.Episode, t: int, variant: str) -> torch.Tensor:
    if variant != "mae_letterbox":
        return adapters.window(episode, t)["inputs"]["image"]
    # Single-factor geometry ablation: preserve the exact uint8 diff/color
    # semantics but retain the full square GSmini field instead of center-cropping
    # it to 4:3 before resize. Neutral difference is 127/255 after quantization.
    from PIL import Image
    from torchvision import transforms
    from tactile_ssl.data.digit.utils import compute_diff

    resize = transforms.Compose([transforms.Resize((240, 240), antialias=True), transforms.ToTensor()])
    frames = []
    for index in (t, max(t - 5, 0)):
        diff = compute_diff(episode.images[index], episode.reference, offset=0.5)
        square = resize(Image.fromarray(diff).convert("RGB"))
        canvas = torch.full((3, 320, 240), 127.0 / 255.0, dtype=square.dtype)
        canvas[:, 40:280, :] = square
        frames.append(canvas)
    return torch.cat(frames, dim=0)


def roles_by_fold(manifest: dict[str, Any], episode_id: str) -> dict[str, str]:
    roles = {}
    for fold, split in manifest["splits"].items():
        if not fold.startswith("htt_leave_"):
            continue
        matched = [role for role, ids in split.items() if episode_id in ids]
        if len(matched) != 1:
            raise ValueError(f"{episode_id}: expected exactly one role in {fold}, got {matched}")
        roles[fold] = matched[0]
    return roles


def configure_determinism() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    encoder_name = canonical_encoder(args.encoder)
    checkpoint_path = p2.ENCODER_CHECKPOINTS[encoder_name].resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    if manifest.get("schema_version") != 2:
        raise ValueError("Round 3 requires the frozen schema-2 manifest")
    manifest_sha = sha256_file(manifest_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    configure_determinism()
    device = torch.device(args.device)
    # I-JEPA does not carry register_tokens. The production constructor creates
    # them, so keep that otherwise-hidden part of the frozen representation fixed.
    torch.manual_seed(args.constructor_seed)
    model = p2.FrozenEncoderSharedForceSlip(encoder_name, decoder_variant="decoupled").to(device)
    if Path(model.load_info["checkpoint"]).resolve() != checkpoint_path:
        raise RuntimeError("Encoder constructor resolved an unexpected checkpoint")
    model.encoder.eval().requires_grad_(False)
    encoder_sha = tensor_state_sha256(model.encoder.state_dict())
    code_sha256 = {
        "cache.py": sha256_file(Path(__file__)),
        "adapters.py": sha256_file(Path(adapters.__file__)),
        "phase2_b_multitask.py": sha256_file(Path(p2.__file__)),
    }

    eligible_roles = {"train", "validation", "calibration"}
    eligible_ids = set()
    for fold, split in manifest["splits"].items():
        if fold.startswith("htt_leave_"):
            for role in eligible_roles:
                eligible_ids.update(split[role])
    rows = {row["id"]: row for row in manifest["episodes"]}
    selected = []
    for episode_id in sorted(eligible_ids):
        row = rows[episode_id]
        if row["domain"] != "htt":
            raise ValueError(f"Non-HTT episode in HTT fold: {episode_id}")
        episode = adapters.load_htt(row["path"])
        if episode.labels is not None:
            selected.append((row, episode))
    selected_total = len(selected)
    if args.shards > 1:
        selected = [item for i, item in enumerate(selected) if i % args.shards == args.shard]
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]
    if not selected:
        raise ValueError("No labeled HTT development episodes selected")

    entries = []
    for index, (row, episode) in enumerate(selected, 1):
        for source, expected in row["source_files"].items():
            actual = sha256_file(Path(source))
            if actual != expected:
                raise ValueError(f"Source changed: {episode.episode_id}: {source}")
        key = episode_key(episode.episode_id)
        token_path = output / "episodes" / f"{key}.tokens.npy"
        label_path = output / "episodes" / f"{key}.labels.npy"
        meta_path = output / "episodes" / f"{key}.json"
        expected_meta = {
            "episode_id": episode.episode_id,
            "frames": len(episode.images),
            "roles_by_fold": roles_by_fold(manifest, episode.episode_id),
            "source_files": row["source_files"],
            "manifest_sha256": manifest_sha,
            "checkpoint_sha256": checkpoint_sha,
            "encoder_state_sha256": encoder_sha,
            "encoder": args.encoder,
            "constructor_seed": args.constructor_seed,
            "load_info": model.load_info,
            "history": 2,
            "stride": 5,
            "dtype": "float32",
            "code_sha256": code_sha256,
        }
        reusable = False
        if token_path.exists() and label_path.exists() and meta_path.exists():
            old_meta = json.loads(meta_path.read_text())
            tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
            labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
            reusable = all(old_meta.get(k) == v for k, v in expected_meta.items()) and tokens.dtype == np.float32 and len(tokens) == len(labels) == len(episode.images)
            if reusable and (tokens.ndim != 3 or labels.shape != (len(episode.images),)):
                reusable = False
            if reusable:
                reusable = old_meta.get("token_sha256") == sha256_file(token_path) and old_meta.get("label_sha256") == sha256_file(label_path)
        if not reusable:
            if meta_path.exists():
                raise RuntimeError(f"Refusing incompatible partial cache for {episode.episode_id}")
            # A sidecar is the commit marker. Data files without it came from an
            # interrupted episode and are safe to regenerate at these exact paths.
            token_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            batches = []
            pending = []
            for t in range(len(episode.images)):
                pending.append(encoder_input(episode, t, args.encoder))
                if len(pending) == args.batch_size or t + 1 == len(episode.images):
                    images = torch.stack(pending).to(device, non_blocking=True)
                    with torch.inference_mode():
                        z = model.encoder(images)
                    if z.ndim != 3 or not torch.isfinite(z).all():
                        raise RuntimeError(f"Invalid encoder tokens for {episode.episode_id}")
                    batches.append(z.detach().float().cpu().numpy())
                    pending.clear()
            token_array = np.concatenate(batches, axis=0)
            label_array = np.asarray(episode.labels, dtype=np.int64)
            if len(token_array) != len(label_array):
                raise RuntimeError("Token/label alignment failed")
            atomic_npy(token_path, token_array)
            atomic_npy(label_path, label_array)
            expected_meta["token_sha256"] = sha256_file(token_path)
            expected_meta["label_sha256"] = sha256_file(label_path)
            atomic_json(meta_path, expected_meta)
        else:
            expected_meta["token_sha256"] = old_meta["token_sha256"]
            expected_meta["label_sha256"] = old_meta["label_sha256"]
        entries.append({
            **expected_meta,
            "token_path": str(token_path),
            "label_path": str(label_path),
            "token_shape": list(np.load(token_path, mmap_mode="r", allow_pickle=False).shape),
            "status": "reused" if reusable else "built",
        })
        print(f"[{index}/{len(selected)}] {episode.episode_id}: {entries[-1]['status']}", flush=True)

    cache_manifest = {
        "format": "round4_htt_full_sparsh_tokens_v1",
        "status": "complete" if args.max_episodes is None and args.shards == 1 else ("partial_shard" if args.max_episodes is None else "partial_smoke"),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "encoder": args.encoder,
        "constructor_seed": args.constructor_seed,
        "load_info": model.load_info,
        "encoder_state_sha256": encoder_sha,
        "preprocess": (
            "letterbox_full_square_240_then_pad_top_bottom_40_neutral_127_over_255; history=2; stride=5; no augmentation"
            if args.encoder == "mae_letterbox" else
            "adapters.preprocess; history=2; stride=5; no augmentation"
        ),
        "unique_labeled_episodes": len(entries),
        "expected_unique_labeled_episodes": selected_total,
        "shards": args.shards,
        "shard": args.shard,
        "frames": sum(entry["frames"] for entry in entries),
        "entries": entries,
    }
    manifest_name = "cache_manifest.json" if args.shards == 1 else f"cache_manifest.part-{args.shard:03d}-of-{args.shards:03d}.json"
    atomic_json(output / manifest_name, cache_manifest)
    print(json.dumps({k: cache_manifest[k] for k in ("status", "unique_labeled_episodes", "frames", "encoder_state_sha256")}, indent=2))
    return cache_manifest


def merge(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir.resolve()
    parts = sorted(root.glob("cache_manifest.part-*-of-*.json"))
    if not parts:
        raise FileNotFoundError("No shard manifests found")
    payloads = [json.loads(path.read_text()) for path in parts]
    shard_count = payloads[0]["shards"]
    if len(parts) != shard_count or {p["shard"] for p in payloads} != set(range(shard_count)):
        raise ValueError(f"Expected all {shard_count} shard manifests")
    provenance = ("format", "manifest", "manifest_sha256", "checkpoint", "checkpoint_sha256", "encoder", "constructor_seed", "load_info", "encoder_state_sha256", "preprocess", "expected_unique_labeled_episodes")
    if any(any(p[key] != payloads[0][key] for key in provenance) for p in payloads[1:]):
        raise ValueError("Shard provenance mismatch")
    entries = [entry for payload in payloads for entry in payload["entries"]]
    ids = [entry["episode_id"] for entry in entries]
    if len(ids) != len(set(ids)) or len(ids) != payloads[0]["expected_unique_labeled_episodes"]:
        raise ValueError("Shard merge has duplicate or missing episodes")
    merged = {key: payloads[0][key] for key in provenance}
    merged.update({"status": "complete", "shards": shard_count, "shard_manifests": [str(p) for p in parts],
                   "unique_labeled_episodes": len(entries), "frames": sum(e["frames"] for e in entries),
                   "entries": sorted(entries, key=lambda e: e["episode_id"])})
    atomic_json(root / "cache_manifest.json", merged)
    print(json.dumps({"status": "complete", "unique_labeled_episodes": len(entries), "frames": merged["frames"]}, indent=2))
    return merged


def verify_cache(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_dir.resolve()
    payload = json.loads((root / "cache_manifest.json").read_text())
    errors = []
    frames = 0
    for entry in payload["entries"]:
        tokens = np.load(entry["token_path"], mmap_mode="r", allow_pickle=False)
        labels = np.load(entry["label_path"], mmap_mode="r", allow_pickle=False)
        if list(tokens.shape) != entry["token_shape"] or tokens.dtype != np.float32:
            errors.append(f"token mismatch: {entry['episode_id']}")
        if sha256_file(Path(entry["token_path"])) != entry.get("token_sha256"):
            errors.append(f"token hash mismatch: {entry['episode_id']}")
        if sha256_file(Path(entry["label_path"])) != entry.get("label_sha256"):
            errors.append(f"label hash mismatch: {entry['episode_id']}")
        if labels.shape != (entry["frames"],) or not np.isin(labels, (0, 1, 2)).all():
            errors.append(f"label mismatch: {entry['episode_id']}")
        if len(tokens) != entry["frames"]:
            errors.append(f"alignment mismatch: {entry['episode_id']}")
        frames += len(tokens)
    result = {"status": "pass" if not errors and frames == payload["frames"] else "fail", "errors": errors, "episodes": len(payload["entries"]), "frames": frames}
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--manifest", type=Path, required=True)
    build_parser.add_argument("--encoder", choices=("dino", "ijepa", "mae_letterbox"), required=True)
    build_parser.add_argument("--constructor-seed", type=int, default=42)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--device", default="cuda:0")
    build_parser.add_argument("--batch-size", type=int, default=32)
    build_parser.add_argument("--max-episodes", type=int)
    build_parser.add_argument("--shards", type=int, default=1)
    build_parser.add_argument("--shard", type=int, default=0)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if getattr(args, "batch_size", 1) < 1 or (getattr(args, "max_episodes", None) is not None and args.max_episodes < 1):
        parser.error("batch-size and max-episodes must be positive")
    if getattr(args, "shards", 1) < 1 or not 0 <= getattr(args, "shard", 0) < getattr(args, "shards", 1):
        parser.error("Require shards >= 1 and 0 <= shard < shards")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    {"build": build, "merge": merge, "verify": verify_cache}[parsed.command](parsed)
