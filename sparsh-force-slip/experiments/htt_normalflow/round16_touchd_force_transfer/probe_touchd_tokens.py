#!/usr/bin/env python3
"""Decode a bounded real Mini sample, encode full MAE tokens, and project cache cost."""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import torch

from touchd_common import (PREPROCESS, SOURCE_CHECKPOINT, TARGET, atomic_json, atomic_torch,
                           load_source_model, make_encoder_input, published_target, sha256,
                           state_sha256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-role", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "pass":
        raise RuntimeError("ToucHD release audit must pass")
    roles = {int(obj): role for role, objects in audit["split"]["roles"].items() for obj in objects}
    compact = json.loads((args.data / "all_data_direction.json").read_text())
    selected = []
    counts = {"fit": 0, "selection": 0}
    for key in sorted(compact):
        obj = int(key[3:6]); role = roles[obj]; values = compact[key]["gelsight"]
        for index in range(5, len(values)):
            if counts[role] < args.per_role:
                selected.append((key, index, role))
                counts[role] += 1
        if all(value >= args.per_role for value in counts.values()):
            break
    if counts != {"fit": args.per_role, "selection": args.per_role}:
        raise RuntimeError(f"could not select bounded role-balanced sample: {counts}")

    device = torch.device(args.device)
    model, payload = load_source_model(device)
    encoder_before = state_sha256(model.encoder.state_dict())
    inputs, metadata, targets = [], [], []
    decode_started = time.monotonic()
    current_key = None
    archive = None
    try:
        for key, index, role in selected:
            if key != current_key:
                if archive is not None:
                    archive.close()
                archive = zipfile.ZipFile(args.data / f"{key}.zip")
                current_key = key
            rows = compact[key]["gelsight"]
            inputs.append(make_encoder_input(archive, key, rows, index))
            targets.append(published_target(rows[index]))
            metadata.append({"key": key, "row_index": index, "image_id": int(rows[index][0]),
                             "previous_image_id": int(rows[index - 3][0]), "role": role,
                             "stage": str(rows[index][4]), "object": int(key[3:6]), "speed": int(key[-1])})
    finally:
        if archive is not None:
            archive.close()
    decode_seconds = time.monotonic() - decode_started

    tokens = []
    torch.cuda.reset_peak_memory_stats(device)
    encode_started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(inputs), args.batch_size):
            batch = torch.stack(inputs[offset:offset + args.batch_size]).to(device)
            output = model.encoder(batch)
            if output.ndim != 3 or output.shape[1:] != (300, 768) or not torch.isfinite(output).all():
                raise RuntimeError(f"unexpected complete token interface {tuple(output.shape)}")
            tokens.append(output.cpu().to(torch.float16))
    torch.cuda.synchronize(device)
    encode_seconds = time.monotonic() - encode_started
    token_tensor = torch.cat(tokens)
    target_tensor = torch.stack(targets)
    if state_sha256(model.encoder.state_dict()) != encoder_before:
        raise RuntimeError("frozen encoder changed during cache probe")

    eligible = sum(max(0, int(row["eligible_rows"]) - 5) for row in audit["archives"])
    bytes_per = 300 * 768 * torch.tensor([], dtype=torch.float16).element_size()
    cache = {
        "schema": "round16_touchd_full_token_smoke_v1", "status": "complete", "smoke": True,
        "tokens": token_tensor, "targets": target_tensor, "metadata": metadata,
        "target": TARGET, "preprocess": PREPROCESS, "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT), "audit_sha256": sha256(args.audit),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.output / "smoke_tokens.pt"
    atomic_torch(cache_path, cache)
    receipt = {
        "schema": "round16_touchd_token_probe_v1", "status": "pass", "device": str(device),
        "samples": len(metadata), "role_counts": counts, "input_shape": list(torch.stack(inputs).shape),
        "token_shape": list(token_tensor.shape), "token_dtype": str(token_tensor.dtype),
        "tokens_finite": bool(torch.isfinite(token_tensor).all()), "encoder_frozen_bitwise": True,
        "decode_seconds": decode_seconds, "encode_seconds": encode_seconds,
        "decoded_samples_per_second": len(metadata) / decode_seconds,
        "encoded_samples_per_second": len(metadata) / encode_seconds,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "eligible_samples": eligible, "projected_float16_token_bytes": eligible * bytes_per,
        "projected_float32_token_bytes": eligible * bytes_per * 2,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path), "bytes": cache_path.stat().st_size},
        "source_epoch": payload.get("epoch"), "target": TARGET, "preprocess": PREPROCESS,
        "no_future_inputs_or_targets": True,
    }
    atomic_json(args.output / "TOKEN_PROBE.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
