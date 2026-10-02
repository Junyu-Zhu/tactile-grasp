#!/usr/bin/env python3
"""Read-only provenance and load-boundary audit for Round 4 encoders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
SPARSH_ROOT = Path("/home/zjy/document/sparsh")
for item in (SPARSH_ROOT, EXP_ROOT, REPO_ROOT / "scripts"):
    sys.path.insert(0, str(item))
os.environ.setdefault("XFORMERS_DISABLED", "1")

import phase2_b_multitask as p2  # noqa: E402


EXPECTED = {
    "dino": "f1f97da5c26bf2b1ba1f31ec9f5d02ee1b4ef649c9f3df119822a06d264c4115",
    "ijepa": "b5c6945a07c9b743d752be4d9bf77dbefa2f05caeb70b8f81083ae542fcd7fef",
}
OFFICIAL_CONFIGS = {
    "dino": SPARSH_ROOT / "config/model/dino_vit.yaml",
    "ijepa": SPARSH_ROOT / "config/model/ijepa_vit.yaml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def audit_encoder(name: str) -> dict[str, Any]:
    checkpoint = p2.ENCODER_CHECKPOINTS[name].resolve()
    torch.manual_seed(42)
    model = p2.FrozenEncoderSharedForceSlip(name, "decoupled")
    torch.manual_seed(42)
    repeated = p2.FrozenEncoderSharedForceSlip(name, "decoupled")
    state = model.encoder.state_dict()
    repeated_state = repeated.encoder.state_dict()
    load_info = model.load_info
    missing = load_info["missing_keys"]
    expected_missing = [] if name == "dino" else ["register_tokens"]
    exact_repeat = set(state) == set(repeated_state) and all(
        torch.equal(state[key], repeated_state[key]) for key in state
    )
    result = {
        "encoder": name,
        "method_identity": "Sparsh DINO (not DINOv2)" if name == "dino" else "Sparsh I-JEPA",
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "expected_checkpoint_sha256": EXPECTED[name],
        "load_info": load_info,
        "constructor_seed": 42,
        "constructor_seeded_missing_keys": missing,
        "expected_missing_keys": expected_missing,
        "complete_encoder_state_tensors": len(state),
        "complete_encoder_parameters": sum(parameter.numel() for parameter in model.encoder.parameters()),
        "complete_encoder_state_sha256": tensor_state_sha256(state),
        "same_seed_reconstruction_exact": exact_repeat,
        "official_model_config": str(OFFICIAL_CONFIGS[name]),
        "official_model_config_sha256": sha256_file(OFFICIAL_CONFIGS[name]),
        "official_downstream_input_config": str(SPARSH_ROOT / "config/task/t2_slip_detection.yaml"),
        "official_downstream_input_config_sha256": sha256_file(SPARSH_ROOT / "config/task/t2_slip_detection.yaml"),
        "runtime_input_config": p2.encoder_input_config(name),
    }
    result["status"] = "pass" if (
        result["checkpoint_sha256"] == EXPECTED[name]
        and load_info["missing_keys"] == expected_missing
        and load_info["unexpected_keys"] == []
        and len(state) == 174
        and exact_repeat
        and result["runtime_input_config"]["in_chans"] == 6
        and result["runtime_input_config"]["num_frames"] == 2
    ) else "fail"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoders = [audit_encoder(name) for name in ("dino", "ijepa")]
    payload = {
        "status": "pass" if all(row["status"] == "pass" for row in encoders) else "fail",
        "encoders": encoders,
        "phase2_source": str(Path(p2.__file__).resolve()),
        "phase2_source_sha256": sha256_file(Path(p2.__file__).resolve()),
        "protocol": str(HERE / "PROTOCOL.md"),
        "protocol_sha256": sha256_file(HERE / "PROTOCOL.md"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, args.output)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
