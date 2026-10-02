"""Shared R16 ToucHD release adapter and force-transfer utilities."""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
HTT_ROOT = HERE.parent
REPO_ROOT = HTT_ROOT.parents[1]
R5 = HTT_ROOT / "round5_force_conditioned_slip/current"
sys.path.insert(0, str(HTT_ROOT))
sys.path.insert(0, str(R5))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from adapters import preprocess  # noqa: E402
from models import ForceAdapter  # noqa: E402
from sources import load_decoupled_decoder  # noqa: E402
import phase2_b_multitask as p2  # noqa: E402


SOURCE_CHECKPOINT = Path("/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth")
TARGET = "released compact [Fx,Fy,-Fz], official implementation uses values as N; domain-local"
PREPROCESS = "source-compatible reference difference offset=0.5, rotate/crop 4:3, resize 320x240, current then index-3 frame"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        array = value.detach().cpu().contiguous()
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_rgb(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(archive.read(name))).convert("RGB"))


def make_encoder_input(archive: zipfile.ZipFile, key: str, rows: list, index: int) -> torch.Tensor:
    if index < 5:
        raise ValueError("official two-frame GelSight route starts at compact row index 5")
    base = f"{key}/gelsight/"
    background = load_rgb(archive, base + "image_1.png")
    current = load_rgb(archive, base + f"image_{int(rows[index][0])}.png")
    previous = load_rgb(archive, base + f"image_{int(rows[index - 3][0])}.png")
    return torch.cat((preprocess(current, background), preprocess(previous, background)), dim=0)


def published_target(row: list) -> torch.Tensor:
    return torch.tensor([float(row[1]), float(row[2]), -float(row[3])], dtype=torch.float32)


def load_source_model(device: torch.device):
    model, payload = p2.load_b_checkpoint(SOURCE_CHECKPOINT, device)
    if model.encoder_name != "mae" or model.decoder_variant != "decoupled":
        raise RuntimeError("R16 requires the locked MAE decoupled source")
    return model, payload


def fresh_adapter(seed: int, *, trunk_state: dict[str, torch.Tensor] | None = None) -> ForceAdapter:
    seed_all(seed)
    decoder, _ = load_decoupled_decoder(SOURCE_CHECKPOINT)
    adapter = ForceAdapter(decoder)
    if trunk_state is not None:
        adapter.force_pooler.load_state_dict(trunk_state["force_pooler"], strict=True)
        adapter.force_trunk.load_state_dict(trunk_state["force_trunk"], strict=True)
    # Reset is explicit and route-independent at the HTT boundary.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adapter.force_head.reset_parameters()
    return adapter


def transferable_private_state(adapter: ForceAdapter) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "force_pooler": {k: v.detach().cpu().clone() for k, v in adapter.force_pooler.state_dict().items()},
        "force_trunk": {k: v.detach().cpu().clone() for k, v in adapter.force_trunk.state_dict().items()},
    }
