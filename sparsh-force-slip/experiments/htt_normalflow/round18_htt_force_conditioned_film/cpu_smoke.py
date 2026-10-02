#!/usr/bin/env python3
"""Small real-input CPU unit smoke; GPU smoke owns timing and recovery proof."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

import train as r18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    r18.validate_cache(data, args.data, args.fold, args.seed)
    norm, weights, _ = r18.fit_assets(data)
    fit = data["roles"]["fit"]
    x = r18.normalize(fit["x"][:64], norm)
    stage = fit["stage"][:64]
    mask = stage.eq(0) | stage.eq(2)
    records = {}
    for group in r18.GROUPS:
        model = r18.init_model(group, args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        loss = (nn.functional.binary_cross_entropy_with_logits(model(x)[mask], stage[mask].eq(2).float(), reduction="none") * weights[:64][mask]).mean()
        loss.backward()
        finite = bool(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        records[group] = {"loss": float(loss), "finite_loss_and_gradients": finite}
    with torch.no_grad():
        identity = torch.equal(r18.init_model("V0", args.seed)(x), r18.init_model("M0", args.seed)(x))
    result = {"schema": "round18_cpu_smoke_v1", "status": "pass" if identity and all(v["finite_loss_and_gradients"] for v in records.values()) else "fail", "real_input": str(args.data), "real_input_sha256": r18.sha(args.data), "records": records, "initial_film_identity_exact": identity, "test_consumed": False}
    r18.atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

