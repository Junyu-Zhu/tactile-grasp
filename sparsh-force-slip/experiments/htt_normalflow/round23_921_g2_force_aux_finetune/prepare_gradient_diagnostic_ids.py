#!/usr/bin/env python3
"""Freeze deterministic fit-only endpoint IDs for the preregistered E3 gradient diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def endpoint_key(fold, seed, index, episode_id, t):
    raw = f"round23-g2-gradient-v1|{fold}|{seed}|{index}|{episode_id}|{t}".encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    datasets = {}
    for run in inventory["runs"]:
        key = (int(run["fold"]), int(run["seed"]))
        datasets[key] = Path(run["data"])
    if len(datasets) != 12:
        raise ValueError("expected exactly 12 fold/seed datasets")
    rows = []
    for (fold, seed), path in sorted(datasets.items()):
        prepared = torch.load(path, map_location="cpu", weights_only=False)
        fit = prepared["roles"]["fit"]
        selected = []
        for stage in (0, 2):
            candidates = []
            for index in torch.nonzero(fit["stage"].eq(stage), as_tuple=False).flatten().tolist():
                episode_id = str(fit["episode_id"][index])
                t = int(fit["t"][index])
                candidates.append((endpoint_key(fold, seed, index, episode_id, t), index, episode_id, t))
            if len(candidates) < 32:
                raise ValueError(f"insufficient stage={stage} endpoints for fold={fold} seed={seed}")
            selected.extend((stage, *item) for item in sorted(candidates)[:32])
        selected.sort(key=lambda row: row[1])
        rows.append({
            "fold": fold,
            "seed": seed,
            "data": str(path),
            "data_sha256": sha(path),
            "batch_size": 64,
            "static": 32,
            "gross": 32,
            "endpoints": [
                {"stage": stage, "rank_sha256": rank, "fit_index": index, "episode_id": episode_id, "t": t}
                for stage, rank, index, episode_id, t in selected
            ],
        })
    output = {
        "schema": "round23_g2_gradient_diagnostic_ids_v1",
        "status": "frozen_before_formal_results",
        "inventory": str(args.inventory),
        "inventory_sha256": sha(args.inventory),
        "selection_rule": "within each fold/seed and stage 0/2, ascending SHA256(round23-g2-gradient-v1|fold|seed|fit_index|episode_id|t), first 32 per stage",
        "datasets": rows,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"datasets": len(rows), "endpoints": sum(len(r["endpoints"]) for r in rows)}))


if __name__ == "__main__":
    main()
