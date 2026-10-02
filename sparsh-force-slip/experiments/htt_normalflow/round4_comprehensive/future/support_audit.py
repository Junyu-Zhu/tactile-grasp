#!/usr/bin/env python3
"""Train-only first-gross support audit for the frozen round-4 future task."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FOLD = "htt_leave_p1"
HISTORY = 4
HORIZONS = (1, 3, 5, 8, 10, 15, 20)
PRIMARY_RULE = {"positive_frames": 20, "positive_episodes": 5, "negative_frames": 100, "negative_episodes": 10}
PROXY_RULE = {"positive_frames": 20, "positive_episodes": 5, "negative_frames": 20, "negative_episodes": 5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(value, encoding="utf-8")
    os.replace(temp, path)


def label_path(row: dict) -> Path:
    paths = [Path(path) for path in row.get("source_files", {}) if path.endswith(".labeled.npz")]
    if len(paths) != 1:
        raise ValueError(f"{row['id']}: expected one labeled.npz, got {paths}")
    return paths[0]


def support_for_episode(labels: np.ndarray, horizon: int, current_stage: int) -> dict:
    gross = np.flatnonzero(labels == 2)
    onset = int(gross[0]) if len(gross) else None
    stop = len(labels) if onset is None else onset
    eligible = [t for t in range(HISTORY - 1, stop) if labels[t] == current_stage and t + horizon < len(labels)]
    positive = [t for t in eligible if onset is not None and t < onset <= t + horizon]
    negative = [t for t in eligible if t not in set(positive)]
    return {
        "eligible_frames": len(eligible), "positive_frames": len(positive), "negative_frames": len(negative),
        "positive_episode": bool(positive), "negative_episode": bool(negative),
        "first_gross_index": onset,
    }


def aggregate(episode_rows: list[dict], stage_name: str, horizon: int, rule: dict) -> dict:
    rows = [row[stage_name][str(horizon)] for row in episode_rows]
    result = {
        "eligible_frames": sum(row["eligible_frames"] for row in rows),
        "positive_frames": sum(row["positive_frames"] for row in rows),
        "negative_frames": sum(row["negative_frames"] for row in rows),
        "positive_episodes": sum(row["positive_episode"] for row in rows),
        "negative_episodes": sum(row["negative_episode"] for row in rows),
    }
    result["requirements"] = dict(rule)
    result["supported"] = all(result[key] >= required for key, required in rule.items())
    return result


def choose_task(primary: dict, proxy: dict) -> dict:
    for horizon in HORIZONS:
        if primary[str(horizon)]["supported"]:
            return {
                "status": "train", "task": "current_static_to_first_gross", "current_stage": 0,
                "horizon": horizon, "selection": "shortest supported primary horizon from train only",
            }
    for horizon in HORIZONS:
        if proxy[str(horizon)]["supported"]:
            return {
                "status": "train", "task": "current_incipient_to_first_gross_proxy", "current_stage": 1,
                "horizon": horizon, "selection": "primary unsupported; shortest supported predefined proxy horizon from train only",
                "claim_limit": "proxy recognizes a label interval backfilled from the later gross boundary; it is not stable-state warning",
            }
    return {
        "status": "do_not_train", "task": None, "current_stage": None, "horizon": None,
        "selection": "neither predefined train-only support rule passed",
    }


def audit(manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("only schema-2 split manifests are accepted")
    split = manifest.get("splits", {}).get(FOLD)
    if not split:
        raise ValueError(f"missing frozen fold {FOLD}")
    train_ids = list(split["train"])
    inventory = {row["id"]: row for row in manifest["episodes"]}
    episode_rows = []
    opened_paths = []
    for episode_id in train_ids:
        row = inventory[episode_id]
        if row.get("domain") != "htt" or row.get("task") != "slip":
            continue
        path = label_path(row)
        expected_hash = row["source_files"][str(path)]
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"label source hash changed: {episode_id}")
        with np.load(path, allow_pickle=False) as archive:
            labels = np.asarray(archive["sliding_labels_bracket"], dtype=np.int8)
        if len(labels) != row["frames"] or not np.isin(labels, (0, 1, 2)).all():
            raise ValueError(f"invalid label array: {episode_id}")
        item = {"episode_id": episode_id, "frames": len(labels), "label_path": str(path), "label_sha256": actual_hash,
                "static": {}, "incipient": {}}
        for horizon in HORIZONS:
            item["static"][str(horizon)] = support_for_episode(labels, horizon, 0)
            item["incipient"][str(horizon)] = support_for_episode(labels, horizon, 1)
        episode_rows.append(item)
        opened_paths.append(str(path))
    if not episode_rows:
        raise ValueError("no labeled slip episodes in train role")
    primary = {str(h): aggregate(episode_rows, "static", h, PRIMARY_RULE) for h in HORIZONS}
    proxy = {str(h): aggregate(episode_rows, "incipient", h, PROXY_RULE) for h in HORIZONS}
    decision = choose_task(primary, proxy)
    result = {
        "schema_version": 1, "status": "complete", "fold": FOLD,
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "candidate_horizons": list(HORIZONS),
        "required_history_frames": HISTORY,
        "partition_policy": {
            "opened_episode_data": ["train"], "validation_data_opened": False,
            "calibration_data_opened": False, "test_data_opened": False,
            "note": "manifest metadata was read; only train-role label archives were opened",
        },
        "train_inventory": {"role_ids": len(train_ids), "labeled_slip_episodes": len(episode_rows),
                            "opened_label_files": len(opened_paths)},
        "primary_current_static": primary, "proxy_current_incipient": proxy,
        "decision": decision, "episode_rows": episode_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "support_audit.json", result)
    csv_rows = []
    for cohort, values in (("primary_current_static", primary), ("proxy_current_incipient", proxy)):
        for horizon, row in values.items():
            csv_rows.append({"cohort": cohort, "horizon": horizon, **{k: v for k, v in row.items() if k != "requirements"}})
    csv_path = output / "support_by_horizon.csv"
    temp = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    os.replace(temp, csv_path)
    lines = ["# Round 4 future-task train-only support audit", "",
             "Only labels from `htt_leave_p1/train` were opened. Validation, calibration, and test episode data were not opened. Counts require the complete four-frame causal model history as well as the complete future window.", "",
             "| cohort | H | positive frames | positive episodes | negative frames | negative episodes | supported |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in csv_rows:
        lines.append(f"| {row['cohort']} | {row['horizon']} | {row['positive_frames']} | {row['positive_episodes']} | {row['negative_frames']} | {row['negative_episodes']} | {row['supported']} |")
    lines += ["", "## Frozen decision", "", f"`{json.dumps(decision, ensure_ascii=False, sort_keys=True)}`", "",
              "The target is first gross onset in `(t,t+H]` with a complete future window and frames after/current gross excluded. The historical future-any target remains a separate diagnostic and is not substituted here.", ""]
    atomic_text(output / "SUPPORT_AUDIT.md", "\n".join(lines))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(audit(args.manifest.resolve(), args.output.resolve())["decision"], indent=2))


if __name__ == "__main__":
    main()
