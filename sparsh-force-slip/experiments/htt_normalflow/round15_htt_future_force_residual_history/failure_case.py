#!/usr/bin/env python3
"""Fixed unfavorable Round-15 failure-case delivery."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import future_train as ft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--force-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows = []
    payloads = {}
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                run = f"{group}_p{fold}_s{seed}"
                payload = torch.load(arguments.evaluation / run / "predictions_validation.pt", map_location="cpu", weights_only=False)
                episode_ids = np.array(payload["episode_id"])
                payloads[(group, fold, seed)] = payload
                for episode in sorted(set(episode_ids)):
                    mask = episode_ids == episode
                    neural = float((payload["predictions"]["neural"][mask] - payload["y"][mask]).abs().mean())
                    persistence = float((payload["predictions"]["predicted_current_persistence"][mask] - payload["y"][mask]).abs().mean())
                    rows.append({"group": group, "fold": fold, "seed": seed, "episode_id": episode,
                                 "endpoints": int(mask.sum()), "neural_future_mae_n": neural,
                                 "persistence_future_mae_n": persistence, "neural_minus_persistence_n": neural - persistence})
    rows.sort(key=lambda row: (-row["neural_minus_persistence_n"], row["group"], row["fold"], row["seed"], row["episode_id"]))
    selected = rows[0]
    payload = payloads[(selected["group"], selected["fold"], selected["seed"])]
    mask = np.array(payload["episode_id"]) == selected["episode_id"]
    times = payload["t"][mask].numpy()
    gt = payload["y"][mask, 2].numpy()
    neural = payload["predictions"]["neural"][mask, 2].numpy()
    persistence = payload["predictions"]["predicted_current_persistence"][mask, 2].numpy()
    with (arguments.output / "failure_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    curve = []
    for index, time_index in enumerate(times):
        for axis_index, axis in enumerate(("fx", "fy", "fz")):
            curve.append({"t": int(time_index), "target_t_plus_10": float(gt[index, axis_index]), "axis": axis,
                          "neural": float(neural[index, axis_index]), "predicted_current_persistence": float(persistence[index, axis_index])})
    with (arguments.output / "selected_failure_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0])); writer.writeheader(); writer.writerows(curve)
    figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for axis_index, (axis, label) in enumerate(zip(axes, ("Fx", "Fy", "Fz"))):
        axis.plot(times, gt[:, axis_index], label="GT(t+10)", lw=1.5)
        axis.plot(times, neural[:, axis_index], label="Residual neural", lw=1)
        axis.plot(times, persistence[:, axis_index], label="Pred-current persistence", lw=1)
        axis.set_ylabel(label + " (N)"); axis.grid(alpha=0.2)
    axes[0].legend(ncol=3, frameon=False); axes[-1].set_xlabel("Current endpoint t (frame)")
    figure.suptitle(f"Fixed worst gap: {selected['group']} p{selected['fold']} s{selected['seed']} {selected['episode_id']}")
    figure.tight_layout(); figure.savefig(arguments.output / "selected_failure_curve.svg"); plt.close(figure)
    frame_gap = np.abs(neural - gt).mean(1) - np.abs(persistence - gt).mean(1)
    index = int(np.argmax(frame_gap))
    support = json.load(open(arguments.force_support / f"fold_p{selected['fold']}.json"))
    entry = next(item for item in support["entries"] if item["episode_id"] == selected["episode_id"])
    source = Path(entry["source_path"])
    with np.load(source, allow_pickle=False) as raw:
        images = np.asarray(raw["tactile_img"])
        current_image = images[int(times[index])]
        future_image = images[int(times[index]) + 10]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(current_image); axes[0].set_title(f"Current tactile frame t={int(times[index])}")
    axes[1].imshow(future_image); axes[1].set_title(f"Source frame t+10={int(times[index]) + 10}")
    for axis in axes: axis.axis("off")
    figure.suptitle(selected["episode_id"]); figure.tight_layout()
    figure.savefig(arguments.output / "selected_failure_source_frames.png", dpi=160); plt.close(figure)
    receipt = {"schema": "round15_failure_case_v1", "status": "complete",
               "selection_timing": "rule pre-registered; selection performed post-result and is descriptive",
               "fixed_rule": "Across all validation runs and complete episodes in both R15 groups, select largest episode mean neural-minus-predicted-current-persistence future MAE over endpoints, axes, horizons; lexical identity tie-break. Within it use earliest argmax h10 frame gap.",
               "candidate_count": len(rows), "selected": {**selected, "source_endpoint_t": int(times[index]), "source_future_t": int(times[index]) + 10, "frame_gap_n": float(frame_gap[index])},
               "source_prediction": "evaluation/<group>_p<fold>_s<seed>/predictions_validation.pt",
               "source_npz": str(source), "source_npz_sha256": ft.sha(source),
               "outputs": {path.name: ft.sha(path) for path in arguments.output.iterdir() if path.is_file() and path.name != "FAILURE_CASE_AUDIT.json"},
               "interpretation": "One deterministic unfavorable case is not population evidence."}
    ft.atomic_json(receipt, arguments.output / "FAILURE_CASE_AUDIT.json")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
