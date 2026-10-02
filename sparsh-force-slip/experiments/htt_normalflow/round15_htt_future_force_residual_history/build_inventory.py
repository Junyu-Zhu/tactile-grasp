#!/usr/bin/env python3
"""Build exact Round-15 run/checkpoint and remote-artifact inventories."""
import argparse
import csv
import datetime
import json
from pathlib import Path

import future_train as ft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    runs = []
    checkpoints = []
    for group in ft.GROUPS:
        for fold in range(1, 5):
            for seed in ft.SEEDS:
                name = f"{group}_p{fold}_s{seed}"
                formal = arguments.remote_root / "formal" / name
                evaluation = arguments.remote_root / "evaluation" / name
                summary = json.loads((formal / "summary.json").read_text())
                record = {"run": name, "group": group, "fold": fold, "seed": seed, "status": summary["status"],
                          "epochs": summary["epochs"], "best_epoch": summary["best_epoch"],
                          "best_metric_native_n_mae": summary["best_metric_native_n_mae"],
                          "formal_summary": str(formal / "summary.json"), "formal_summary_sha256": ft.sha(formal / "summary.json"),
                          "best_checkpoint": str(formal / "best.pth"), "best_sha256": ft.sha(formal / "best.pth"),
                          "latest_checkpoint": str(formal / "latest.pth"), "latest_sha256": ft.sha(formal / "latest.pth"),
                          "evaluation_summary": str(evaluation / "SUMMARY.json"), "evaluation_summary_sha256": ft.sha(evaluation / "SUMMARY.json")}
                runs.append(record)
                checkpoints.append({key: record[key] for key in ("run", "group", "fold", "seed", "best_checkpoint", "best_sha256", "latest_checkpoint", "latest_sha256")})
    ft.atomic_json({"schema": "round15_run_inventory_v1", "status": "complete", "formal_runs": len(runs),
                    "accepted_runs": sum(record["status"] == "complete" for record in runs), "runs": runs}, arguments.output / "RUN_INVENTORY.json")
    with (arguments.output / "CHECKPOINT_INDEX.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoints[0])); writer.writeheader(); writer.writerows(checkpoints)
    indexed_roots = ("formal", "evaluation", "diagnostic", "reporting", "failure_case", "final_report", "logs")
    files = []
    for root_name in indexed_roots:
        root = arguments.remote_root / root_name
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append({"relative_path": path.relative_to(arguments.remote_root).as_posix(), "bytes": path.stat().st_size, "sha256": ft.sha(path)})
    remote = {"schema": "round15_remote_artifact_index_v1", "status": "complete",
              "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "remote_root": str(arguments.remote_root), "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files),
              "files": files, "large_files_remain_server_only": True}
    ft.atomic_json(remote, arguments.output / "REMOTE_ARTIFACT_INDEX.json")
    print(json.dumps({"status": "complete", "runs": len(runs), "remote_files": len(files), "remote_bytes": remote["total_bytes"]}))


if __name__ == "__main__":
    main()
