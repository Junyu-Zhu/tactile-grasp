#!/usr/bin/env python3
"""Metadata/development-only audit for DeformableObjectsGrasping; never opens testing content."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def slip_role(root: Path, role: str) -> dict:
    base = root / "dataset" / "Whole Dataset" / role
    rows = []
    objects = set()
    trials = set()
    labels = Counter()
    for label_file in sorted(base.glob("object*_result.dat")):
        object_id = re.search(r"object(\d+)_result", label_file.name).group(1)
        objects.add(object_id)
        for line in label_file.read_text().splitlines():
            fields = line.split()
            if len(fields) < 8:
                raise ValueError(f"malformed development label: {label_file}")
            status = int(fields[2][0])
            trial = f"object{object_id}/{fields[-3]}_mm"
            rows.append((object_id, trial, status, fields[5]))
            trials.add(trial)
            labels[status] += 1
    external = list(base.glob("object*/*_mm/external_*.jpg"))
    tactile = list(base.glob("object*/*_mm/gelsight_*.jpg"))
    external_names = {str(path).replace("external_", "{modality}_") for path in external}
    tactile_names = {str(path).replace("gelsight_", "{modality}_") for path in tactile}
    return {
        "role": role,
        "label_rows": len(rows),
        "objects": len(objects),
        "object_ids": sorted(objects),
        "trial_directories_referenced": len(trials),
        "labels": {str(key): value for key, value in sorted(labels.items())},
        "external_jpg": len(external),
        "gelsight_jpg": len(tactile),
        "paired_by_filename_substitution": len(external_names & tactile_names),
        "trials": sorted(trials),
        "rows": rows,
    }


def fruit_audit(root: Path) -> dict:
    base = root / "dataset" / "new_upload"
    fruits = {}
    all_specimens = set()
    all_trials = set()
    for fruit_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        rows = []
        labels = Counter()
        thresholds = []
        for line in (fruit_dir / "labels.txt").read_text().splitlines():
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"malformed fruit label: {fruit_dir}")
            rel, label, threshold = fields
            specimen, trial = rel.split("/", 1)
            rows.append(rel)
            labels[int(label)] += 1
            thresholds.append(float(threshold))
            all_specimens.add(f"{fruit_dir.name}/{specimen}")
            all_trials.add(f"{fruit_dir.name}/{rel}")
        modalities = {}
        for action in ("Grasping", "Sliding"):
            for sensor in ("Gelsight", "RealSense"):
                modalities[f"{action}/{sensor}"] = len(list(fruit_dir.glob(f"*/*/{action}/{sensor}/*.png")))
        fruits[fruit_dir.name] = {
            "labeled_trials": len(rows),
            "specimens": len({row.split("/")[0] for row in rows}),
            "labels": {str(key): value for key, value in sorted(labels.items())},
            "threshold_min": min(thresholds),
            "threshold_max": max(thresholds),
            "modalities": modalities,
            "force_npy": len(list(fruit_dir.glob("*/*/**/Force.npy"))),
            "position_npy": len(list(fruit_dir.glob("*/*/**/Position.npy"))),
        }
    return {"fruits": fruits, "specimens": len(all_specimens), "labeled_trials": len(all_trials)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = slip_role(args.dataset_root, "training")
    validation = slip_role(args.dataset_root, "validation")
    object_overlap = sorted(set(train["object_ids"]) & set(validation["object_ids"]))
    trial_overlap = sorted(set(train["trials"]) & set(validation["trials"]))
    testing = args.dataset_root / "dataset" / "Whole Dataset" / "testing"
    # Directory-entry metadata only: no testing file is opened, parsed, hashed, decoded, or sampled.
    testing_metadata = {
        "exists": testing.is_dir(),
        "top_level_entry_count": sum(1 for _ in testing.iterdir()) if testing.is_dir() else 0,
        "content_opened": False,
        "filenames_recorded": False,
    }
    for role in (train, validation):
        role.pop("rows")
        role.pop("trials")
    fruit = fruit_audit(args.dataset_root)
    audit = {
        "schema": "round18_deformable_objects_development_audit_v1",
        "status": "limited_development_use_only",
        "root": str(args.dataset_root.resolve()),
        "testing": testing_metadata,
        "slip_detection_subset": {
            "training": train,
            "validation": validation,
            "object_id_overlap_count": len(object_overlap),
            "object_id_overlap": object_overlap,
            "trial_directory_overlap_count": len(trial_overlap),
            "object_isolated_split": len(object_overlap) == 0,
            "trial_directory_isolated_split": len(trial_overlap) == 0,
            "input": "raw paired external RGB and GelSight RGB JPEG frames",
            "label_granularity": "one binary status per listed sequence/window; loader selects consecutive frames; not a per-frame slip onset label",
            "label_semantics": "repository loader maps 0 to slip/fail and 1 to success/non-slip",
            "sensor": "GelSight tactile plus RealSense external camera (repository documentation)",
            "marker_and_background": "not specified by labels/loader; no explicit no-contact background reference contract found",
            "timing": "paired by external/gelsight filename substitution and lexicographic order; no explicit synchronization timestamps or frame-rate contract found",
            "feature_tensor": False,
        },
        "fruit_grasping_subset": {
            **fruit,
            "input": "raw Grasping/Sliding x GelSight/RealSense PNG sequences plus Force.npy and Position.npy arrays",
            "label_granularity": "one outcome class and force-threshold scalar per Data trial",
            "label_semantics": "classes 0/1/2 are grasp-outcome categories in repository code; Sliding is an action/input sequence, not a slip ground-truth event",
            "split": "fruit-level code configuration uses five fruits for training and kiwi for test; no independent calibration/validation object role is defined",
            "timing": "modalities are sorted independently and sampled at fixed indices; filenames look timestamp-like but loader has no explicit cross-sensor synchronization check",
            "marker_and_background": "GelSight images present; marker type and explicit background/reference frame contract are not documented",
            "force_use_boundary": "force arrays exist, but this audit does not claim calibrated three-axis physical compatibility with HTT",
        },
        "applicability": {
            "frozen_HTT_direct_evaluation_ready": False,
            "reasons": [
                "HTT preprocessing/background/sensor contract is not established for these GelSight images",
                "slip labels are sequence/window level without onset timestamps, so HTT event delay cannot be computed",
                "published training/validation share object identities and are not an object-isolated external validation",
                "fruit Sliding action is not a slip label and fruit outcome labels are a different task",
                "no independent development calibration-object role is defined for threshold transfer",
            ],
            "permitted_next_use": "development-only preprocessing/label-contract study; no training or test access in Round 18",
        },
        "representative_development_contact_sheet": str(args.output.with_name("DEFORMABLE_OBJECTS_DEV_CONTACT_SHEET.jpg")),
        "independent_visual_check": {
            "reviewer": "root_agent",
            "status": "complete",
            "finding": "left column contains external-camera bottle/apple scenes; right column contains colored tactile images with visible black dot arrays, denser and more regular in the fruit examples",
            "boundary": "image appearance alone does not identify the exact GelSight/GSmini model; external-camera frames are not tactile input",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    report = f"""# DeformableObjectsGrasping development-only availability audit

Testing content was not opened. Only the testing directory entry count was observed to confirm that a held-out area exists.

The slip subset contains raw paired GelSight and RealSense JPEG sequences. Training has {train['label_rows']} labeled windows and validation has {validation['label_rows']}. Their object-ID overlap is {len(object_overlap)}; trial-directory overlap is {len(trial_overlap)}. The split is therefore trial-directory isolated in this copy but not object isolated. Each label row describes a sequence/window selected by the loader, not a per-frame onset. There is no explicit background-reference, marker-type, frame-rate, or synchronization contract.

The fruit subset contains {fruit['labeled_trials']} labeled trials over {fruit['specimens']} named specimens with raw Grasping/Sliding × GelSight/RealSense PNG streams plus Force/Position arrays. Sliding is an input action and cannot be relabeled as slip. Each row has one grasp-outcome class and force threshold for a Data trial. The code's fruit holdout is a fruit class split, with no separate calibration-object role.

The dataset is not ready for direct frozen HTT evaluation. Sensor/background preprocessing is unresolved, slip timing is too coarse for event delay, and the development split does not isolate object identity. Round 18 may use this audit to design a future preprocessing and role-lock study; it must not train, annotate, or inspect testing content.
"""
    args.output.with_suffix(".md").write_text(report)
    print(json.dumps({"status": audit["status"], "testing_content_opened": False, "slip_object_overlap": len(object_overlap), "slip_trial_overlap": len(trial_overlap), "fruit_trials": fruit["labeled_trials"]}, indent=2))


if __name__ == "__main__":
    main()
