"""Deterministic grouped manifests; no frame-level partitioning."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in ("tactile_img", "6d_force"):
            value = np.ascontiguousarray(data[key])
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def verify(manifest):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    rows = {r["id"]: r for r in manifest["episodes"]}
    require(len(rows) == len(manifest["episodes"]), "Duplicate episode IDs")
    for name, split in manifest["splits"].items():
        seen_ids, seen_hash, seen_groups = set(), set(), set()
        expected = {r["id"] for r in rows.values() if r["domain"] == ("htt" if name.startswith("htt") else "normalflow")}
        for part, ids in split.items():
            require(len(ids) == len(set(ids)), f"{name}/{part}: repeated IDs")
            require(not (set(ids) & seen_ids), f"{name}/{part}: episode leakage")
            hashes = {rows[i]["physical_hash"] for i in ids}
            groups = {rows[i]["leakage_group"] for i in ids}
            require(not hashes & seen_hash, f"{name}/{part}: duplicate leakage")
            require(not groups & seen_groups, f"{name}/{part}: suspected trial leakage")
            seen_ids.update(ids)
            seen_hash.update(hashes)
            seen_groups.update(groups)
        require(seen_ids == expected, f"{name}: coverage mismatch")
        if name.startswith("htt"):
            probe = name.rsplit("_", 1)[1]
            require(all(rows[i]["group"] == probe for i in split["test"]), "Held probe mismatch")
            require(all(rows[i]["group"] != probe for p, ids in split.items() if p != "test" for i in ids), "Probe leakage")
        else:
            objects = [{rows[i]["group"] for i in ids} for ids in split.values()]
            require(sum(map(len, objects)) == len(set.union(*objects)), "Object leakage")
    return {"episode_overlap": 0, "physical_hash_overlap": 0, "suspected_trial_overlap": 0, "coverage": "all", "folds": len(manifest["splits"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/vla1/zjy/tactile_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    root = Path(args.data_root)
    rows = []
    for task in ("force", "slip"):
        for path in sorted((root / "HTT-dataset" / task / "gsmini" / "processed").glob("*/*.npz")):
            with np.load(path, allow_pickle=False) as data:
                n = len(data["6d_force"])
            source_files = {str(path): file_hash(path)}
            labels = path.parent.parent.parent / "sliding_labeled" / path.parent.name / (path.stem + ".labeled.npz")
            if labels.exists():
                source_files[str(labels)] = file_hash(labels)
            rows.append({"id": f"htt/{path.parent.name}/{path.stem}", "domain": "htt",
                         "source_files": source_files,
                         "group": path.parent.name.split("_")[0], "task": task, "frames": n,
                         "leakage_group": f"htt/{path.parent.name.split('_')[0]}/{path.stem}",
                         "path": str(path), "physical_hash": fingerprint(path)})
    for path in sorted((root / "NormalFlow-dataset" / "extracted_round1").rglob("gelsight.avi")):
        obj = re.sub(r"\d+$", "", path.parent.name)
        rows.append({"id": f"normalflow/{path.parent.name}", "domain": "normalflow", "group": obj,
                     "source_files": {str(p): file_hash(p) for p in (path, path.parent / "true_start_T_currs.npy")},
                     "leakage_group": f"normalflow/{path.parent.name}",
                     "path": str(path.parent), "frames": len(np.load(path.parent / "true_start_T_currs.npy")),
                     "physical_hash": hashlib.sha256(path.read_bytes()).hexdigest()})
    if sum(r["domain"] == "htt" for r in rows) != 202 or sum(r["domain"] == "normalflow" for r in rows) != 84:
        raise ValueError("Unexpected episode inventory")
    splits = {}
    for held in ("p1", "p2", "p3", "p4"):
        partition = {p: [] for p in ("train", "validation", "calibration", "test")}
        assigned = {}
        assigned_trial = {}
        for probe in ("p1", "p2", "p3", "p4"):
            for task in ("force", "slip"):
                group = [r for r in rows if r["domain"] == "htt" and r["group"] == probe and r["task"] == task]
                random.Random(f"{args.seed}/{held}/{probe}/{task}").shuffle(group)
                n_val = max(1, round(len(group) * .15))
                for j, row in enumerate(group):
                    part = "test" if probe == held else ("validation" if j < n_val else "calibration" if j < 2 * n_val else "train")
                    # Exact duplicates always stay together. Cross-probe duplicates
                    # would invalidate the leave-probe protocol rather than hide it.
                    previous = assigned_trial.get(row["leakage_group"], assigned.get(row["physical_hash"]))
                    if previous is not None:
                        if (previous == "test") != (part == "test"):
                            raise ValueError("Cross-probe duplicate prevents this split")
                        part = previous
                    assigned[row["physical_hash"]] = part
                    assigned_trial[row["leakage_group"]] = part
                    partition[part].append(row["id"])
        splits[f"htt_leave_{held}"] = partition
    objects = sorted({r["group"] for r in rows if r["domain"] == "normalflow"})
    if len(objects) != 12:
        raise ValueError("Unexpected NormalFlow object inventory")
    random.Random(args.seed).shuffle(objects)
    object_split = {"train": objects[:8], "validation": objects[8:10], "test": objects[10:]}
    splits["normalflow_objects"] = {p: [r["id"] for r in rows if r["domain"] == "normalflow" and r["group"] in objs] for p, objs in object_split.items()}
    manifest = {"schema_version": 2, "seed": args.seed, "episodes": rows, "splits": splits,
                "normalflow_objects": object_split, "real_data_role": "development_only",
                "duplicate_check": "exact image+force array hash HTT; same-probe/stem conservatively bound regardless of uncertain acquisition identity; video SHA256 NormalFlow; other near duplicates unresolved"}
    manifest["verification"] = verify(manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError("Refusing to replace a different frozen manifest")
    else:
        output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"verification": manifest["verification"], "counts": {k: {p: len(v) for p, v in s.items()} for k,s in splits.items()}}, indent=2))


if __name__ == "__main__":
    main()
