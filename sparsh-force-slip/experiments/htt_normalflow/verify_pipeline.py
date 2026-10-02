"""CPU smoke: real adapters, masked supervised gradients, causal regression checks."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapters import collate, future_onset, iter_split_windows, load_htt, load_normalflow, window
from prepare_splits import verify


def masked_mse(pred, target, valid):
    # Index before arithmetic: NaN placeholders must never contaminate gradients.
    return F.mse_loss(pred[valid], target[valid]) if valid.any() else pred.sum() * 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260913)
    manifest = json.loads(Path(args.manifest).read_text())
    split_check = verify(manifest)
    for name in ("htt_leave_p1", "normalflow_objects"):
        first_sample = next(iter_split_windows(manifest, name, "train"))
        assert first_sample["metadata"]["episode_id"] == manifest["splits"][name]["train"][0]
        assert first_sample["metadata"]["reset"]
    modified = copy.deepcopy(manifest)
    first_id = modified["splits"]["htt_leave_p1"]["train"][0]
    row = next(r for r in modified["episodes"] if r["id"] == first_id)
    row["source_files"][next(iter(row["source_files"]))] = "wrong hash"
    try:
        next(iter_split_windows(modified, "htt_leave_p1", "train"))
    except ValueError as error:
        assert "Source content changed" in str(error)
    else:
        raise AssertionError("Changed data hash was accepted")
    selected = [next(r for r in manifest["episodes"] if r["domain"] == "htt" and r["task"] == task) for task in ("force", "slip")]
    selected.append(next(r for r in manifest["episodes"] if r["domain"] == "normalflow"))
    episodes = [load_htt(r["path"]) if r["domain"] == "htt" else load_normalflow(r["path"]) for r in selected]
    samples = [window(e, 5) for e in episodes]
    batch = collate(samples)
    assert batch["inputs"]["image"].shape == (3, 6, 320, 240)
    assert set(batch["inputs"]) == {"image", "history_valid", "history_age", "time_valid"}
    for e, sample in zip(episodes, samples):
        altered = copy.deepcopy(e)
        altered.images[6:] = 0
        if altered.force is not None:
            altered.force[:] = 123
        if altered.pose is not None:
            altered.pose[:] = 0
        # Future physical supervision must not be evaluated for this mutation.
        altered.pose = None
        other = window(altered, 5)
        assert torch.equal(sample["inputs"]["image"], other["inputs"]["image"])
        assert sample["metadata"]["indices"] == [5, 0]
        first, last = window(e, 0), window(e, len(e.images)-1)
        assert first["metadata"]["reset"] and first["inputs"]["history_valid"].tolist() == [True, False]
        assert not last["targets"]["future_onset_valid"].any()
        assert not last["targets"]["relative_pose_valid"].any()
    y = np.array([0, 0, 1, 2, 2, 0])
    risk, valid = future_onset(y, 0, (1, 3, 5))
    assert risk.tolist() == [0, 1, 1] and valid.all()
    assert not future_onset(y, 3)[1].any()
    assert not future_onset(y, 5)[1].any()  # post-event recovery isn't first onset
    assert not future_onset(None, 0)[1].any()
    assert not batch["targets"]["stage_valid"][[0, 2]].any()
    assert not batch["targets"]["force_valid"][2].any()
    assert not batch["targets"]["pose_valid"][:2].any()
    slip = episodes[1]
    # Verify frame-level stage alignment throughout the full labeled episode.
    with np.load(slip.metadata["label_path"], allow_pickle=False) as archive:
        assert np.array_equal(slip.labels, archive["sliding_labels_bracket"])
    # Add a genuinely eligible onset example, independent of the arbitrary smoke t=5.
    eligible = next((i for i in range(len(slip.images)) if future_onset(slip.labels, i)[1].any()), None)
    assert eligible is not None
    batch = collate(samples + [window(slip, eligible)])
    features = batch["inputs"]["image"].mean(dim=(-2, -1))
    heads = torch.nn.ModuleDict({"stage": torch.nn.Linear(6, 3), "force": torch.nn.Linear(6, 6),
                                 "future_force": torch.nn.Linear(6, 18), "pose": torch.nn.Linear(6, 48),
                                 "risk": torch.nn.Linear(6, 3)})
    targets = batch["targets"]
    stage_mask = targets["stage_valid"]
    losses = {"stage": F.cross_entropy(heads["stage"](features)[stage_mask], targets["stage"][stage_mask]),
              "force": masked_mse(heads["force"](features), targets["force"], targets["force_valid"]),
              "future_force": masked_mse(heads["future_force"](features).reshape(-1,3,6), targets["future_force"], targets["future_force_valid"]),
              "relative_pose": masked_mse(heads["pose"](features).reshape(-1,3,4,4), targets["relative_pose"], targets["relative_pose_valid"])}
    mask = targets["future_onset_valid"]
    losses["risk"] = F.binary_cross_entropy_with_logits(heads["risk"](features)[mask], targets["future_onset"][mask])
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters())
    # Adversarial leakage check: validator must reject a deliberately polluted split.
    broken = copy.deepcopy(manifest)
    fold = broken["splits"]["htt_leave_p1"]
    fold["train"].append(fold["test"][0])
    try:
        verify(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Leakage validator accepted contaminated split")
    result = {"status": "passed", "split_checks": split_check,
              "episodes_loaded": [e.episode_id for e in episodes],
              "batch_shape": list(batch["inputs"]["image"].shape),
              "losses": {k: float(v.detach()) for k,v in losses.items()},
              "checks": ["source hash mutation rejected", "future image and physical input independence", "current/past order", "episode start padding mask", "tail censoring", "first onset vs persistence", "missing target masks", "full slip label alignment", "finite masked backward", "reject deliberate split leakage"],
              "scope": "random tiny heads for plumbing only; no optimizer step, not model training or accuracy evaluation"}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
