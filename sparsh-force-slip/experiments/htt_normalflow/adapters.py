"""Episode-first adapters. Observations and supervised targets stay separate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


@dataclass
class Episode:
    episode_id: str
    domain: str
    group: str
    images: np.ndarray
    reference: np.ndarray | None
    force: np.ndarray | None
    pose: np.ndarray | None
    labels: np.ndarray | None
    time: np.ndarray | None
    metadata: dict

    def validate(self):
        n = len(self.images)
        if n == 0 or self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError("Expected nonempty THWC images")
        for key in ("force", "pose", "labels", "time"):
            value = getattr(self, key)
            if value is not None and len(value) != n:
                raise ValueError(f"{self.episode_id}: {key} alignment")
        if self.labels is not None and not np.isin(self.labels, [0, 1, 2]).all():
            raise ValueError("Unknown slip stage")
        if self.time is not None and not (np.isfinite(self.time).all() and (np.diff(self.time) > 0).all()):
            raise ValueError("Invalid time axis")
        return self


def load_htt(path: str | Path) -> Episode:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        images = data["tactile_img"]
        reference = data["ref_frame"]
        force = data["6d_force"].astype(np.float32)
        ref_force = data["ref_force"].tolist()
        raw_probe = data["probe"].item()
        probe = path.parent.name.split("_")[0]
        mode = str(data["mode"].item())
    if not np.isfinite(reference).all() or reference.min() < 0 or reference.max() > 255:
        raise ValueError("Reference is not finite in 0..255")
    # No automatic 0..1 scaling: preserve the audited raw 0..255 float reference.
    label_path = path.parent.parent.parent / "sliding_labeled" / path.parent.name / (path.stem + ".labeled.npz")
    labels, meta = None, {}
    if label_path.exists():
        with np.load(label_path, allow_pickle=False) as data:
            labels = data["sliding_labels_bracket"].astype(np.int64)
            meta = {"labeling_meta": str(data["labeling_meta"].item())}
    return Episode(
        f"htt/{path.parent.name}/{path.stem}", "htt_gsmini", probe,
        images, reference, force, None, labels, None,
        {"path": str(path), "probe_id": probe, "raw_probe": raw_probe, "object_id": None, "mode": mode,
         "label_source": "sliding_labels_bracket" if labels is not None else None,
         "label_path": str(label_path) if labels is not None else None,
         "time_source": "unknown", "force_convention": "raw ATI 6D; axes/units unverified",
         "ref_force": ref_force, "contact_validity": "unavailable", **meta},
    ).validate()


def load_normalflow(path: str | Path) -> Episode:
    path = Path(path)
    capture = cv2.VideoCapture(str(path / "gelsight.avi"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    import re
    obj = re.sub(r"\d+$", "", path.name)
    time = np.arange(len(frames), dtype=np.float64) / fps if np.isfinite(fps) and fps > 0 else None
    return Episode(
        f"normalflow/{path.name}", "normalflow_gsmini", obj,
        np.asarray(frames), None, None,
        np.load(path / "true_start_T_currs.npy", allow_pickle=False).astype(np.float32),
        None, time,
        {"path": str(path), "object_id": obj, "probe_id": None,
         "label_source": None, "time_source": "nominal_video_fps" if time is not None else "unknown",
         "fps": float(fps), "pose_source": "true_start_T_currs.npy",
         "pose_convention": "as supplied; translation units/frame direction require verification",
         "reference_source": "absent; no background subtraction",
         "contact_validity": "unavailable; image-derived masks are not independent truth"},
    ).validate()


def preprocess(image, reference=None, sparsh_repo="/home/zjy/document/sparsh"):
    if sparsh_repo not in sys.path:
        sys.path.insert(0, sparsh_repo)
    from tactile_ssl.data.digit.utils import get_resize_transform, load_sample_from_buf
    return get_resize_transform((320, 240))(load_sample_from_buf(image, reference))


def future_onset(labels, t, horizons=(1, 3, 5)):
    """First gross onset, evaluated separately for currently static/incipient frames.

    Full-horizon masking is deliberately conservative, even if an event was
    observed before a truncated tail. Already-gross and post-first-event excluded.
    """
    targets = np.full(len(horizons), np.nan, dtype=np.float32)
    valid = np.zeros(len(horizons), dtype=bool)
    if labels is None:
        return targets, valid
    if np.any(labels[:t + 1] == 2):
        return targets, valid
    for j, horizon in enumerate(horizons):
        if horizon < 1:
            raise ValueError("Horizons must be positive")
        if t + horizon < len(labels):
            valid[j] = True
            targets[j] = float(np.any(labels[t + 1:t + horizon + 1] == 2))
    return targets, valid


def window(episode: Episode, t: int, history=2, stride=5, horizons=(1, 3, 5)):
    """Current-first MAE history; targets may contain future supervision only."""
    if not 0 <= t < len(episode.images) or history < 1 or stride < 1:
        raise ValueError("Invalid window")
    indices = t - np.arange(history) * stride
    history_valid = indices >= 0
    indices = np.maximum(indices, 0)
    image = torch.cat([preprocess(episode.images[i], episode.reference) for i in indices])
    risk, risk_valid = future_onset(episode.labels, t, horizons)
    force = np.full(6, np.nan, np.float32) if episode.force is None else episode.force[t]
    pose = np.full((4, 4), np.nan, np.float32) if episode.pose is None else episode.pose[t]
    future_force = np.full((len(horizons), 6), np.nan, np.float32)
    relative_pose = np.full((len(horizons), 4, 4), np.nan, np.float32)
    for j, horizon in enumerate(horizons):
        if t + horizon < len(episode.images):
            if episode.force is not None:
                future_force[j] = episode.force[t + horizon]
            if episode.pose is not None and np.isfinite(episode.pose[[t, t+horizon]]).all():
                relative_pose[j] = np.linalg.solve(episode.pose[t], episode.pose[t + horizon])
    dt = np.full(history, np.nan, np.float32)
    if episode.time is not None:
        dt = (episode.time[t] - episode.time[indices]).astype(np.float32)
    return {
        "inputs": {"image": image, "history_valid": torch.tensor(history_valid),
                   "history_age": torch.tensor(dt), "time_valid": torch.tensor(np.isfinite(dt))},
        "targets": {"stage": torch.tensor(-1 if episode.labels is None else int(episode.labels[t])),
                    "stage_valid": torch.tensor(episode.labels is not None),
                    "force": torch.tensor(force), "force_valid": torch.tensor(np.isfinite(force)),
                    "pose": torch.tensor(pose), "pose_valid": torch.tensor(np.isfinite(pose).all()),
                    "future_force": torch.tensor(future_force),
                    "future_force_valid": torch.tensor(np.isfinite(future_force)),
                    "relative_pose": torch.tensor(relative_pose),
                    "relative_pose_valid": torch.tensor(np.isfinite(relative_pose).all(axis=(1, 2))),
                    "future_onset": torch.tensor(risk), "future_onset_valid": torch.tensor(risk_valid)},
        "metadata": {"episode_id": episode.episode_id, "domain": episode.domain, "group": episode.group,
                     "object_id": episode.metadata.get("object_id"), "probe_id": episode.metadata.get("probe_id"),
                     "label_source": episode.metadata.get("label_source"), "path": episode.metadata["path"],
                     "t": t, "indices": indices.tolist(), "reset": t == 0,
                     "time_source": episode.metadata["time_source"]},
    }


def collate(samples):
    return {"inputs": {k: torch.stack([s["inputs"][k] for s in samples]) for k in samples[0]["inputs"]},
            "targets": {k: torch.stack([s["targets"][k] for s in samples]) for k in samples[0]["targets"]},
            "metadata": [s["metadata"] for s in samples]}


def iter_split_windows(manifest, split_name, partition, history=2, stride=5, horizons=(1, 3, 5)):
    """Read only an explicitly chosen frozen partition, one episode at a time.

    Consumers must pass only batch['inputs'] to a deployable model. This basic
    generator favors inspectability; feature caching belongs in the next round.
    """
    rows = {row["id"]: row for row in manifest["episodes"]}
    for episode_id in manifest["splits"][split_name][partition]:
        row = rows[episode_id]
        from prepare_splits import file_hash
        if not row.get("source_files"):
            raise ValueError("Manifest lacks source-file hashes")
        for source, expected_hash in row["source_files"].items():
            if file_hash(source) != expected_hash:
                raise ValueError(f"Source content changed: {episode_id}: {source}")
        episode = load_htt(row["path"]) if row["domain"] == "htt" else load_normalflow(row["path"])
        if episode.episode_id != episode_id or len(episode.images) != row["frames"] or episode.group != row["group"]:
            raise ValueError(f"Manifest mismatch: {episode_id}")
        for t in range(len(episode.images)):
            yield window(episode, t, history, stride, horizons)
