#!/usr/bin/env python3
"""Synthetic boundary checks for round-2 first-gross support statistics."""
from __future__ import annotations

import numpy as np

from collect_event_support import episode_stats


def fake_row(frames):
    return {"id": "htt/p1_sliding/example", "group": "p1", "task": "slip", "frames": frames}


def main():
    labels = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    meta = {"params_bracket": {"incipient_window": 1, "t_start": 3, "t_stop": 4}}
    row = episode_stats("htt_leave_p2", "validation", fake_row(len(labels)), labels, meta, "/tmp/label.npz")
    assert row["first_gross_index"] == 3
    assert row["static_h1_eligible"] == 2 and row["static_h1_positive"] == 0
    assert row["static_h3_positive"] == 2
    assert row["incipient_h1_positive"] == 1
    assert row["incipient_h3_tail_masked"] == 1
    assert row["static_h3_positive_event"] == 1
    assert row["meta_first_gross_equals_t_start"] is True
    assert row["meta_last_gross_equals_t_stop"] is True
    assert row["meta_bracket_pattern_matches"] is True

    recovered = np.array([0, 1, 2, 2, 0], dtype=np.int64)
    meta = {"params_bracket": {"incipient_window": 1, "t_start": 2, "t_stop": 3}}
    row = episode_stats("htt_leave_p2", "validation", fake_row(5), recovered, meta, "/tmp/label.npz")
    assert row["frames_recovery_after_first_gross"] == 1
    assert row["meta_bracket_pattern_matches"] is True
    assert row["static_h1_eligible"] == 1  # recovered static remains post-event and is excluded

    no_event = np.array([0, 1, 1], dtype=np.int64)
    row = episode_stats("htt_leave_p2", "train", fake_row(3), no_event, {}, "/tmp/label.npz")
    assert row["no_gross"] is True
    assert row["incipient_h1_eligible"] == 1
    assert row["incipient_h1_negative"] == 1
    assert row["incipient_h1_tail_masked"] == 1

    missing = episode_stats("htt_leave_p2", "calibration", {**fake_row(4), "task": "force"}, None, None, None)
    assert missing["labels_available"] is False
    assert missing["frames_static"] == ""
    assert missing["static_h1_negative"] == ""
    print("synthetic boundary checks: passed")


if __name__ == "__main__":
    main()
