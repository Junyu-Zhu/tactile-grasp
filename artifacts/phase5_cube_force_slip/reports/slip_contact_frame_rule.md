# Slip contact-frame rule

- Frame: `phase5_gripper_contact_frame_v1`.
- Origin: midpoint between left and right GelSight soft-mesh AABB centers from
  each simulation sample.
- Normal axis: left-to-right gelpad center axis (closing axis).
- Tangent axes: world-up projected into the plane orthogonal to the normal plus
  its orthogonal cross-axis.
- Label: `slip_label=1` when object relative motion in the two tangent axes
  reaches at least `0.00075` m within
  `3` future sample(s).
- Valid stages: `contact_close`, `hold`, `micro_lift`.
- Masked stages: `end_trial`, `release`.
- Release/end rows are not counted as metric-valid slip labels.

This is simulation-valid for cube-only contact-frame consistency checks; it is
not a real-world tactile slip benchmark.
