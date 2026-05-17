# Phase6-3 Frozen vs Joint Training Ablation

- generated_at: `2026-05-18T02:03:35`
- raw_data_modified: `False`

## Comparison

| condition | force RMSE | slip F1 | H1 F1 | H1 AUPRC | H1 ECE | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC | checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frozen_two_stage_full_plus_q | 0.0284 | 0.9593 | 0.9668 | 0.9959 | 0.0053 | 0.9593 | 0.9945 | 0.9381 | 0.9815 | `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_3_friction_features_20260518_0154/heads/phase6_3_frozen_full_plus_q_seed42_20260518_0154/checkpoints/best.pth` |
| joint_lightweight_force_slip_future | 0.0734 | 0.9816 | 0.9838 | 0.9988 | 0.0036 | 0.9649 | 0.9956 | 0.9374 | 0.9816 | `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_3_friction_features_20260518_0154/heads/phase6_3_joint_lightweight_seed42_20260518_0154/checkpoints/best.pth` |

## Interpretation

The frozen two-stage route keeps the Phase4 decoupled current-task metrics (force RMSE=0.0284) while the lightweight joint head reaches force RMSE=0.0734. Use this as design-justification evidence rather than a full end-to-end finetuning claim.

## Notes

- Joint training is a lightweight frozen-feature proxy that jointly supervises force, slip, and future stability heads.
- The ablation tests whether coupling all losses immediately is preferable to preserving a trained perception module before dynamics learning.
