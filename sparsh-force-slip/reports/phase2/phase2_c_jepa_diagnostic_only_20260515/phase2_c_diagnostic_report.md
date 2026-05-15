# Phase2-C JEPA diagnostic-only report

- generated_at: `2026-05-15T20:01:38`
- C training launched: **no**
- reason: all selected B checkpoints are `hard_fail`, so C is diagnostic-only by the user rule.

## Selected B checkpoints that blocked C training
| encoder | lambda | gate | force RMSE A | force RMSE B | force Δ% | slip F1 A | slip F1 B | slip Δ pp | checkpoint |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| ijepa | 0.50 | hard_fail | 0.064935 | 0.074211 | 14.29 | 0.902162 | 0.957392 | -5.52 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0045.pth` |
| vjepa | 0.25 | hard_fail | 0.046089 | 0.050712 | 10.03 | 0.967087 | 0.966653 | 0.04 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0030.pth` |

Because the force RMSE increase hard-fails for the selected JEPA B checkpoints, no consistency decoder checkpoint/W&B run exists for C.
