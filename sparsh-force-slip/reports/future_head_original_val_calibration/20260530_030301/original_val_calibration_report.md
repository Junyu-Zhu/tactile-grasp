# Original-validation Calibrated Future Head

- source_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_future_transition/20260530_025949/heads/transition_hard_delta_balanced/checkpoints/best.pth`
- calibrated_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_future_calibrated/20260530_030301/checkpoints/best_original_val_calibrated.pth`
- calibration data: original flat/sharp/sphere validation only; real data remains validation-only.

## Chosen original-val thresholds

| horizon | threshold | bal acc | F1 | stable acc | recall |
|---|---:|---:|---:|---:|---:|
| H1 | 0.090 | 0.972 | 0.949 | 0.979 | 0.964 |
| H3 | 0.095 | 0.948 | 0.923 | 0.972 | 0.925 |
| H5 | 0.115 | 0.922 | 0.896 | 0.967 | 0.877 |

## Real comparison

| score | old stable | old slip | new stable | new slip | old gap | new gap | old F1 | new F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p_instability_H1 | 0.915 | 0.987 | 0.944 | 0.984 | 0.072 | 0.040 | 0.581 | 0.582 |
| p_instability_H3 | 0.944 | 0.991 | 0.960 | 0.984 | 0.048 | 0.025 | 0.581 | 0.578 |
| p_instability_H5 | 0.956 | 0.993 | 0.967 | 0.983 | 0.037 | 0.017 | 0.581 | 0.577 |

## Decision: insufficient_calibrated_improvement

{
  "stable_drop_H1": -0.02885236956856463,
  "gap_gain_H1": -0.03208867243496216,
  "f1_gain_H1": 0.0011934602602376998,
  "status": "insufficient_calibrated_improvement"
}
