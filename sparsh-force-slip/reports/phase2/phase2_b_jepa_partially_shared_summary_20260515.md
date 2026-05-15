# Phase2-B JEPA partially_shared summary

- generated_at: `2026-05-15T19:32:43`
- launch_id: `phase2_b_jepa_ps_gsmini_20260515_075611`

## Lambda comparison
| encoder | lambda | gate | force RMSE A | force RMSE B | force Δ% | slip F1 A | slip F1 B | slip Δ pp | contradiction B | monotonic err B | checkpoint |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ijepa | 0.25 | hard_fail | 0.064935 | 0.073105 | 12.58 | 0.902162 | 0.952601 | -5.04 | 0.0037207266360253883 | 0.0021287462674081325 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0025.pth` |
| ijepa | 0.50 | hard_fail | 0.064935 | 0.074211 | 14.29 | 0.902162 | 0.957392 | -5.52 | 0.0018209408194233688 | 0.0002775545231997967 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0045.pth` |
| vjepa | 0.25 | hard_fail | 0.046089 | 0.050712 | 10.03 | 0.967087 | 0.966653 | 0.04 | 0.0010351966873706005 | 0.0003816230455413461 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0030.pth` |
| vjepa | 0.50 | hard_fail | 0.046089 | 0.052817 | 14.60 | 0.967087 | 0.949516 | 1.76 | 0.0014331780723754925 | 0.0006332227378152311 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |

## Selected best B for C
- **ijepa**: lambda `0.50`, run `phase2_b_jepa_lam050_gsmini_20260515_075611`, gate `hard_fail`, checkpoint `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0045.pth`
- **vjepa**: lambda `0.25`, run `phase2_b_jepa_lam025_gsmini_20260515_075611`, gate `hard_fail`, checkpoint `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0030.pth`
