# Phase2-B Quick Checkpoint Selection Analysis

- generated_at: `2026-05-14T03:27:09`
- run_id: `phase2_b_gsmini_20260513_163448`
- method: quick scan of training `history.json`; no checkpoint re-evaluation.
- policy: first require `force_rmse_increase_pct <= 10` and `slip_f1_drop_pp <= 2`, then choose the highest `slip_f1`.

## dinov2

- A force RMSE mean N: `0.043613`
- A slip F1: `0.947743`
- feasible checkpoints under force<=A+10% and slip F1 drop<=2pp: `0`

### Selected by quick policy

- none: no checkpoint satisfies the force <= A+10% filter.
- best force checkpoint: `epoch-0050.pth` with force RMSE `0.052603` (+20.61%) and slip F1 `0.958339`.
- best F1 checkpoint: `epoch-0025.pth` with force RMSE `0.056630` (+29.85%) and slip F1 `0.969345`.

### Validation checkpoints

| epoch | force RMSE N | force Δ vs A | slip F1 | F1 drop pp | gate |
|---:|---:|---:|---:|---:|---|
| 5 | 0.085869 | +96.89% | 0.928748 | +1.90 | hard_fail |
| 10 | 0.067341 | +54.41% | 0.931164 | +1.66 | hard_fail |
| 15 | 0.060540 | +38.81% | 0.941544 | +0.62 | hard_fail |
| 20 | 0.064352 | +47.55% | 0.968203 | -2.05 | hard_fail |
| 25 | 0.056630 | +29.85% | 0.969345 | -2.16 | hard_fail |
| 30 | 0.059653 | +36.78% | 0.932234 | +1.55 | hard_fail |
| 35 | 0.055073 | +26.28% | 0.964324 | -1.66 | hard_fail |
| 40 | 0.055990 | +28.38% | 0.959306 | -1.16 | hard_fail |
| 45 | 0.056451 | +29.44% | 0.942481 | +0.53 | hard_fail |
| 50 | 0.052603 | +20.61% | 0.958339 | -1.06 | hard_fail |
| 51 | 0.057883 | +32.72% | 0.958361 | -1.06 | hard_fail |

## mae

- A force RMSE mean N: `0.032243`
- A slip F1: `0.973104`
- feasible checkpoints under force<=A+10% and slip F1 drop<=2pp: `2`

### Selected by quick policy

- checkpoint: `epoch-0051.pth`
- force RMSE mean N: `0.034337` (+6.50%)
- slip F1: `0.976561` (drop -0.35 pp; negative means improvement)
- slip recall: `0.978816`
- gate: `warning_band`

### Validation checkpoints

| epoch | force RMSE N | force Δ vs A | slip F1 | F1 drop pp | gate |
|---:|---:|---:|---:|---:|---|
| 5 | 0.055549 | +72.28% | 0.971998 | +0.11 | hard_fail |
| 10 | 0.053309 | +65.34% | 0.954211 | +1.89 | hard_fail |
| 15 | 0.042191 | +30.85% | 0.968165 | +0.49 | hard_fail |
| 20 | 0.040734 | +26.34% | 0.957750 | +1.54 | hard_fail |
| 25 | 0.040506 | +25.63% | 0.974497 | -0.14 | hard_fail |
| 30 | 0.041077 | +27.40% | 0.973869 | -0.08 | hard_fail |
| 35 | 0.038807 | +20.36% | 0.969941 | +0.32 | hard_fail |
| 40 | 0.037206 | +15.39% | 0.959481 | +1.36 | hard_fail |
| 45 | 0.035026 | +8.63% | 0.956144 | +1.70 | warning_band |
| 50 | 0.036397 | +12.88% | 0.978400 | -0.53 | hard_fail |
| 51 | 0.034337 | +6.50% | 0.976561 | -0.35 | warning_band |

## Quick conclusion

- DINOv2: quick selector cannot rescue B; every saved validation checkpoint still exceeds the +10% force hard-fail threshold.
- MAE: quick selector finds `epoch-0051.pth` as a viable non-hard-fail candidate; it is `warning_band` because force increase is +6.50% while slip F1 is 0.976561.
- Recommendation: if continuing, strictly re-evaluate the selected MAE checkpoint and DINOv2 best-force checkpoint before changing the formal gate report.
