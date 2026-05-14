# Phase2-B partially shared + loss-weight follow-up summary

- generated_at: `2026-05-14T09:31:01`
- decision: `phase2_b_ps_lam025_gsmini_20260514_063052` is the first run without hard-fail; DINOv2 remains in warning band, MAE passes.
- W&B logging: train/val metrics are logged against `global_step`; epoch is stored only as `*/epoch_index` metric.

## `phase2_b_ps_gsmini_20260514_035248`

B triggers a hard fail for dinov2; Phase2-C may only be run as diagnostic-only for those encoders and must not be used for a formal improvement claim.

| encoder | gate | checkpoint epoch | force A | force B | Δ force | F1 A | F1 B | F1 drop pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | hard_fail | 51 | 0.043613 | 0.053106 | +21.77% | 0.947743 | 0.963013 | -1.53 |
| mae | warning_band | 40 | 0.032242 | 0.034102 | +5.77% | 0.973104 | 0.975274 | -0.22 |

Report: `sparsh-force-slip/reports/phase2/phase2_b_ps_gsmini_20260514_035248/phase2_b_diagnostic_report.md`

## `phase2_b_ps_lam050_gsmini_20260514_063052`

B triggers a hard fail for dinov2; Phase2-C may only be run as diagnostic-only for those encoders and must not be used for a formal improvement claim.

| encoder | gate | checkpoint epoch | force A | force B | Δ force | F1 A | F1 B | F1 drop pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | hard_fail | 25 | 0.043613 | 0.048957 | +12.25% | 0.947743 | 0.924811 | +2.29 |
| mae | pass | 35 | 0.032242 | 0.031925 | -0.98% | 0.973104 | 0.979929 | -0.68 |

Report: `sparsh-force-slip/reports/phase2/phase2_b_ps_lam050_gsmini_20260514_063052/phase2_b_diagnostic_report.md`

## `phase2_b_ps_lam025_gsmini_20260514_063052`

B is inside the warning band for dinov2; Phase2-C can proceed, but reports must explicitly mark the main-metric risk.

| encoder | gate | checkpoint epoch | force A | force B | Δ force | F1 A | F1 B | F1 drop pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | warning_band | 51 | 0.043613 | 0.047934 | +9.91% | 0.947743 | 0.971274 | -2.35 |
| mae | pass | 50 | 0.032242 | 0.031079 | -3.61% | 0.973104 | 0.977060 | -0.40 |

Report: `sparsh-force-slip/reports/phase2/phase2_b_ps_lam025_gsmini_20260514_063052/phase2_b_diagnostic_report.md`
