# Phase3-0 Sanity + DINO ABC Report

- generated_at: `2026-05-16T15:34:25`
- stamp: `20260516_023557`
- derived_dataset: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- manual_push_pull: `false`

## Force-only multitask sanity

| encoder | A force RMSE N | force-only RMSE N | Δ vs A | gate | checkpoint |
|---|---:|---:|---:|---|---|
| mae | 0.0322 | 0.0304 | -5.51% | pass | `epoch-0050.pth` |
| dino | 0.0414 | 0.0432 | 4.36% | pass | `epoch-0030.pth` |
| dinov2 | 0.0436 | 0.0429 | -1.69% | pass | `epoch-0025.pth` |
| ijepa | 0.0649 | 0.0696 | 7.15% | warning_band | `epoch-0035.pth` |
| vjepa | 0.0461 | 0.0483 | 4.83% | pass | `epoch-0045.pth` |

## DINO ABC

| stage | force RMSE N | slip F1 | gate/status | run/checkpoint |
|---|---:|---:|---|---|
| A | 0.0414 | 0.9259 | baseline | `/vla1/zjy/sparsh_runs/experiments/2026.05.16_07-29_phase3_0_dino_a_gsmini_20260516_023557_dino_force` |
| B λ=0.25 | 0.0458 | 0.9482 | hard_fail | `epoch-0035.pth` |
| B λ=0.50 | 0.0530 | 0.9629 | hard_fail | `epoch-0051.pth` |
| C | n/a | n/a | diagnostic_only_skipped_b_hard_fail | `None` |

## A/B/C summary across backbones

| encoder | A force | A slip F1 | B force | B slip F1 | B gate | C force | C slip F1 | C status |
|---|---:|---:|---:|---:|---|---:|---:|---|
| mae | 0.0322 | 0.9731 | 0.0319 | 0.9799 | pass | 0.0335 | 0.9819 | pass |
| dino | 0.0414 | 0.9259 | 0.0458 | 0.9482 | hard_fail | n/a | n/a | diagnostic_only_skipped_b_hard_fail |
| dinov2 | 0.0436 | 0.9477 | 0.0476 | 0.9622 | warning_band | 0.0480 | 0.9665 | warning_band |
| ijepa | 0.0649 | 0.9022 | 0.0742 | 0.9574 | hard_fail | n/a | n/a | not_run_b_hard_fail |
| vjepa | 0.0461 | 0.9671 | 0.0507 | 0.9667 | hard_fail | n/a | n/a | not_run_b_hard_fail |

## Gate conclusion

P3-0 completed. Use force-only sanity gates plus DINO ABC rows to decide whether P3-1 decoupled multitask may proceed.
