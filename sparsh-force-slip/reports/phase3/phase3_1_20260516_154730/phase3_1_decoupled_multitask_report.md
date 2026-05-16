# Phase3-1 Decoupled Multitask Report

- generated_at: `2026-05-17T01:38:57`
- run_id: `phase3_1_decoupled_gsmini_20260516_154730`
- decoder_variant: `decoupled`
- derived_dataset: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- selection_priority: force RMSE non-degradation > slip F1 retained/improved > consistency improvement > per-domain stability.

## A vs decoupled summary

| encoder | A force RMSE | decoupled force RMSE | Δ force | A slip F1 | decoupled slip F1 | F1 drop pp | slip acc | contradiction Δ | monotonic Δ | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mae | 0.0322 | 0.0304 | -5.75% | 0.9731 | 0.9652 | 0.79 | 0.9825 | 0.0002 | -0.0000 | pass |
| dinov2 | 0.0436 | 0.0451 | 3.51% | 0.9477 | 0.9488 | -0.11 | 0.9739 | 0.0107 | 0.0003 | pass |
| dino | 0.0414 | 0.0435 | 4.94% | 0.9257 | 0.9616 | -3.60 | 0.9811 | -0.0065 | -0.0013 | pass |
| ijepa | 0.0649 | 0.0695 | 7.00% | 0.9022 | 0.9438 | -4.16 | 0.9714 | 0.0029 | 0.0004 | warning_band |
| vjepa | 0.0461 | 0.0508 | 10.20% | 0.9671 | 0.9539 | 1.32 | 0.9767 | 0.0015 | -0.0012 | hard_fail |

## Best backbone decision

- best_backbone: `mae`
- best_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- reason: `lowest lexicographic rank by force non-degradation, slip F1 retention, consistency deltas, and per-domain stability`

## Per-domain stability snapshot

| encoder | domain | A force | decoupled force | Δ force | A slip F1 | decoupled slip F1 | F1 drop pp | contradiction | monotonic error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mae | flat | 0.0342 | 0.0352 | 2.85% | 0.9607 | 0.9510 | 0.97 | 0.0025 | 0.0492 |
| mae | sharp | 0.0443 | 0.0395 | -10.76% | 0.9770 | 0.9698 | 0.72 | 0.0000 | 0.0185 |
| mae | sphere | 0.0232 | 0.0224 | -3.58% | 0.9759 | 0.9687 | 0.72 | 0.0018 | 0.0019 |
| dinov2 | flat | 0.0443 | 0.0435 | -1.72% | 0.9310 | 0.9438 | -1.28 | 0.0026 | 0.0511 |
| dinov2 | sharp | 0.0616 | 0.0632 | 2.71% | 0.9657 | 0.9409 | 2.48 | 0.0212 | 0.0073 |
| dinov2 | sphere | 0.0327 | 0.0342 | 4.64% | 0.9464 | 0.9547 | -0.83 | 0.0115 | 0.0050 |
| dino | flat | 0.0411 | 0.0475 | 15.63% | 0.9124 | 0.9702 | -5.78 | 0.0000 | 0.0510 |
| dino | sharp | 0.0568 | 0.0604 | 6.41% | 0.8862 | 0.9290 | -4.27 | 0.0028 | 0.0033 |
| dino | sphere | 0.0312 | 0.0316 | 1.31% | 0.9502 | 0.9741 | -2.39 | 0.0011 | 0.0010 |
| ijepa | flat | 0.0663 | 0.0699 | 5.52% | 0.8683 | 0.9040 | -3.57 | 0.0000 | 0.0168 |
| ijepa | sharp | 0.0899 | 0.0939 | 4.44% | 0.9524 | 0.9436 | 0.88 | 0.0151 | 0.0057 |
| ijepa | sphere | 0.0487 | 0.0537 | 10.08% | 0.8912 | 0.9596 | -6.84 | 0.0070 | 0.0025 |
| vjepa | flat | 0.0535 | 0.0604 | 13.01% | 0.9606 | 0.9427 | 1.79 | 0.0029 | 0.0508 |
| vjepa | sharp | 0.0629 | 0.0710 | 12.93% | 0.9596 | 0.9514 | 0.82 | 0.0070 | 0.0046 |
| vjepa | sphere | 0.0355 | 0.0373 | 5.08% | 0.9731 | 0.9585 | 1.45 | 0.0045 | 0.0036 |

## Artifacts

- JSON report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/phase3_1_decoupled_multitask_report.json`
- Markdown report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/phase3_1_decoupled_multitask_report.md`
