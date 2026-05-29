# Transition-redefined Future-instability Head Report

- generated_at: `2026-05-30T03:01:09`
- report_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949`
- run_dir: `/vla1/zjy/sparsh_runs/force_slip_future_transition/20260530_025949`
- real-data policy: real banana/cucumber/hammer/lemon data is validation-only; no training, checkpoint selection, or final threshold selection uses it.

## Original flat/sharp/sphere validation selection

| tag | schema | hard neg | mean AUPRC | mean AUROC | stable mean | gap | stable sat>=0.99 | ckpt |
|---|---|---:|---:|---:|---:|---:|---:|---|
| horizon_fix_best | z,aux | previous | 0.9574 | 0.9665 | 0.4882 | 0.4583 | 0.0001 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/sharp_sphere_to_flat_balanced_bce_smooth/checkpoints/best.pth` |
| transition_hard_aux_balanced | z,aux | True | 0.9721 | 0.9810 | 0.0448 | 0.8067 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_transition/20260530_025949/heads/transition_hard_aux_balanced/checkpoints/best.pth` |
| transition_hard_delta_balanced | z,z_delta_prev,aux | True | 0.9732 | 0.9825 | 0.0387 | 0.8218 | 0.0001 | `/vla1/zjy/sparsh_runs/force_slip_future_transition/20260530_025949/heads/transition_hard_delta_balanced/checkpoints/best.pth` |

## Real future-label validation comparison

| score | previous stable | previous slip | new stable | new slip | previous gap | new gap | previous F1@0.5 | new F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p_instability_H1 | 0.9153 | 0.9871 | 0.7695 | 0.9331 | 0.0718 | 0.1636 | 0.5808 | 0.6031 |
| p_instability_H3 | 0.9436 | 0.9912 | 0.8068 | 0.9370 | 0.0476 | 0.1302 | 0.5808 | 0.5939 |
| p_instability_H5 | 0.9558 | 0.9931 | 0.8446 | 0.9386 | 0.0372 | 0.0940 | 0.5808 | 0.5901 |

## Decision

- improvement_status: **insufficient_transition_improvement**
- reason: Transition redefinition did not reduce real stable-window future-instability enough; further optimization is still required.

## Files

- model_selection_summary: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/transition_model_selection_summary.csv`
- real_report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/real_zero_shot_eval/20260530_030018/real_force_slip_model_test_report.md`
- quick_analysis: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/quick_real_future_analysis.md`
