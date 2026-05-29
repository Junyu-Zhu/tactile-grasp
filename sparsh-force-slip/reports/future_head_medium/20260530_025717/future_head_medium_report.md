# Medium Future-instability Head Report

- generated_at: `2026-05-30T02:58:17`
- report_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717`
- run_dir: `/vla1/zjy/sparsh_runs/force_slip_future_medium/20260530_025717`
- real-data policy: real banana/cucumber/hammer/lemon data is validation-only; no training, checkpoint selection, or final threshold selection uses it.

## Original flat/sharp/sphere validation selection

| tag | schema | hard neg | mean AUPRC | mean AUROC | stable mean | gap | stable sat>=0.99 | ckpt |
|---|---|---:|---:|---:|---:|---:|---:|---|
| horizon_fix_best | z,aux | previous | 0.9574 | 0.9665 | 0.4882 | 0.4583 | 0.0001 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/sharp_sphere_to_flat_balanced_bce_smooth/checkpoints/best.pth` |
| medium_hard_aux_balanced | z,aux | True | 0.9745 | 0.9825 | 0.0665 | 0.7938 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_medium/20260530_025717/heads/medium_hard_aux_balanced/checkpoints/best.pth` |
| medium_hard_delta_balanced | z,z_delta_prev,aux | True | 0.9723 | 0.9819 | 0.0303 | 0.8174 | 0.0001 | `/vla1/zjy/sparsh_runs/force_slip_future_medium/20260530_025717/heads/medium_hard_delta_balanced/checkpoints/best.pth` |

## Real future-label validation comparison

| score | previous stable | previous slip | new stable | new slip | previous gap | new gap | previous F1@0.5 | new F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p_instability_H1 | 0.9153 | 0.9871 | 0.8222 | 0.9698 | 0.0718 | 0.1477 | 0.5808 | 0.5992 |
| p_instability_H3 | 0.9436 | 0.9912 | 0.8108 | 0.9619 | 0.0476 | 0.1512 | 0.5808 | 0.5896 |
| p_instability_H5 | 0.9558 | 0.9931 | 0.8348 | 0.9645 | 0.0372 | 0.1297 | 0.5808 | 0.5848 |

## Decision

- improvement_status: **insufficient_improvement**
- reason: Small medium head did not reduce real stable-window future-instability enough; skip medium ablations and proceed to transition-label/full redefinition.

## Files

- model_selection_summary: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/medium_model_selection_summary.csv`
- real_report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_model_test_report.md`
- quick_analysis: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/quick_real_future_analysis.md`
