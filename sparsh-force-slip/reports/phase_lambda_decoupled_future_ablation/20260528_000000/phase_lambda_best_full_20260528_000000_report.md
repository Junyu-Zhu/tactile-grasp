# Phase4 Future Stability Head: phase_lambda_best_full_20260528_000000

- generated_at: `2026-05-28T04:14:01`
- feature_run_id: `phase_lambda_best_future_features_20260528_000000`
- feature_kind: `decoupled_joint`
- input_mode: `full`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- best_epoch: `40`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9654 | 0.9978 | 0.9951 | 0.9978 | 0.9989 | 0.0055 | 3.5248 |
| 3 | 0.9567 | 0.9969 | 0.9936 | 0.9969 | 0.9983 | 0.0068 | n/a |
| 5 | 0.9351 | 0.9859 | 0.9803 | 0.9859 | 0.9912 | 0.0093 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0283`
- slip_f1: `0.9627`
- slip_accuracy: `0.9835`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000/phase_lambda_best_full_20260528_000000_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/features/feature_manifest.json`
