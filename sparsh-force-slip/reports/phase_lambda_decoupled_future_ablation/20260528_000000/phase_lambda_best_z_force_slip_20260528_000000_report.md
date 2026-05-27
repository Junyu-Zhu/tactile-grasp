# Phase4 Future Stability Head: phase_lambda_best_z_force_slip_20260528_000000

- generated_at: `2026-05-28T04:13:13`
- feature_run_id: `phase_lambda_best_future_features_20260528_000000`
- feature_kind: `decoupled_joint`
- input_mode: `z_force_slip`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current']`
- best_epoch: `40`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9631 | 0.9908 | 0.9845 | 0.9908 | 0.9963 | 0.0101 | 4.7313 |
| 3 | 0.9157 | 0.9777 | 0.9652 | 0.9777 | 0.9895 | 0.0202 | n/a |
| 5 | 0.8713 | 0.9564 | 0.9415 | 0.9564 | 0.9746 | 0.0296 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0283`
- slip_f1: `0.9627`
- slip_accuracy: `0.9835`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000/phase_lambda_best_z_force_slip_20260528_000000_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_z_force_slip_20260528_000000/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/features/feature_manifest.json`
