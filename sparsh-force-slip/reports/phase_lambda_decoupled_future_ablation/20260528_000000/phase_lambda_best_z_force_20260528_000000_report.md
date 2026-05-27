# Phase4 Future Stability Head: phase_lambda_best_z_force_20260528_000000

- generated_at: `2026-05-28T04:12:25`
- feature_run_id: `phase_lambda_best_future_features_20260528_000000`
- feature_kind: `decoupled_joint`
- input_mode: `z_force`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred']`
- best_epoch: `37`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9589 | 0.9840 | 0.9763 | 0.9840 | 0.9925 | 0.0029 | 3.3929 |
| 3 | 0.9019 | 0.9582 | 0.9455 | 0.9582 | 0.9753 | 0.0114 | n/a |
| 5 | 0.8569 | 0.9329 | 0.9200 | 0.9329 | 0.9549 | 0.0176 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0283`
- slip_f1: `0.9627`
- slip_accuracy: `0.9835`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000/phase_lambda_best_z_force_20260528_000000_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_z_force_20260528_000000/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/features/feature_manifest.json`
