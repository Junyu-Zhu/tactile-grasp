# Phase4 Future Stability Head: phase_lambda_best_z_p_slip_20260528_000000

- generated_at: `2026-05-28T04:11:38`
- feature_run_id: `phase_lambda_best_future_features_20260528_000000`
- feature_kind: `decoupled_joint`
- input_mode: `z_p_slip`
- input_schema: `['z_t_mean_pooled', 'p_slip_current']`
- best_epoch: `39`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9612 | 0.9900 | 0.9838 | 0.9900 | 0.9954 | 0.0088 | 6.1389 |
| 3 | 0.9173 | 0.9759 | 0.9645 | 0.9759 | 0.9876 | 0.0173 | n/a |
| 5 | 0.8723 | 0.9542 | 0.9404 | 0.9542 | 0.9722 | 0.0254 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0283`
- slip_f1: `0.9627`
- slip_accuracy: `0.9835`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000/phase_lambda_best_z_p_slip_20260528_000000_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_z_p_slip_20260528_000000/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/features/feature_manifest.json`
