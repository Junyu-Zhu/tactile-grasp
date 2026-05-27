# Phase4 Future Stability Head: phase_lambda_best_z_only_20260528_000000

- generated_at: `2026-05-28T04:10:50`
- feature_run_id: `phase_lambda_best_future_features_20260528_000000`
- feature_kind: `decoupled_joint`
- input_mode: `z_only`
- input_schema: `['z_t_mean_pooled']`
- best_epoch: `39`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9450 | 0.9752 | 0.9655 | 0.9752 | 0.9880 | 0.0136 | 2.4865 |
| 3 | 0.8890 | 0.9396 | 0.9258 | 0.9396 | 0.9647 | 0.0169 | n/a |
| 5 | 0.8401 | 0.9105 | 0.8971 | 0.9105 | 0.9398 | 0.0184 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0283`
- slip_f1: `0.9627`
- slip_accuracy: `0.9835`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000/phase_lambda_best_z_only_20260528_000000_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_z_only_20260528_000000/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/features/feature_manifest.json`
