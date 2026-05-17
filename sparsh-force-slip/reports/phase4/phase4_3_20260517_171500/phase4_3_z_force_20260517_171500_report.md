# Phase4 Future Stability Head: phase4_3_z_force_20260517_171500

- generated_at: `2026-05-17T17:18:11`
- feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- feature_kind: `decoupled_joint`
- input_mode: `z_force`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred']`
- best_epoch: `40`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9572 | 0.9851 | 0.9778 | 0.9851 | 0.9930 | 0.0165 | 6.3913 |
| 3 | 0.9028 | 0.9621 | 0.9494 | 0.9621 | 0.9778 | 0.0419 | n/a |
| 5 | 0.8656 | 0.9389 | 0.9252 | 0.9389 | 0.9599 | 0.0509 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`
- slip_accuracy: `0.9819`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_z_force_20260517_171500_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/heads/phase4_3_z_force_20260517_171500/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/features/feature_manifest.json`
