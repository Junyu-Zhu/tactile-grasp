# Phase4 Future Stability Head: phase4_3_z_p_slip_20260517_171500

- generated_at: `2026-05-17T17:17:16`
- feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- feature_kind: `decoupled_joint`
- input_mode: `z_p_slip`
- input_schema: `['z_t_mean_pooled', 'p_slip_current']`
- best_epoch: `38`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9608 | 0.9892 | 0.9833 | 0.9892 | 0.9950 | 0.0057 | 5.8471 |
| 3 | 0.9175 | 0.9730 | 0.9621 | 0.9730 | 0.9857 | 0.0082 | n/a |
| 5 | 0.8790 | 0.9501 | 0.9374 | 0.9501 | 0.9688 | 0.0197 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`
- slip_accuracy: `0.9819`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_z_p_slip_20260517_171500_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/heads/phase4_3_z_p_slip_20260517_171500/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/features/feature_manifest.json`
