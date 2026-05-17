# Phase4 Future Stability Head: phase4_2_separate_late_fusion_20260517_171500

- generated_at: `2026-05-17T17:26:45`
- feature_run_id: `phase4_2_separate_features_20260517_171500`
- feature_kind: `separate_late_fusion`
- input_mode: `z_force_slip`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current']`
- best_epoch: `36`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9644 | 0.9902 | 0.9849 | 0.9902 | 0.9958 | 0.0068 | 3.6180 |
| 3 | 0.9203 | 0.9731 | 0.9625 | 0.9731 | 0.9859 | 0.0115 | n/a |
| 5 | 0.8800 | 0.9503 | 0.9390 | 0.9503 | 0.9692 | 0.0131 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0300`
- slip_f1: `0.9684`
- slip_accuracy: `0.9861`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_2_20260517_171500/phase4_2_separate_late_fusion_20260517_171500_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_2_separate_features_20260517_171500/heads/phase4_2_separate_late_fusion_20260517_171500/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_2_separate_features_20260517_171500/features/feature_manifest.json`
