# Phase4 Future Stability Head: phase4_1_world_seed1_20260517_144424

- generated_at: `2026-05-17T20:08:33`
- feature_run_id: `phase4_1_world_features_seed1_20260517_144424`
- feature_kind: `decoupled_joint`
- input_mode: `full`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- best_epoch: `36`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9670 | 0.9978 | 0.9954 | 0.9978 | 0.9986 | 0.0038 | 5.3153 |
| 3 | 0.9549 | 0.9968 | 0.9934 | 0.9968 | 0.9978 | 0.0080 | n/a |
| 5 | 0.9349 | 0.9849 | 0.9789 | 0.9849 | 0.9901 | 0.0208 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0302`
- slip_f1: `0.9722`
- slip_accuracy: `0.9879`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_1_20260517_144424/phase4_1_world_seed1_20260517_144424_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_1_world_features_seed1_20260517_144424/heads/phase4_1_world_seed1_20260517_144424/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_1_world_features_seed1_20260517_144424/features/feature_manifest.json`
