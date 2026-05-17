# Phase4 Future Stability Head: phase4_1_world_seed2_20260517_144424

- generated_at: `2026-05-17T20:19:24`
- feature_run_id: `phase4_1_world_features_seed2_20260517_144424`
- feature_kind: `decoupled_joint`
- input_mode: `full`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- best_epoch: `40`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9651 | 0.9987 | 0.9962 | 0.9987 | 0.9996 | 0.0061 | 5.0729 |
| 3 | 0.9589 | 0.9972 | 0.9940 | 0.9972 | 0.9988 | 0.0041 | n/a |
| 5 | 0.9363 | 0.9873 | 0.9813 | 0.9873 | 0.9929 | 0.0062 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0307`
- slip_f1: `0.9741`
- slip_accuracy: `0.9887`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_1_20260517_144424/phase4_1_world_seed2_20260517_144424_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_1_world_features_seed2_20260517_144424/heads/phase4_1_world_seed2_20260517_144424/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_1_world_features_seed2_20260517_144424/features/feature_manifest.json`
