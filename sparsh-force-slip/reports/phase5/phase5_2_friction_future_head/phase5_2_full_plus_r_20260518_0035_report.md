# Phase5 Head Report: full_plus_r

- generated_at: `2026-05-18T00:33:51`
- feature_run_id: `phase5_2_friction_features_20260518_0035`
- experiment_name: `phase5_2_full_plus_r_20260518_0035`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_r_clipped_pred']`
- train_count / val_count: `65938` / `14328`
- best_epoch: `40`
- selection_score_mean_auprc: `0.9896`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9640 | 0.9977 | 0.9950 | 0.0061 | 0.9950 | 0.0012 | 5.3143 |
| 3 | 0.9562 | 0.9968 | 0.9935 | 0.0068 | 0.9958 | 0.0012 | n/a |
| 5 | 0.9343 | 0.9865 | 0.9804 | 0.0107 | 0.9962 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_2_friction_features_20260518_0035/heads/phase5_2_full_plus_r_20260518_0035/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_2_friction_features_20260518_0035/features/feature_manifest.json`
