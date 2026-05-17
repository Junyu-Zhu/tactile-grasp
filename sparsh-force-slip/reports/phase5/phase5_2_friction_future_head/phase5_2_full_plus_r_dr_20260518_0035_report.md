# Phase5 Head Report: full_plus_r_dr

- generated_at: `2026-05-18T00:35:29`
- feature_run_id: `phase5_2_friction_features_20260518_0035`
- experiment_name: `phase5_2_full_plus_r_dr_20260518_0035`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_r_clipped_pred', 'friction_dr_clipped_pred']`
- train_count / val_count: `65938` / `14328`
- best_epoch: `40`
- selection_score_mean_auprc: `0.9899`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9652 | 0.9979 | 0.9953 | 0.0068 | 0.9969 | 0.0012 | 3.8842 |
| 3 | 0.9545 | 0.9969 | 0.9937 | 0.0108 | 0.9965 | 0.0004 | n/a |
| 5 | 0.9292 | 0.9870 | 0.9807 | 0.0188 | 0.9969 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_2_friction_features_20260518_0035/heads/phase5_2_full_plus_r_dr_20260518_0035/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_2_friction_features_20260518_0035/features/feature_manifest.json`
