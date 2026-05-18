# Phase5 Head Report: decoupled_dynamics

- generated_at: `2026-05-19T01:39:31`
- feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- experiment_name: `phase7_step2_sharp_sphere_to_flat_decoupled_dynamics_20260519_013152`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- train_count / val_count: `57857` / `1811`
- best_epoch: `40`
- selection_score_mean_auprc: `0.9891`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9539 | 0.9957 | 0.9938 | 0.0244 | 0.9885 | 0.0000 | 11.0811 |
| 3 | 0.9285 | 0.9937 | 0.9916 | 0.0411 | 0.9924 | n/a | n/a |
| 5 | 0.9120 | 0.9836 | 0.9820 | 0.0539 | 0.9962 | n/a | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0350`
- slip_f1: `0.9338`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/heads/phase7_step2_sharp_sphere_to_flat_decoupled_dynamics_20260519_013152/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/features/feature_manifest.json`
