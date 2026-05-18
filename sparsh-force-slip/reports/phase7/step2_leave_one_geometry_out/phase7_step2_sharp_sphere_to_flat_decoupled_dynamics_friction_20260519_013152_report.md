# Phase5 Head Report: decoupled_dynamics_friction

- generated_at: `2026-05-19T01:42:17`
- feature_run_id: `phase5_2_friction_features_20260518_0035`
- experiment_name: `phase7_step2_sharp_sphere_to_flat_decoupled_dynamics_friction_20260519_013152`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_q_pred']`
- train_count / val_count: `57857` / `1811`
- best_epoch: `37`
- selection_score_mean_auprc: `0.9911`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9596 | 0.9979 | 0.9966 | 0.0230 | 1.0000 | 0.0000 | 9.5789 |
| 3 | 0.9415 | 0.9953 | 0.9933 | 0.0269 | 1.0000 | n/a | n/a |
| 5 | 0.9183 | 0.9851 | 0.9834 | 0.0507 | 1.0000 | n/a | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0350`
- slip_f1: `0.9338`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_2_friction_features_20260518_0035/heads/phase7_step2_sharp_sphere_to_flat_decoupled_dynamics_friction_20260519_013152/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_2_friction_features_20260518_0035/features/feature_manifest.json`
