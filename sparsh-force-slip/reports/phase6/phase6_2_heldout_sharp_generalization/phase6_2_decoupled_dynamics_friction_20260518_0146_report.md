# Phase5 Head Report: decoupled_dynamics_friction

- generated_at: `2026-05-18T01:51:20`
- feature_run_id: `phase6_2_friction_features_20260518_0146`
- experiment_name: `phase6_2_decoupled_dynamics_friction_20260518_0146`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_q_pred']`
- train_count / val_count: `52081` / `2799`
- best_epoch: `38`
- selection_score_mean_auprc: `0.9829`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9644 | 0.9980 | 0.9951 | 0.0048 | 1.0000 | 0.0052 | 8.7000 |
| 3 | 0.9447 | 0.9926 | 0.9870 | 0.0183 | 1.0000 | 0.0223 | n/a |
| 5 | 0.8756 | 0.9748 | 0.9665 | 0.0619 | 1.0000 | 0.0888 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0372`
- slip_f1: `0.9606`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_2_friction_features_20260518_0146/heads/phase6_2_decoupled_dynamics_friction_20260518_0146/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_2_friction_features_20260518_0146/features/feature_manifest.json`
