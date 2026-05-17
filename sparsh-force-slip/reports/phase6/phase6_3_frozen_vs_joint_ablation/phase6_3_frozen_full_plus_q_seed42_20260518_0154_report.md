# Phase5 Head Report: frozen_two_stage_full_plus_q

- generated_at: `2026-05-18T01:56:31`
- feature_run_id: `phase6_3_friction_features_20260518_0154`
- experiment_name: `phase6_3_frozen_full_plus_q_seed42_20260518_0154`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_q_pred']`
- train_count / val_count: `65938` / `14328`
- best_epoch: `40`
- selection_score_mean_auprc: `0.9906`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9668 | 0.9983 | 0.9959 | 0.0053 | 0.9981 | 0.0016 | 5.6019 |
| 3 | 0.9593 | 0.9975 | 0.9945 | 0.0054 | 0.9981 | 0.0008 | n/a |
| 5 | 0.9381 | 0.9872 | 0.9815 | 0.0078 | 0.9985 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_3_friction_features_20260518_0154/heads/phase6_3_frozen_full_plus_q_seed42_20260518_0154/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_3_friction_features_20260518_0154/features/feature_manifest.json`
