# Phase5 Head Report: full_dynamics_baseline

- generated_at: `2026-05-18T01:41:22`
- feature_run_id: `phase6_1_friction_features_20260518_0130`
- experiment_name: `phase6_1_full_dynamics_baseline_seed44_20260518_0130`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- train_count / val_count: `65938` / `14328`
- best_epoch: `40`
- selection_score_mean_auprc: `0.9899`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9634 | 0.9978 | 0.9952 | 0.0074 | 0.9954 | 0.0008 | 5.2151 |
| 3 | 0.9553 | 0.9968 | 0.9936 | 0.0081 | 0.9962 | 0.0004 | n/a |
| 5 | 0.9323 | 0.9869 | 0.9809 | 0.0112 | 0.9962 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/heads/phase6_1_full_dynamics_baseline_seed44_20260518_0130/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/features/feature_manifest.json`
