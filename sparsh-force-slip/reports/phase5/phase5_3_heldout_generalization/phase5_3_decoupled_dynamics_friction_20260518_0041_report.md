# Phase5 Head Report: decoupled_dynamics_friction

- generated_at: `2026-05-18T00:45:52`
- feature_run_id: `phase5_3_friction_features_20260518_0041`
- experiment_name: `phase5_3_decoupled_dynamics_friction_20260518_0041`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_q_pred']`
- train_count / val_count: `21938` / `9718`
- best_epoch: `40`
- selection_score_mean_auprc: `0.9774`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9620 | 0.9980 | 0.9934 | 0.0054 | 0.9967 | 0.0013 | 1.5769 |
| 3 | 0.9400 | 0.9901 | 0.9814 | 0.0123 | 0.9980 | 0.0000 | n/a |
| 5 | 0.9169 | 0.9698 | 0.9573 | 0.0275 | 0.9993 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0229`
- slip_f1: `0.9672`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_3_friction_features_20260518_0041/heads/phase5_3_decoupled_dynamics_friction_20260518_0041/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_3_friction_features_20260518_0041/features/feature_manifest.json`
