# Phase5 Head Report: decoupled_dynamics

- generated_at: `2026-05-18T00:44:45`
- feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- experiment_name: `phase5_3_decoupled_dynamics_20260518_0041`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- train_count / val_count: `21938` / `9718`
- best_epoch: `40`
- selection_score_mean_auprc: `0.9747`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9552 | 0.9973 | 0.9921 | 0.0223 | 0.9980 | 0.0046 | 6.3521 |
| 3 | 0.8922 | 0.9876 | 0.9778 | 0.1289 | 1.0000 | 0.0157 | n/a |
| 5 | 0.7392 | 0.9675 | 0.9543 | 0.2118 | 1.0000 | 0.0893 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0229`
- slip_f1: `0.9672`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/heads/phase5_3_decoupled_dynamics_20260518_0041/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/features/feature_manifest.json`
