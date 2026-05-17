# Phase5 Head Report: decoupled_dynamics

- generated_at: `2026-05-18T01:50:02`
- feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- experiment_name: `phase6_2_decoupled_dynamics_20260518_0146`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- train_count / val_count: `52081` / `2799`
- best_epoch: `40`
- selection_score_mean_auprc: `0.9825`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9659 | 0.9978 | 0.9948 | 0.0066 | 1.0000 | 0.0000 | 11.6667 |
| 3 | 0.9397 | 0.9924 | 0.9866 | 0.0222 | 1.0000 | 0.0223 | n/a |
| 5 | 0.8675 | 0.9740 | 0.9661 | 0.0748 | 1.0000 | 0.1006 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0372`
- slip_f1: `0.9606`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/heads/phase6_2_decoupled_dynamics_20260518_0146/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/features/feature_manifest.json`
