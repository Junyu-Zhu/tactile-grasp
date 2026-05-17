# Phase5 Head Report: full_plus_q

- generated_at: `2026-05-18T01:43:04`
- feature_run_id: `phase6_1_friction_features_20260518_0130`
- experiment_name: `phase6_1_full_plus_q_seed44_20260518_0130`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_q_pred']`
- train_count / val_count: `65938` / `14328`
- best_epoch: `38`
- selection_score_mean_auprc: `0.9903`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9644 | 0.9984 | 0.9958 | 0.0084 | 0.9973 | 0.0004 | 4.1548 |
| 3 | 0.9542 | 0.9975 | 0.9942 | 0.0103 | 0.9985 | 0.0004 | n/a |
| 5 | 0.9312 | 0.9872 | 0.9810 | 0.0109 | 0.9988 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/heads/phase6_1_full_plus_q_seed44_20260518_0130/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/features/feature_manifest.json`
