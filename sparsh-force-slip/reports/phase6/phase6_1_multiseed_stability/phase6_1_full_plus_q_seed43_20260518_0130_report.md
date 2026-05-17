# Phase5 Head Report: full_plus_q

- generated_at: `2026-05-18T01:39:43`
- feature_run_id: `phase6_1_friction_features_20260518_0130`
- experiment_name: `phase6_1_full_plus_q_seed43_20260518_0130`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_q_pred']`
- train_count / val_count: `65938` / `14328`
- best_epoch: `36`
- selection_score_mean_auprc: `0.9904`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9646 | 0.9982 | 0.9956 | 0.0085 | 0.9973 | 0.0008 | 3.5556 |
| 3 | 0.9503 | 0.9973 | 0.9941 | 0.0162 | 0.9977 | 0.0000 | n/a |
| 5 | 0.9248 | 0.9875 | 0.9814 | 0.0270 | 0.9977 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/heads/phase6_1_full_plus_q_seed43_20260518_0130/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/features/feature_manifest.json`
