# Phase5 Head Report: full_plus_q

- generated_at: `2026-05-28T04:14:59`
- feature_run_id: `phase_lambda_best_friction_features_20260528_000000`
- experiment_name: `phase_lambda_best_full_plus_q_20260528_000000`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N', 'friction_q_pred']`
- train_count / val_count: `65938` / `14328`
- best_epoch: `38`
- selection_score_mean_auprc: `0.9902`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9640 | 0.9981 | 0.9954 | 0.0127 | 0.9973 | 0.0000 | 2.0455 |
| 3 | 0.9416 | 0.9972 | 0.9938 | 0.0238 | 0.9977 | 0.0012 | n/a |
| 5 | 0.9142 | 0.9875 | 0.9813 | 0.0389 | 0.9977 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0283`
- slip_f1: `0.9627`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase_lambda_best_friction_features_20260528_000000/heads/phase_lambda_best_full_plus_q_20260528_000000/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase5/phase_lambda_best_friction_features_20260528_000000/features/feature_manifest.json`
