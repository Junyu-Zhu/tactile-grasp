# Phase4 Future Stability Head: phase_lambda_arch_future_consistency_decoder_full_20260528_000000

- generated_at: `2026-05-28T04:10:47`
- feature_run_id: `phase_lambda_arch_future_consistency_decoder_features_20260528_000000`
- feature_kind: `decoupled_joint`
- input_mode: `full`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- best_epoch: `40`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9662 | 0.9979 | 0.9954 | 0.9979 | 0.9988 | 0.0076 | 2.8370 |
| 3 | 0.9582 | 0.9973 | 0.9946 | 0.9973 | 0.9984 | 0.0115 | n/a |
| 5 | 0.9341 | 0.9876 | 0.9825 | 0.9876 | 0.9920 | 0.0188 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0319`
- slip_f1: `0.9787`
- slip_accuracy: `0.9908`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/phase_lambda_arch_future_consistency_decoder_full_20260528_000000_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_arch_future_consistency_decoder_features_20260528_000000/heads/phase_lambda_arch_future_consistency_decoder_full_20260528_000000/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_arch_future_consistency_decoder_features_20260528_000000/features/feature_manifest.json`
