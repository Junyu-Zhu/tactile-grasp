# Phase4 Future Stability Head: phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000

- generated_at: `2026-05-28T04:10:47`
- feature_run_id: `phase_lambda_arch_future_partially_shared_lam025_features_20260528_000000`
- feature_kind: `decoupled_joint`
- input_mode: `full`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current', 'dFx_causal_N', 'dFy_causal_N', 'dFz_causal_N']`
- best_epoch: `40`
- raw_data_modified: `False`

| H | future slip F1 | future slip AUROC | future slip AUPRC | stability AUROC | stability AUPRC | ECE | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9617 | 0.9971 | 0.9944 | 0.9971 | 0.9980 | 0.0100 | 4.0000 |
| 3 | 0.9535 | 0.9965 | 0.9933 | 0.9965 | 0.9973 | 0.0156 | n/a |
| 5 | 0.9274 | 0.9867 | 0.9813 | 0.9867 | 0.9909 | 0.0251 | n/a |

## Current-task metrics from feature source

- force_rmse_mean_N: `0.0294`
- slip_f1: `0.9735`
- slip_accuracy: `0.9885`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_arch_future_partially_shared_lam025_features_20260528_000000/heads/phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_arch_future_partially_shared_lam025_features_20260528_000000/features/feature_manifest.json`
