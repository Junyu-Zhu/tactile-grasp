# Phase5 Head Report: separate_late_fusion

- generated_at: `2026-05-19T01:33:56`
- feature_run_id: `phase4_2_separate_features_20260517_171500`
- experiment_name: `phase7_step2_sharp_sphere_to_flat_separate_late_fusion_20260519_013152`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current']`
- train_count / val_count: `57857` / `1811`
- best_epoch: `34`
- selection_score_mean_auprc: `0.9719`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9569 | 0.9868 | 0.9872 | 0.0181 | 1.0000 | 0.0085 | 4.2000 |
| 3 | 0.9296 | 0.9691 | 0.9725 | 0.0247 | 1.0000 | 0.0000 | n/a |
| 5 | 0.8904 | 0.9456 | 0.9560 | 0.0392 | 1.0000 | 0.0000 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0339`
- slip_f1: `0.9498`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_2_separate_features_20260517_171500/heads/phase7_step2_sharp_sphere_to_flat_separate_late_fusion_20260519_013152/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_2_separate_features_20260517_171500/features/feature_manifest.json`
