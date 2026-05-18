# Phase5 Head Report: decoupled_static

- generated_at: `2026-05-19T01:36:35`
- feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- experiment_name: `phase7_step2_sharp_sphere_to_flat_decoupled_static_20260519_013152`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current']`
- train_count / val_count: `57857` / `1811`
- best_epoch: `34`
- selection_score_mean_auprc: `0.9713`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9582 | 0.9887 | 0.9869 | 0.0150 | 1.0000 | 0.0000 | 8.4444 |
| 3 | 0.9160 | 0.9728 | 0.9715 | 0.0431 | 1.0000 | n/a | n/a |
| 5 | 0.8659 | 0.9522 | 0.9556 | 0.0747 | 1.0000 | n/a | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0350`
- slip_f1: `0.9338`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/heads/phase7_step2_sharp_sphere_to_flat_decoupled_static_20260519_013152/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/features/feature_manifest.json`
