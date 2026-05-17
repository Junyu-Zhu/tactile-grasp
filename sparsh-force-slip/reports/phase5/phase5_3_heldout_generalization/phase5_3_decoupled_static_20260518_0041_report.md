# Phase5 Head Report: decoupled_static

- generated_at: `2026-05-18T00:43:34`
- feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- experiment_name: `phase5_3_decoupled_static_20260518_0041`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current']`
- train_count / val_count: `21938` / `9718`
- best_epoch: `36`
- selection_score_mean_auprc: `0.9469`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9617 | 0.9861 | 0.9787 | 0.0049 | 1.0000 | 0.0021 | 3.7308 |
| 3 | 0.9185 | 0.9611 | 0.9456 | 0.0344 | 1.0000 | 0.0000 | n/a |
| 5 | 0.8636 | 0.9353 | 0.9164 | 0.0650 | 1.0000 | 0.0058 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0229`
- slip_f1: `0.9672`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/heads/phase5_3_decoupled_static_20260518_0041/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/features/feature_manifest.json`
