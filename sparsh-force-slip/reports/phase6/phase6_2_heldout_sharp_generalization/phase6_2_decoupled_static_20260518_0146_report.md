# Phase5 Head Report: decoupled_static

- generated_at: `2026-05-18T01:48:44`
- feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- experiment_name: `phase6_2_decoupled_static_20260518_0146`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current']`
- train_count / val_count: `52081` / `2799`
- best_epoch: `37`
- selection_score_mean_auprc: `0.9490`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9597 | 0.9836 | 0.9784 | 0.0130 | 1.0000 | 0.0000 | 12.5000 |
| 3 | 0.8914 | 0.9564 | 0.9475 | 0.0382 | 1.0000 | 0.0056 | n/a |
| 5 | 0.8494 | 0.9292 | 0.9211 | 0.0592 | 1.0000 | 0.0118 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0372`
- slip_f1: `0.9606`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/heads/phase6_2_decoupled_static_20260518_0146/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500/features/feature_manifest.json`
