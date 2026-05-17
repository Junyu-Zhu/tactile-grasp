# Phase5 Head Report: separate_late_fusion

- generated_at: `2026-05-18T01:47:23`
- feature_run_id: `phase4_2_separate_features_20260517_171500`
- experiment_name: `phase6_2_separate_late_fusion_20260518_0146`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current']`
- train_count / val_count: `52081` / `2799`
- best_epoch: `37`
- selection_score_mean_auprc: `0.9458`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9556 | 0.9805 | 0.9765 | 0.0241 | 1.0000 | 0.0000 | 2.8571 |
| 3 | 0.9031 | 0.9499 | 0.9435 | 0.0352 | 1.0000 | 0.0078 | n/a |
| 5 | 0.8614 | 0.9210 | 0.9174 | 0.0399 | 1.0000 | 0.0083 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0429`
- slip_f1: `0.9701`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_2_separate_features_20260517_171500/heads/phase6_2_separate_late_fusion_20260518_0146/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_2_separate_features_20260517_171500/features/feature_manifest.json`
