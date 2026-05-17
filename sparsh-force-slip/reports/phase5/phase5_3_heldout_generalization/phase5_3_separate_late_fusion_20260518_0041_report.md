# Phase5 Head Report: separate_late_fusion

- generated_at: `2026-05-18T00:42:27`
- feature_run_id: `phase4_2_separate_features_20260517_171500`
- experiment_name: `phase5_3_separate_late_fusion_20260518_0041`
- input_schema: `['z_t_mean_pooled', 'Fn_pred_N', 'Ft_pred_N', 'Ft_over_Fn_pred', 'p_slip_current']`
- train_count / val_count: `21938` / `9718`
- best_epoch: `39`
- selection_score_mean_auprc: `0.9457`
- raw_data_modified: `False`

| horizon | F1 | AUROC | AUPRC | ECE | high-risk recall | low-risk false alarm | lead mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9623 | 0.9865 | 0.9791 | 0.0071 | 0.9900 | 0.0000 | 6.2955 |
| 3 | 0.9089 | 0.9594 | 0.9436 | 0.0376 | 0.9935 | 0.0076 | n/a |
| 5 | 0.8525 | 0.9335 | 0.9145 | 0.0586 | 0.9936 | 0.0292 | n/a |

## Current-task subset metrics

- force_rmse_mean_N: `0.0232`
- slip_f1: `0.9739`

## Artifacts

- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_2_separate_features_20260517_171500/heads/phase5_3_separate_late_fusion_20260518_0041/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_2_separate_features_20260517_171500/features/feature_manifest.json`
