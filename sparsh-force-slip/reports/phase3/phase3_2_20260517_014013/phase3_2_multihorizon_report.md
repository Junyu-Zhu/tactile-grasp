# Phase3-2 multihorizon Report

- generated_at: `2026-05-17T01:55:27`
- run_id: `phase3_2_world_model_mae_20260517_014013`
- encoder: `mae`
- source_decoupled_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- best_epoch: `32`
- target: future slip-free probability / instability risk (`not grasp success`).
- raw_data_modified: `False`

## Future prediction metrics

| horizon | future slip F1 | future slip AUROC | future slip AUPRC | future stability AUROC | future stability AUPRC | stability ECE | lead time mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9645 | 0.9977 | 0.9947 | 0.9977 | 0.9988 | 0.0053 | 5.2190 |
| 3 | 0.9506 | 0.9966 | 0.9928 | 0.9966 | 0.9982 | 0.0103 | n/a |
| 5 | 0.9222 | 0.9868 | 0.9800 | 0.9868 | 0.9924 | 0.0227 | n/a |

## Retained current force/slip metrics from frozen decoupled model

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`
- slip_accuracy: `0.9819`
- contradiction_rate: `0.0016`
- monotonic_calibration_error: `0.0002`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_2_20260517_014013/phase3_2_multihorizon_report.json`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase3/phase3_2_world_model_mae_20260517_014013/multihorizon/checkpoints/best.pth`
- feature_manifest: `/vla1/zjy/sparsh_runs/force_slip_phase3/phase3_2_world_model_mae_20260517_014013/features/feature_manifest.json`
