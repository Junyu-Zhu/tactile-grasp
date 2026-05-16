# Phase3 Lightweight Tactile World Model Summary

- generated_at: `2026-05-17T01:55:27`
- run_id: `phase3_2_world_model_mae_20260517_014013`
- best_backbone: `mae`
- source_phase3_1_report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/phase3_1_decoupled_multitask_report.json`
- target wording: future slip-free probability / instability risk, not grasp success.

## Stage comparison

| stage | horizons | best epoch | H1 future slip F1 | H1 future slip AUROC | H1 future slip AUPRC | H1 stability ECE | latent MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| proxy | [1] | 35 | 0.9594 | 0.9985 | 0.9957 | 0.0326 | n/a |
| latent | [1] | 35 | 0.9625 | 0.9981 | 0.9951 | 0.0085 | 0.002233 |
| multihorizon | [1, 3, 5] | 32 | 0.9645 | 0.9977 | 0.9947 | 0.0053 | n/a |

## Frozen decoupled current-task metrics retained during P3-2

- force_rmse_mean_N: `0.0284`
- slip_f1: `0.9593`
- slip_accuracy: `0.9819`
- contradiction_rate: `0.0016`
- monotonic_calibration_error: `0.0002`

## Artifacts

- JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_2_20260517_014013/phase3_world_model_summary.json`
