# Phase4-1 MAE Multi-seed Report

- generated_at: `2026-05-17T20:19:25`
- stamp: `20260517_144424`
- seeds: seed0=42 reused from Phase3, seed1=43, seed2=44
- raw_data_modified: `False`
- split: Phase1 derived all-source train/val split

## Per-seed metrics

| seed | separate force RMSE | separate slip F1 | separate slip acc | decoupled force RMSE | decoupled slip F1 | world H1 F1 | world H1 AUROC | world H1 AUPRC | world H1 ECE | H3 F1 | H5 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed0 | 0.0322 | 0.9731 | 0.9866 | 0.0304 | 0.9652 | 0.9645 | 0.9977 | 0.9947 | 0.0053 | 0.9506 | 0.9222 |
| seed1 | 0.0305 | 0.9738 | 0.9870 | 0.0320 | 0.9761 | 0.9670 | 0.9978 | 0.9954 | 0.0038 | 0.9549 | 0.9349 |
| seed2 | 0.0317 | 0.9678 | 0.9838 | 0.0327 | 0.9775 | 0.9651 | 0.9987 | 0.9962 | 0.0061 | 0.9589 | 0.9363 |

## Mean ± std

| metric | mean | std | n |
|---|---:|---:|---:|
| separate_force_rmse_mean_N | 0.0315 | 0.0009 | 3 |
| separate_slip_f1 | 0.9716 | 0.0033 | 3 |
| separate_slip_accuracy | 0.9858 | 0.0017 | 3 |
| decoupled_force_rmse_mean_N | 0.0317 | 0.0012 | 3 |
| decoupled_slip_f1 | 0.9729 | 0.0067 | 3 |
| decoupled_slip_accuracy | 0.9865 | 0.0035 | 3 |
| world_current_force_rmse_mean_N | 0.0298 | 0.0012 | 3 |
| world_current_slip_f1 | 0.9685 | 0.0080 | 3 |
| world_current_slip_accuracy | 0.9862 | 0.0037 | 3 |
| world_H1_future_slip_f1 | 0.9655 | 0.0013 | 3 |
| world_H1_future_slip_auroc | 0.9981 | 0.0006 | 3 |
| world_H1_future_slip_auprc | 0.9954 | 0.0007 | 3 |
| world_H1_future_stability_auroc | 0.9981 | 0.0006 | 3 |
| world_H1_future_stability_auprc | 0.9990 | 0.0005 | 3 |
| world_H1_stability_ece | 0.0051 | 0.0012 | 3 |
| world_H3_future_slip_f1 | 0.9548 | 0.0041 | 3 |
| world_H3_future_slip_auroc | 0.9969 | 0.0003 | 3 |
| world_H3_future_slip_auprc | 0.9934 | 0.0006 | 3 |
| world_H5_future_slip_f1 | 0.9311 | 0.0078 | 3 |
| world_H5_future_slip_auroc | 0.9863 | 0.0013 | 3 |
| world_H5_future_slip_auprc | 0.9801 | 0.0012 | 3 |

## Interpretation

- This phase checks whether the MAE paper-facing route is stable across three seeds rather than being a single-run artifact.
- Separate force/slip remains the SPARSH-style current-task baseline; decoupled multitask is judged by force protection and its ability to feed the future-stability head.
- The world-model row uses only derived feature caches and predicts future slip/instability, not grasp success.

## Artifacts

- seed0 force_exp: `/vla1/zjy/sparsh_runs/experiments/2026.05.12_04-39_phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652`
- seed0 slip_exp: `/vla1/zjy/sparsh_runs/experiments/2026.05.13_01-21_phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000`
- seed0 decoupled_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- seed0 world_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase3/phase3_2_world_model_mae_20260517_014013/multihorizon/checkpoints/best.pth`
- seed1 force_exp: `/vla1/zjy/sparsh_runs/experiments/2026.05.17_14-44_phase4_1_mae_force_seed1_20260517_144424`
- seed1 slip_exp: `/vla1/zjy/sparsh_runs/experiments/2026.05.17_14-44_phase4_1_mae_slip_seed1_20260517_144424`
- seed1 decoupled_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase4_1_mae_decoupled_seed1_20260517_144424/mae_decoupled_multitask/checkpoints/epoch-0040.pth`
- seed1 world_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_1_world_features_seed1_20260517_144424/heads/phase4_1_world_seed1_20260517_144424/checkpoints/best.pth`
- seed2 force_exp: `/vla1/zjy/sparsh_runs/experiments/2026.05.17_14-44_phase4_1_mae_force_seed2_20260517_144424`
- seed2 slip_exp: `/vla1/zjy/sparsh_runs/experiments/2026.05.17_16-35_phase4_1_mae_slip_seed2_20260517_144424`
- seed2 decoupled_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase4_1_mae_decoupled_seed2_20260517_144424/mae_decoupled_multitask/checkpoints/epoch-0025.pth`
- seed2 world_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_1_world_features_seed2_20260517_144424/heads/phase4_1_world_seed2_20260517_144424/checkpoints/best.pth`
