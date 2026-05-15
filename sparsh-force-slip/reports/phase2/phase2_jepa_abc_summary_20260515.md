# Phase2 IJEPA/VJEPA ABC summary

- generated_at: `2026-05-15T20:01:38`
- updated_at: `2026-05-15T20:04:08`
- derived dataset: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- A run: `phase2_jepa_a_gsmini_20260515_014447`
- B launch: `phase2_b_jepa_ps_gsmini_20260515_075611`
- C status: **diagnostic-only, no consistency training launched**
- push/pull: not performed (manual per user rule)

## A/B/C decision table
| encoder | A force RMSE | A slip F1 | best B lambda | best B gate | best B force RMSE | best B slip F1 | C gate vs A | C gate vs B | C force RMSE | C slip F1 | C contradiction | C monotonic err | paper recommendation |
|---|---:|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---|
| ijepa | 0.064935 | 0.902162 | 0.50 | hard_fail | 0.074211 | 0.957392 | not_run_b_hard_fail | not_run_b_hard_fail | NA | NA | NA | NA | do_not_include_as_main_result; use as diagnostic/supplementary negative result only |
| vjepa | 0.046089 | 0.967087 | 0.25 | hard_fail | 0.050712 | 0.966653 | not_run_b_hard_fail | not_run_b_hard_fail | NA | NA | NA | NA | do_not_include_as_main_result; use as diagnostic/supplementary negative result only |

## Full B lambda sweep
| encoder | lambda | gate | force RMSE | force Δ% vs A | slip F1 | slip Δ pp vs A | contradiction | monotonic err | W&B | checkpoint |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| ijepa | 0.25 | hard_fail | 0.073105 | 12.58 | 0.952601 | -5.04 | 0.0037207266360253883 | 0.0021287462674081325 | https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_b_jepa_lam025_gsmini_20260515_075611_ijepa_partially_shared_lambda025 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0025.pth` |
| ijepa | 0.50 | hard_fail | 0.074211 | 14.29 | 0.957392 | -5.52 | 0.0018209408194233688 | 0.0002775545231997967 | https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_b_jepa_lam050_gsmini_20260515_075611_ijepa_partially_shared_lambda050 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0045.pth` |
| vjepa | 0.25 | hard_fail | 0.050712 | 10.03 | 0.966653 | 0.04 | 0.0010351966873706005 | 0.0003816230455413461 | https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_b_jepa_lam025_gsmini_20260515_075611_vjepa_partially_shared_lambda025 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0030.pth` |
| vjepa | 0.50 | hard_fail | 0.052817 | 14.60 | 0.949516 | 1.76 | 0.0014331780723754925 | 0.0006332227378152311 | https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_b_jepa_lam050_gsmini_20260515_075611_vjepa_partially_shared_lambda050 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |

## C diagnostic metrics/gates
C consistency training was not launched, therefore C force RMSE, slip F1, contradiction rate, and monotonic calibration error are `NA`. The C gate fields are `not_run_b_hard_fail` because every selected B checkpoint hard-failed the force RMSE gate.

## B tmux / GPU allocation
| session | encoder | lambda | GPU | W&B name | run dir | checkpoint |
|---|---|---:|---:|---|---|---|
| `p2b_jepa_20260515_075611_ijepa_l025_g0` | ijepa | 0.25 | 0 | `phase2_b_jepa_lam025_gsmini_20260515_075611_ijepa_partially_shared_lambda025` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/ijepa_partially_shared_multitask` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |
| `p2b_jepa_20260515_075611_ijepa_l050_g1` | ijepa | 0.50 | 1 | `phase2_b_jepa_lam050_gsmini_20260515_075611_ijepa_partially_shared_lambda050` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/ijepa_partially_shared_multitask` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |
| `p2b_jepa_20260515_075611_vjepa_l025_g2` | vjepa | 0.25 | 2 | `phase2_b_jepa_lam025_gsmini_20260515_075611_vjepa_partially_shared_lambda025` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/vjepa_partially_shared_multitask` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |
| `p2b_jepa_20260515_075611_vjepa_l050_g3` | vjepa | 0.50 | 3 | `phase2_b_jepa_lam050_gsmini_20260515_075611_vjepa_partially_shared_lambda050` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/vjepa_partially_shared_multitask` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |

## A checkpoints and W&B
### ijepa
- A force checkpoint: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_force/checkpoints/epoch-0051.pth`
- A slip checkpoint: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_slip/checkpoints/epoch-0051.pth`
- A force W&B: https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_force_0
- A slip W&B: https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_slip_0

### vjepa
- A force checkpoint: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_force/checkpoints/epoch-0051.pth`
- A slip checkpoint: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_slip/checkpoints/epoch-0051.pth`
- A force W&B: https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_force_0
- A slip W&B: https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_slip_0

## MAE/DINOv2 comparison
| encoder | previous C force RMSE | previous C slip F1 | gate vs A | gate vs B | success | W&B |
|---|---:|---:|---|---|---|---|
| dinov2 | 0.047968 | 0.966474 | warning_band | pass | True | https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_c_consistency_gsmini_20260514_161845_dinov2_c_consistency_lam0.20_beta0.05 |
| mae | 0.033529 | 0.981873 | pass | warning_band | True | https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_c_consistency_gsmini_20260514_161845_mae_c_consistency_lam0.50_beta0.05 |

## Recommendation
MAE/DINOv2 had usable Phase2-C consistency results (MAE pass, DINOv2 warning-band success). IJEPA/VJEPA B hard-failed on force RMSE, so JEPA should not be promoted to main paper result without further tuning/checkpoint selection.
