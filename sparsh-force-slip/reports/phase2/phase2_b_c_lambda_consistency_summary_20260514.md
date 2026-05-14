# Phase2 B DINOv2 lambda probe + C consistency summary

- generated_at: `2026-05-14T20:15:00`
- C report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase2/phase2_c_consistency_gsmini_20260514_161845/phase2_c_consistency_report.md`

## DINOv2 B lambda probe

| lambda_slip | run_id | selected epoch | force RMSE | Δ force vs A | slip F1 | F1 drop pp vs A | gate | note | W&B |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 0.25 | `phase2_b_ps_lam025_gsmini_20260514_063052` | 51 | 0.047934 | +9.91% | 0.971274 | -2.35 | warning_band | previous warning-band slip-F1 best | [wandb](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_b_ps_lam025_gsmini_20260514_063052_dinov2_b_partially_shared_lambda0.25) |
| 0.20 | `phase2_b_ps_dino_lam020_gsmini_20260514_160907` | 50 | 0.047565 | +9.06% | 0.962206 | -1.45 | warning_band | new force-safer warning-band baseline used for C | [wandb](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_b_ps_dino_lam020_gsmini_20260514_160907_dinov2_b_partially_shared_lambda0.20) |
| 0.15 | `phase2_b_ps_dino_lam015_gsmini_20260514_160907` | 51 | 0.047984 | +10.02% | 0.954165 | -0.64 | hard_fail | new lower-lambda probe | [wandb](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_b_ps_dino_lam015_gsmini_20260514_160907_dinov2_b_partially_shared_lambda0.15) |

Selection note: `lambda_slip=0.20` is the DINOv2 B baseline used for C because it stays in warning band with lower force RMSE increase than `0.25`; `0.25` remains the slip-F1-best warning-band DINOv2 B checkpoint. `0.15` hard-fails by force gate.

## Phase2-C consistency

| encoder | lambda_slip | beta | tau source | tau | force RMSE | slip F1 | gate vs A | gate vs B | contradiction reduction vs B | MCE reduction vs B | C success | W&B |
|---|---:|---:|---|---:|---:|---:|---|---|---:|---:|---|---|
| dinov2 | 0.20 | 0.05 | p65 | 0.081703 | 0.047968 | 0.966474 | warning_band | pass | +38.76% | +100.00% | true | [wandb](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_c_consistency_gsmini_20260514_161845_dinov2_c_consistency_lam0.20_beta0.05) |
| mae | 0.50 | 0.05 | p65 | 0.081703 | 0.033529 | 0.981873 | pass | warning_band | +32.33% | -172.99% | true | [wandb](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase2_c_consistency_gsmini_20260514_161845_mae_c_consistency_lam0.50_beta0.05) |

C satisfies the Step3/Step4 consistency success criteria for dinov2, mae.
