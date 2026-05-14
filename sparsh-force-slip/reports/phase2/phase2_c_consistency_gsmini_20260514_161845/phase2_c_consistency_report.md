# Phase 2-C Consistency Decoder Report

- generated_at: `2026-05-14T20:12:13`
- phase2_c_run_id: `phase2_c_consistency_gsmini_20260514_161845`
- phase1_run_id: `phase1_gsmini_20260512_043331`
- derived_root: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- scope: Step3 C consistency decoder training/evaluation on train+val-defined hyperparameters.
- consistency loss: `BCE(p_slip, sigmoid(alpha * (Ft_pred/(Fn_pred+eps) - tau)).detach())`.
- test set tuning: `false` (train/val only).

## A/B/C validation comparison

| encoder | B run id | C checkpoint | C force RMSE mean N | C slip F1 | C slip recall | B→C gate | A→C gate | diagnostic-only |
|---|---|---|---:|---:|---:|---|---|---|
| dinov2 | `phase2_b_ps_dino_lam020_gsmini_20260514_160907` | `epoch-0040.pth` | 0.0480 | 0.9665 | 0.9747 | pass | warning_band | false |
| mae | `phase2_b_ps_lam050_gsmini_20260514_063052` | `epoch-0015.pth` | 0.0335 | 0.9819 | 0.9783 | warning_band | pass | false |

## Main metric gates

| encoder | A force | B force | C force | C Δ vs A | C Δ vs B | A slip F1 | B slip F1 | C slip F1 | C F1 drop vs A pp | C F1 drop vs B pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dinov2 | 0.0436 | 0.0476 | 0.0480 | +9.99% | +0.85% | 0.9477 | 0.9622 | 0.9665 | -1.87 | -0.43 |
| mae | 0.0322 | 0.0319 | 0.0335 | +3.99% | +5.02% | 0.9731 | 0.9799 | 0.9819 | -0.88 | -0.19 |

## Consistency diagnostics vs B

| encoder | contradiction B | contradiction C | reduction | monotonic B | monotonic C | reduction | high-recall drop pp | low-FA increase pp | C success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dinov2 | 0.0006 | 0.0004 | 38.76% | 0.0001 | 0.0000 | 100.00% | 0.00 | -0.01 | true |
| mae | 0.0004 | 0.0002 | 32.33% | 0.0001 | 0.0003 | -172.99% | 0.00 | -0.04 | true |

## C hyperparameters

| encoder | lambda_slip | beta_consistency | alpha | tau | tau_source | detach_q |
|---|---:|---:|---:|---:|---|---|
| dinov2 | 0.2000 | 0.0500 | 10.0000 | 0.081703 | `p65` | `True` |
| mae | 0.5000 | 0.0500 | 10.0000 | 0.081703 | `p65` | `True` |

## Conclusion

C satisfies the Step3/Step4 consistency success criteria for dinov2, mae.

## Artifacts

- JSON report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase2/phase2_c_consistency_gsmini_20260514_161845/phase2_c_consistency_report.json`
