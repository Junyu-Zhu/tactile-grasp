# Phase 2-B Shared Multitask Decoder Diagnostic Report

- generated_at: `2026-05-14T06:30:28`
- phase2_b_run_id: `phase2_b_ps_gsmini_20260514_035248`
- phase1_run_id: `phase1_gsmini_20260512_043331`
- derived_root: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- scope: Step1 B shared multitask decoder training + Step2 B diagnostics and C sweep plan.
- test set tuning: `false` (train/val only).

## Step1 B training outputs

| encoder | checkpoint | W&B/run dir | val force RMSE mean N | val slip F1 | val slip recall | gate vs A |
|---|---|---|---:|---:|---:|---|
| dinov2 | `epoch-0051.pth` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_ps_gsmini_20260514_035248/dinov2_partially_shared_multitask` | 0.0531 | 0.9630 | 0.9758 | hard_fail |
| mae | `epoch-0040.pth` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_ps_gsmini_20260514_035248/mae_partially_shared_multitask` | 0.0341 | 0.9753 | 0.9802 | warning_band |

## A vs B validation comparison

| encoder | force RMSE A | force RMSE B | Δ force RMSE | slip F1 A | slip F1 B | F1 drop pp | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| dinov2 | 0.0436 | 0.0531 | +21.77% | 0.9477 | 0.9630 | -1.53 | hard_fail |
| mae | 0.0322 | 0.0341 | +5.77% | 0.9731 | 0.9753 | -0.22 | warning_band |

Gate definitions: pass if B force RMSE increase <=5% and slip F1 drop <1pp; warning band if force RMSE increase is 5-10% or slip F1 drop is 1-2pp; hard fail if force RMSE increase >10% or slip F1 drop >2pp.

## Step2 tau/alpha/beta/slip_horizon decision

Tau candidates are computed from train+val ground-truth force labels only, using `Ft/(Fn+1e-6)` in Newton units.

| tau label | tau |
|---|---:|
| p50 | 0.050356 |
| p65 | 0.081703 |
| p80 | 0.710001 |

- alpha candidates: `[5, 10, 20]`
- beta_cons candidates: `[0.01, 0.05, 0.1]`
- sweep grid count: `27`
- slip_horizon fixed to `0` for Phase2-B/C train/val work.
- slip_horizon rationale: the official Sparsh downstream slip config and Phase1 alignment audit used horizon 0; keeping it fixed avoids using test data to tune temporal label semantics. If later changed, it must be swept on train/val only before any test evaluation.

## Consistency diagnostic snapshot

| encoder | model | contradiction rate | monotonic calib error | high-ratio recall | low-ratio false alarm |
|---|---|---:|---:|---:|---:|
| dinov2 | A | 0.0012 | 0.0000 | 1.0000 | 0.0020 |
| dinov2 | B | 0.0014 | 0.0009 | 1.0000 | 0.0026 |
| mae | A | 0.0013 | 0.0003 | 1.0000 | 0.0015 |
| mae | B | 0.0014 | 0.0003 | 1.0000 | 0.0030 |

## C entry note

B triggers a hard fail for dinov2; Phase2-C may only be run as diagnostic-only for those encoders and must not be used for a formal improvement claim.

## Artifacts

- JSON report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase2/phase2_b_ps_gsmini_20260514_035248/phase2_b_diagnostic_report.json`
- C sweep plan JSON: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase2/phase2_b_ps_gsmini_20260514_035248/phase2_c_sweep_plan.json`
