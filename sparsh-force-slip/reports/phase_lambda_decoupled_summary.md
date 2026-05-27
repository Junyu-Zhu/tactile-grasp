# Phase Lambda Decoupled Summary

- generated_at: `2026-05-28T04:15:33`
- Phase A report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled/20260528_000000/phase_lambda_decoupled_report.md`
- Phase B report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000/phase_lambda_decoupled_future_ablation_report.md`

## Phase A current force-slip λ sweep

| λ | F RMSE | ΔF | SF1 | ΔSF1 drop | SA | ΔSA drop | gate | checkpoint |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.10 | 0.0303 | -5.8780% | 0.9676 | 0.5277% | 0.9837 | 0.2718% | pass | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth` |
| 0.25 | 0.0312 | -3.0348% | 0.9653 | 0.7660% | 0.9825 | 0.3941% | pass | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam025_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0045.pth` |
| 0.50 | 0.0307 | -4.6381% | 0.9663 | 0.6575% | 0.9830 | 0.3397% | pass | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam050_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0045.pth` |
| 0.75 | 0.0307 | -4.4525% | 0.9686 | 0.4240% | 0.9842 | 0.2174% | pass | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam075_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth` |
| 1.00 | 0.0304 | -5.4178% | 0.9649 | 0.7982% | 0.9824 | 0.4077% | pass | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth` |

Selected best λ: `0.1` using non-hard-fail -> lowest Force RMSE -> higher SF1/SA.

## Phase B best-λ future ablation

| condition | H1 F1 | H1 AUPRC | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| z_only | 0.9450 | 0.9655 | 0.8890 | 0.9258 | 0.8401 | 0.8971 |
| z_p_slip | 0.9612 | 0.9838 | 0.9173 | 0.9645 | 0.8723 | 0.9404 |
| z_force | 0.9589 | 0.9763 | 0.9019 | 0.9455 | 0.8569 | 0.9200 |
| z_force_slip | 0.9631 | 0.9845 | 0.9157 | 0.9652 | 0.8713 | 0.9415 |
| full | 0.9654 | 0.9951 | 0.9567 | 0.9936 | 0.9351 | 0.9803 |
| full_plus_q | 0.9640 | 0.9954 | 0.9416 | 0.9938 | 0.9142 | 0.9813 |

Best future condition by mean AUPRC: `full_plus_q`.

## Paper-use recommendation

Replace the current λ=1.0 main result only if the best-λ row is not a hard fail in Phase A and improves or matches the old future-ablation metrics. If the improvement is mixed, keep λ sweep and best-λ future prediction as ablation evidence rather than changing Table2 main claims.
