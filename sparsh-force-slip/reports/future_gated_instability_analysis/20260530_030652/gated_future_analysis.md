# Gated Future-instability Analysis

- Real data is used only for validation/analysis; no model weights are trained on real data.
- `mul_gate` computes `p_instability_H * p_slip_current`, suppressing future-risk scores when the current tactile observation is confidently stable.

## H1 overall ablation

| model | mode | stable mean | slip mean | gap | F1 | BalAcc | stable acc | slip recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_horizon_fix | raw | 0.915 | 0.987 | 0.072 | 0.581 | 0.500 | 0.000 | 1.000 |
| baseline_horizon_fix | mul_gate | 0.118 | 0.875 | 0.757 | 0.859 | 0.882 | 0.875 | 0.890 |
| baseline_horizon_fix | sqrt_gate | 0.177 | 0.890 | 0.712 | 0.842 | 0.868 | 0.838 | 0.898 |
| baseline_horizon_fix | avg_blend | 0.519 | 0.932 | 0.413 | 0.826 | 0.854 | 0.802 | 0.906 |
| medium_hard_negative | raw | 0.822 | 0.970 | 0.148 | 0.599 | 0.538 | 0.082 | 0.995 |
| medium_hard_negative | mul_gate | 0.114 | 0.867 | 0.753 | 0.861 | 0.884 | 0.878 | 0.890 |
| medium_hard_negative | sqrt_gate | 0.167 | 0.882 | 0.714 | 0.844 | 0.870 | 0.842 | 0.898 |
| medium_hard_negative | avg_blend | 0.472 | 0.923 | 0.451 | 0.833 | 0.860 | 0.815 | 0.906 |
| transition_redefined | raw | 0.770 | 0.933 | 0.164 | 0.603 | 0.553 | 0.135 | 0.971 |
| transition_redefined | mul_gate | 0.113 | 0.850 | 0.737 | 0.853 | 0.877 | 0.876 | 0.877 |
| transition_redefined | sqrt_gate | 0.162 | 0.863 | 0.701 | 0.840 | 0.866 | 0.847 | 0.885 |
| transition_redefined | avg_blend | 0.446 | 0.905 | 0.459 | 0.835 | 0.862 | 0.825 | 0.898 |

## H1 cucumber/hammer ablation

| model | mode | stable mean | slip mean | gap | F1 | BalAcc | stable acc | slip recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_horizon_fix | raw | 0.962 | 0.991 | 0.029 | 0.831 | 0.500 | 0.000 | 1.000 |
| baseline_horizon_fix | mul_gate | 0.281 | 0.928 | 0.647 | 0.907 | 0.809 | 0.681 | 0.937 |
| baseline_horizon_fix | sqrt_gate | 0.421 | 0.937 | 0.516 | 0.889 | 0.757 | 0.578 | 0.937 |
| baseline_horizon_fix | avg_blend | 0.623 | 0.960 | 0.338 | 0.875 | 0.709 | 0.474 | 0.944 |
| medium_hard_negative | raw | 0.846 | 0.979 | 0.133 | 0.844 | 0.547 | 0.095 | 1.000 |
| medium_hard_negative | mul_gate | 0.270 | 0.921 | 0.651 | 0.907 | 0.809 | 0.681 | 0.937 |
| medium_hard_negative | sqrt_gate | 0.394 | 0.929 | 0.535 | 0.891 | 0.766 | 0.595 | 0.937 |
| medium_hard_negative | avg_blend | 0.565 | 0.955 | 0.390 | 0.882 | 0.731 | 0.517 | 0.944 |
| transition_redefined | raw | 0.804 | 0.950 | 0.146 | 0.840 | 0.558 | 0.138 | 0.979 |
| transition_redefined | mul_gate | 0.274 | 0.906 | 0.632 | 0.901 | 0.804 | 0.681 | 0.926 |
| transition_redefined | sqrt_gate | 0.394 | 0.913 | 0.519 | 0.884 | 0.756 | 0.586 | 0.926 |
| transition_redefined | avg_blend | 0.544 | 0.940 | 0.396 | 0.878 | 0.727 | 0.517 | 0.937 |

## Decision

- status: `significant_gated_improvement`
- best: `medium_hard_negative` + `mul_gate`
- H1 stable drop vs raw baseline: `0.801`
- H1 F1 gain vs raw baseline: `0.281`

## Interpretation

- Raw future heads preserve ranking but overestimate real stable windows.
- Multiplicative pSlip gating makes the deployment score interpretable: stable windows become low risk while slip windows remain high risk.
- This should be reported as a gated deployment score / ablation, not as evidence that the standalone future head is fully calibrated.

## Files

- overall_metrics: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_gated_instability_analysis/20260530_030652/gated_future_metrics_overall.csv`
- focus_metrics: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_gated_instability_analysis/20260530_030652/gated_future_metrics_cucumber_hammer.csv`
- window_detail: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_gated_instability_analysis/20260530_030652/gated_future_window_detail.csv`
- plot: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_gated_instability_analysis/20260530_030652/h1_gating_ablation.png`
- plot: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_gated_instability_analysis/20260530_030652/medium_cucumber_hammer_h1_gated_windows.png`
