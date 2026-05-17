# Phase5-1 Friction-aware Diagnostic

- generated_at: `2026-05-18T00:26:42`
- source_feature_run_id: `phase4_3_reuse_p3_features_20260517_171500`
- raw_data_modified: `False`

## Thresholds frozen from train split

### gt

- valid contact: `Fn > train Fn p10 for the same force source` = `0.171570 N`
- Ft/Fn p20/p50/p80/p99: `0.0290` / `0.0501` / `0.4943` / `1.1249`
- q = sigmoid(alpha * (clipped(Ft/Fn)-tau)), tau=`0.4943`, alpha=`17.1910`

### pred

- valid contact: `Fn > train Fn p10 for the same force source` = `0.167060 N`
- Ft/Fn p20/p50/p80/p99: `0.0325` / `0.0529` / `0.5105` / `1.1095`
- q = sigmoid(alpha * (clipped(Ft/Fn)-tau)), tau=`0.5105`, alpha=`16.7350`

## Diagnostic table

| source | group | valid n | slip+ valid | high ratio n | high slip rate | high recall(dataset) | high model recall | low false alarm | p_slip~ratio Spearman | contradiction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gt | overall | 12733 | 0.2237 | 2633 | 1.0000 | 0.9245 | 0.9951 | 0.0033 | 0.6855 | 0.0017 |
| gt | flat | 1801 | 0.3232 | 530 | 1.0000 | 0.9107 | 0.9830 | n/a | 0.7944 | 0.0050 |
| gt | sharp | 2543 | 0.2363 | 562 | 1.0000 | 0.9351 | 0.9964 | 0.0236 | 0.5980 | 0.0031 |
| gt | sphere | 8389 | 0.1985 | 1541 | 1.0000 | 0.9255 | 0.9987 | 0.0012 | 0.6282 | 0.0006 |
| pred | overall | 12835 | 0.2226 | 2604 | 1.0000 | 0.9114 | 1.0000 | 0.0050 | 0.6760 | 0.0011 |
| pred | flat | 1800 | 0.3222 | 523 | 1.0000 | 0.9017 | 1.0000 | 0.5000 | 0.7500 | 0.0006 |
| pred | sharp | 2561 | 0.2339 | 546 | 1.0000 | 0.9115 | 1.0000 | 0.0152 | 0.5929 | 0.0012 |
| pred | sphere | 8474 | 0.1980 | 1535 | 1.0000 | 0.9148 | 1.0000 | 0.0038 | 0.6141 | 0.0012 |

## Interpretation

Predicted-force Ft/Fn has Spearman correlation 0.6760 with current p_slip; ground-truth-force Ft/Fn has Spearman correlation 0.6855. The proxy is therefore treated as a physical diagnostic/conditioning cue rather than a hard friction law.

## Artifacts

- figure: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase5/phase5_1_friction_diagnostic/figures/phase5_1_ratio_hist_gt.png`
- figure: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase5/phase5_1_friction_diagnostic/figures/phase5_1_p_slip_vs_ratio_gt.png`
- figure: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase5/phase5_1_friction_diagnostic/figures/phase5_1_ratio_hist_pred.png`
- figure: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase5/phase5_1_friction_diagnostic/figures/phase5_1_p_slip_vs_ratio_pred.png`
- CSV: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase5/phase5_1_friction_diagnostic/phase5_1_friction_diagnostic_rows.csv`
