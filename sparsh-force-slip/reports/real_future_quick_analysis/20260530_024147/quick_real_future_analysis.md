# Quick Real Future-instability Analysis

- generated_at: `2026-05-30T02:41:49`
- source_report: `sparsh-force-slip/reports/real_force_slip_future_labels_eval/20260530_023127`
- output_dir: `sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147`
- real-data policy: this is **analysis only**. The real banana/cucumber/hammer/lemon dataset is not used for model training, checkpoint selection, or final threshold selection.
- threshold sweep on real data is marked diagnostic; original-validation thresholds are reported separately.

## Score separation summary

| subset | score | n | stable mean | slip mean | gap | AUROC | AUPRC | F1@0.5 | BalAcc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | p_slip_current | 931 | 0.122 | 0.877 | 0.755 | 0.937 | 0.948 | 0.857 | 0.880 |
| overall | p_instability_H1 | 931 | 0.915 | 0.987 | 0.072 | 0.923 | 0.921 | 0.581 | 0.500 |
| overall | p_instability_H3 | 931 | 0.944 | 0.991 | 0.048 | 0.904 | 0.894 | 0.581 | 0.500 |
| overall | p_instability_H5 | 931 | 0.956 | 0.993 | 0.037 | 0.904 | 0.895 | 0.581 | 0.500 |
| cucumber_hammer | p_slip_current | 401 | 0.283 | 0.930 | 0.647 | 0.947 | 0.984 | 0.907 | 0.809 |
| cucumber_hammer | p_instability_H1 | 401 | 0.962 | 0.991 | 0.029 | 0.867 | 0.950 | 0.831 | 0.500 |
| cucumber_hammer | p_instability_H3 | 401 | 0.977 | 0.994 | 0.016 | 0.798 | 0.917 | 0.831 | 0.500 |
| cucumber_hammer | p_instability_H5 | 401 | 0.984 | 0.995 | 0.011 | 0.801 | 0.919 | 0.831 | 0.500 |

## Original-validation thresholds applied to real data

| subset | score | threshold | stable acc | slip recall | balanced acc | F1 | FP | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | p_slip_current | 0.500 | 0.871 | 0.890 | 0.880 | 0.857 | 71 | 479 |
| overall | p_instability_H1 | 0.848 | 0.198 | 0.974 | 0.586 | 0.622 | 441 | 109 |
| overall | p_instability_H3 | 0.728 | 0.009 | 1.000 | 0.505 | 0.583 | 545 | 5 |
| overall | p_instability_H5 | 0.784 | 0.009 | 1.000 | 0.505 | 0.583 | 545 | 5 |
| cucumber_hammer | p_slip_current | 0.500 | 0.681 | 0.937 | 0.809 | 0.907 | 37 | 79 |
| cucumber_hammer | p_instability_H1 | 0.848 | 0.000 | 0.979 | 0.489 | 0.821 | 116 | 0 |
| cucumber_hammer | p_instability_H3 | 0.728 | 0.000 | 1.000 | 0.500 | 0.831 | 116 | 0 |
| cucumber_hammer | p_instability_H5 | 0.784 | 0.000 | 1.000 | 0.500 | 0.831 | 116 | 0 |

## Diagnostic best thresholds on real data (not for model selection)

| subset | score | best threshold | balanced acc | F1 | stable acc | slip recall |
|---|---|---:|---:|---:|---:|---:|
| overall | p_instability_H1 | 0.980 | 0.890 | 0.869 | 0.900 | 0.879 |
| overall | p_instability_H3 | 0.990 | 0.879 | 0.857 | 0.902 | 0.856 |
| overall | p_instability_H5 | 0.990 | 0.883 | 0.862 | 0.893 | 0.874 |
| cucumber_hammer | p_instability_H1 | 0.970 | 0.737 | 0.883 | 0.534 | 0.940 |
| cucumber_hammer | p_instability_H3 | 0.980 | 0.737 | 0.883 | 0.534 | 0.940 |
| cucumber_hammer | p_instability_H5 | 0.990 | 0.732 | 0.877 | 0.534 | 0.930 |

## Cucumber/Hammer window means

| sequence | side | window | range | n | pSlip | H1 | H3 | H5 |
|---|---|---|---|---:|---:|---:|---:|---:|
| cucumber_1 | left | slip | 35-75 | 41 | 0.838 | 0.975 | 0.982 | 0.986 |
| cucumber_1 | right | slip | 48-73 | 26 | 0.984 | 0.998 | 0.998 | 0.998 |
| cucumber_2 | left | slip | 39-52 | 14 | 0.862 | 0.956 | 0.963 | 0.970 |
| cucumber_2 | right | slip | 38-51 | 14 | 0.996 | 0.995 | 0.995 | 0.996 |
| cucumber_3 | left | slip | 40-62 | 23 | 0.591 | 0.978 | 0.983 | 0.986 |
| cucumber_3 | left | stable | 80-110 | 31 | 0.025 | 0.939 | 0.961 | 0.969 |
| cucumber_3 | right | slip | 41-64 | 24 | 0.936 | 0.995 | 0.997 | 0.998 |
| cucumber_3 | right | stable | 80-110 | 31 | 0.039 | 0.925 | 0.956 | 0.972 |
| hammer_1 | left | slip | 50-80 | 31 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer_1 | right | slip | 50-80 | 31 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer_2 | left | slip | 34-44 | 11 | 0.995 | 0.999 | 1.000 | 1.000 |
| hammer_2 | left | stable | 90-116 | 27 | 0.723 | 0.998 | 0.999 | 0.999 |
| hammer_2 | right | slip | 34-41 | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer_2 | right | stable | 90-116 | 27 | 0.419 | 0.996 | 0.998 | 0.999 |
| hammer_3 | left | slip | 40-70 | 31 | 1.000 | 1.000 | 1.000 | 1.000 |
| hammer_3 | right | slip | 40-70 | 31 | 1.000 | 1.000 | 1.000 | 1.000 |

## Main observations

- `p_slip_current` remains the most deployment-ready signal: it has a large stable/slip score gap and good F1 at the default 0.5 threshold.
- The future-instability head has useful ranking separation, but H1/H3/H5 raw probabilities remain shifted upward on real images; this is why stable windows still look unstable at threshold 0.5.
- Cucumber and hammer slip windows are clearer/longer, so they are better for qualitative figures. However hammer stable windows still produce high future-instability scores, highlighting the remaining domain-calibration gap.
- For paper use without training on real data, report AUROC/AUPRC/gap and original-validation threshold results, not a threshold tuned on real validation labels.

## Generated files

- threshold_sweep_csv: `sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147/threshold_sweep_metrics.csv`
- object_window_means_csv: `sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147/object_window_means.csv`
- sequence_window_means_csv: `sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147/sequence_window_means.csv`
- cucumber_hammer_window_means_csv: `sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147/cucumber_hammer_window_means.csv`
- markdown: `sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147/quick_real_future_analysis.md`
- plots: `['sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147/score_distributions_stable_vs_slip.png', 'sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147/future_threshold_sweep_balanced_accuracy.png', 'sparsh-force-slip/reports/real_future_quick_analysis/20260530_024147/cucumber_hammer_window_scores.png']`
