# Quick Real Future-instability Analysis

- generated_at: `2026-05-30T03:03:55`
- source_report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/real_zero_shot_eval/20260530_030306`
- output_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354`
- real-data policy: this is **analysis only**. The real banana/cucumber/hammer/lemon dataset is not used for model training, checkpoint selection, or final threshold selection.
- threshold sweep on real data is marked diagnostic; original-validation thresholds are reported separately.

## Score separation summary

| subset | score | n | stable mean | slip mean | gap | AUROC | AUPRC | F1@0.5 | BalAcc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | p_slip_current | 931 | 0.122 | 0.877 | 0.755 | 0.937 | 0.948 | 0.857 | 0.880 |
| overall | p_instability_H1 | 931 | 0.944 | 0.984 | 0.040 | 0.856 | 0.870 | 0.582 | 0.505 |
| overall | p_instability_H3 | 931 | 0.960 | 0.984 | 0.025 | 0.842 | 0.848 | 0.578 | 0.498 |
| overall | p_instability_H5 | 931 | 0.967 | 0.983 | 0.017 | 0.811 | 0.816 | 0.577 | 0.497 |
| cucumber_hammer | p_slip_current | 401 | 0.283 | 0.930 | 0.647 | 0.947 | 0.984 | 0.907 | 0.809 |
| cucumber_hammer | p_instability_H1 | 401 | 0.943 | 0.989 | 0.046 | 0.822 | 0.932 | 0.834 | 0.515 |
| cucumber_hammer | p_instability_H3 | 401 | 0.963 | 0.989 | 0.026 | 0.806 | 0.927 | 0.830 | 0.505 |
| cucumber_hammer | p_instability_H5 | 401 | 0.970 | 0.988 | 0.018 | 0.772 | 0.912 | 0.827 | 0.496 |

## Original-validation thresholds applied to real data

| subset | score | threshold | stable acc | slip recall | balanced acc | F1 | FP | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | p_slip_current | 0.500 | 0.871 | 0.890 | 0.880 | 0.857 | 71 | 479 |
| overall | p_instability_H1 | 0.848 | 0.093 | 0.982 | 0.537 | 0.596 | 499 | 51 |
| overall | p_instability_H3 | 0.728 | 0.036 | 0.987 | 0.512 | 0.584 | 530 | 20 |
| overall | p_instability_H5 | 0.784 | 0.035 | 0.987 | 0.511 | 0.584 | 531 | 19 |
| cucumber_hammer | p_slip_current | 0.500 | 0.681 | 0.937 | 0.809 | 0.907 | 37 | 79 |
| cucumber_hammer | p_instability_H1 | 0.848 | 0.103 | 0.989 | 0.546 | 0.841 | 104 | 12 |
| cucumber_hammer | p_instability_H3 | 0.728 | 0.034 | 0.989 | 0.512 | 0.831 | 112 | 4 |
| cucumber_hammer | p_instability_H5 | 0.784 | 0.034 | 0.989 | 0.512 | 0.831 | 112 | 4 |

## Diagnostic best thresholds on real data (not for model selection)

| subset | score | best threshold | balanced acc | F1 | stable acc | slip recall |
|---|---|---:|---:|---:|---:|---:|
| overall | p_instability_H1 | 0.995 | 0.820 | 0.787 | 0.858 | 0.782 |
| overall | p_instability_H3 | 0.990 | 0.800 | 0.767 | 0.773 | 0.827 |
| overall | p_instability_H5 | 0.990 | 0.752 | 0.715 | 0.738 | 0.766 |
| cucumber_hammer | p_instability_H1 | 0.990 | 0.706 | 0.856 | 0.517 | 0.895 |
| cucumber_hammer | p_instability_H3 | 0.995 | 0.679 | 0.821 | 0.526 | 0.832 |
| cucumber_hammer | p_instability_H5 | 0.995 | 0.659 | 0.801 | 0.517 | 0.800 |

## Cucumber/Hammer window means

| sequence | side | window | range | n | pSlip | H1 | H3 | H5 |
|---|---|---|---|---:|---:|---:|---:|---:|
| cucumber_1 | left | slip | 35-75 | 41 | 0.838 | 0.990 | 0.996 | 0.996 |
| cucumber_1 | right | slip | 48-73 | 26 | 0.984 | 0.997 | 0.998 | 0.997 |
| cucumber_2 | left | slip | 39-52 | 14 | 0.862 | 0.984 | 0.992 | 0.993 |
| cucumber_2 | right | slip | 38-51 | 14 | 0.996 | 0.997 | 0.997 | 0.996 |
| cucumber_3 | left | slip | 40-62 | 23 | 0.591 | 0.973 | 0.986 | 0.987 |
| cucumber_3 | left | stable | 80-110 | 31 | 0.025 | 0.980 | 0.990 | 0.992 |
| cucumber_3 | right | slip | 41-64 | 24 | 0.936 | 0.993 | 0.996 | 0.997 |
| cucumber_3 | right | stable | 80-110 | 31 | 0.039 | 0.810 | 0.875 | 0.901 |
| hammer_1 | left | slip | 50-80 | 31 | 1.000 | 0.962 | 0.944 | 0.940 |
| hammer_1 | right | slip | 50-80 | 31 | 1.000 | 1.000 | 0.999 | 0.999 |
| hammer_2 | left | slip | 34-44 | 11 | 0.995 | 0.997 | 0.990 | 0.988 |
| hammer_2 | left | stable | 90-116 | 27 | 0.723 | 0.999 | 0.998 | 0.998 |
| hammer_2 | right | slip | 34-41 | 8 | 1.000 | 0.996 | 0.995 | 0.995 |
| hammer_2 | right | stable | 90-116 | 27 | 0.419 | 0.998 | 0.998 | 0.998 |
| hammer_3 | left | slip | 40-70 | 31 | 1.000 | 0.990 | 0.984 | 0.981 |
| hammer_3 | right | slip | 40-70 | 31 | 1.000 | 1.000 | 1.000 | 0.999 |

## Main observations

- `p_slip_current` remains the most deployment-ready signal: it has a large stable/slip score gap and good F1 at the default 0.5 threshold.
- The future-instability head has useful ranking separation, but H1/H3/H5 raw probabilities remain shifted upward on real images; this is why stable windows still look unstable at threshold 0.5.
- Cucumber and hammer slip windows are clearer/longer, so they are better for qualitative figures. However hammer stable windows still produce high future-instability scores, highlighting the remaining domain-calibration gap.
- For paper use without training on real data, report AUROC/AUPRC/gap and original-validation threshold results, not a threshold tuned on real validation labels.

## Generated files

- threshold_sweep_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354/threshold_sweep_metrics.csv`
- object_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354/object_window_means.csv`
- sequence_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354/sequence_window_means.csv`
- cucumber_hammer_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354/cucumber_hammer_window_means.csv`
- markdown: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354/quick_real_future_analysis.md`
- plots: `['/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354/score_distributions_stable_vs_slip.png', '/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354/future_threshold_sweep_balanced_accuracy.png', '/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_original_val_calibration/20260530_030301/quick_analysis/20260530_030354/cucumber_hammer_window_scores.png']`
