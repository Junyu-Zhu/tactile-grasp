# Quick Real Future-instability Analysis

- generated_at: `2026-05-30T03:01:09`
- source_report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/real_zero_shot_eval/20260530_030018`
- output_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107`
- real-data policy: this is **analysis only**. The real banana/cucumber/hammer/lemon dataset is not used for model training, checkpoint selection, or final threshold selection.
- threshold sweep on real data is marked diagnostic; original-validation thresholds are reported separately.

## Score separation summary

| subset | score | n | stable mean | slip mean | gap | AUROC | AUPRC | F1@0.5 | BalAcc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | p_slip_current | 931 | 0.122 | 0.877 | 0.755 | 0.937 | 0.948 | 0.857 | 0.880 |
| overall | p_instability_H1 | 931 | 0.770 | 0.933 | 0.164 | 0.856 | 0.870 | 0.603 | 0.553 |
| overall | p_instability_H3 | 931 | 0.807 | 0.937 | 0.130 | 0.842 | 0.848 | 0.594 | 0.533 |
| overall | p_instability_H5 | 931 | 0.845 | 0.939 | 0.094 | 0.811 | 0.816 | 0.590 | 0.524 |
| cucumber_hammer | p_slip_current | 401 | 0.283 | 0.930 | 0.647 | 0.947 | 0.984 | 0.907 | 0.809 |
| cucumber_hammer | p_instability_H1 | 401 | 0.804 | 0.950 | 0.146 | 0.822 | 0.932 | 0.840 | 0.558 |
| cucumber_hammer | p_instability_H3 | 401 | 0.863 | 0.957 | 0.095 | 0.806 | 0.927 | 0.842 | 0.551 |
| cucumber_hammer | p_instability_H5 | 401 | 0.890 | 0.960 | 0.070 | 0.772 | 0.912 | 0.838 | 0.538 |

## Original-validation thresholds applied to real data

| subset | score | threshold | stable acc | slip recall | balanced acc | F1 | FP | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | p_slip_current | 0.500 | 0.871 | 0.890 | 0.880 | 0.857 | 71 | 479 |
| overall | p_instability_H1 | 0.848 | 0.465 | 0.877 | 0.671 | 0.662 | 294 | 256 |
| overall | p_instability_H3 | 0.728 | 0.144 | 0.948 | 0.546 | 0.595 | 471 | 79 |
| overall | p_instability_H5 | 0.784 | 0.133 | 0.937 | 0.535 | 0.588 | 477 | 73 |
| cucumber_hammer | p_slip_current | 0.500 | 0.681 | 0.937 | 0.809 | 0.907 | 37 | 79 |
| cucumber_hammer | p_instability_H1 | 0.848 | 0.422 | 0.909 | 0.666 | 0.848 | 67 | 49 |
| cucumber_hammer | p_instability_H3 | 0.728 | 0.147 | 0.965 | 0.556 | 0.835 | 99 | 17 |
| cucumber_hammer | p_instability_H5 | 0.784 | 0.129 | 0.961 | 0.545 | 0.830 | 101 | 15 |

## Diagnostic best thresholds on real data (not for model selection)

| subset | score | best threshold | balanced acc | F1 | stable acc | slip recall |
|---|---|---:|---:|---:|---:|---:|
| overall | p_instability_H1 | 0.960 | 0.823 | 0.790 | 0.876 | 0.769 |
| overall | p_instability_H3 | 0.940 | 0.806 | 0.771 | 0.833 | 0.780 |
| overall | p_instability_H5 | 0.950 | 0.762 | 0.716 | 0.825 | 0.698 |
| cucumber_hammer | p_instability_H1 | 0.990 | 0.768 | 0.756 | 0.905 | 0.632 |
| cucumber_hammer | p_instability_H3 | 0.990 | 0.781 | 0.719 | 1.000 | 0.561 |
| cucumber_hammer | p_instability_H5 | 0.990 | 0.745 | 0.676 | 0.974 | 0.516 |

## Cucumber/Hammer window means

| sequence | side | window | range | n | pSlip | H1 | H3 | H5 |
|---|---|---|---|---:|---:|---:|---:|---:|
| cucumber_1 | left | slip | 35-75 | 41 | 0.838 | 0.934 | 0.966 | 0.975 |
| cucumber_1 | right | slip | 48-73 | 26 | 0.984 | 0.973 | 0.981 | 0.980 |
| cucumber_2 | left | slip | 39-52 | 14 | 0.862 | 0.915 | 0.948 | 0.959 |
| cucumber_2 | right | slip | 38-51 | 14 | 0.996 | 0.969 | 0.976 | 0.972 |
| cucumber_3 | left | slip | 40-62 | 23 | 0.591 | 0.829 | 0.895 | 0.917 |
| cucumber_3 | left | stable | 80-110 | 31 | 0.025 | 0.829 | 0.916 | 0.940 |
| cucumber_3 | right | slip | 41-64 | 24 | 0.936 | 0.948 | 0.969 | 0.975 |
| cucumber_3 | right | stable | 80-110 | 31 | 0.039 | 0.464 | 0.602 | 0.677 |
| hammer_1 | left | slip | 50-80 | 31 | 1.000 | 0.928 | 0.905 | 0.900 |
| hammer_1 | right | slip | 50-80 | 31 | 1.000 | 0.998 | 0.994 | 0.992 |
| hammer_2 | left | slip | 34-44 | 11 | 0.995 | 0.969 | 0.928 | 0.927 |
| hammer_2 | left | stable | 90-116 | 27 | 0.723 | 0.989 | 0.983 | 0.984 |
| hammer_2 | right | slip | 34-41 | 8 | 1.000 | 0.963 | 0.960 | 0.962 |
| hammer_2 | right | stable | 90-116 | 27 | 0.419 | 0.982 | 0.981 | 0.985 |
| hammer_3 | left | slip | 40-70 | 31 | 1.000 | 0.965 | 0.946 | 0.942 |
| hammer_3 | right | slip | 40-70 | 31 | 1.000 | 0.999 | 0.997 | 0.996 |

## Main observations

- `p_slip_current` remains the most deployment-ready signal: it has a large stable/slip score gap and good F1 at the default 0.5 threshold.
- The future-instability head has useful ranking separation, but H1/H3/H5 raw probabilities remain shifted upward on real images; this is why stable windows still look unstable at threshold 0.5.
- Cucumber and hammer slip windows are clearer/longer, so they are better for qualitative figures. However hammer stable windows still produce high future-instability scores, highlighting the remaining domain-calibration gap.
- For paper use without training on real data, report AUROC/AUPRC/gap and original-validation threshold results, not a threshold tuned on real validation labels.

## Generated files

- threshold_sweep_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/threshold_sweep_metrics.csv`
- object_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/object_window_means.csv`
- sequence_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/sequence_window_means.csv`
- cucumber_hammer_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/cucumber_hammer_window_means.csv`
- markdown: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/quick_real_future_analysis.md`
- plots: `['/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/score_distributions_stable_vs_slip.png', '/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/future_threshold_sweep_balanced_accuracy.png', '/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_transition/20260530_025949/quick_analysis/20260530_030107/cucumber_hammer_window_scores.png']`
