# Quick Real Future-instability Analysis

- generated_at: `2026-05-30T02:58:17`
- source_report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749`
- output_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816`
- real-data policy: this is **analysis only**. The real banana/cucumber/hammer/lemon dataset is not used for model training, checkpoint selection, or final threshold selection.
- threshold sweep on real data is marked diagnostic; original-validation thresholds are reported separately.

## Score separation summary

| subset | score | n | stable mean | slip mean | gap | AUROC | AUPRC | F1@0.5 | BalAcc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | p_slip_current | 931 | 0.122 | 0.877 | 0.755 | 0.937 | 0.948 | 0.857 | 0.880 |
| overall | p_instability_H1 | 931 | 0.822 | 0.970 | 0.148 | 0.931 | 0.941 | 0.599 | 0.538 |
| overall | p_instability_H3 | 931 | 0.811 | 0.962 | 0.151 | 0.924 | 0.925 | 0.590 | 0.519 |
| overall | p_instability_H5 | 931 | 0.835 | 0.964 | 0.130 | 0.927 | 0.928 | 0.585 | 0.508 |
| cucumber_hammer | p_slip_current | 401 | 0.283 | 0.930 | 0.647 | 0.947 | 0.984 | 0.907 | 0.809 |
| cucumber_hammer | p_instability_H1 | 401 | 0.846 | 0.979 | 0.133 | 0.944 | 0.981 | 0.844 | 0.547 |
| cucumber_hammer | p_instability_H3 | 401 | 0.878 | 0.979 | 0.101 | 0.955 | 0.985 | 0.841 | 0.534 |
| cucumber_hammer | p_instability_H5 | 401 | 0.894 | 0.979 | 0.085 | 0.953 | 0.985 | 0.835 | 0.513 |

## Original-validation thresholds applied to real data

| subset | score | threshold | stable acc | slip recall | balanced acc | F1 | FP | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| overall | p_slip_current | 0.500 | 0.871 | 0.890 | 0.880 | 0.857 | 71 | 479 |
| overall | p_instability_H1 | 0.848 | 0.375 | 0.948 | 0.661 | 0.665 | 344 | 206 |
| overall | p_instability_H3 | 0.728 | 0.196 | 0.990 | 0.593 | 0.628 | 442 | 108 |
| overall | p_instability_H5 | 0.784 | 0.207 | 0.979 | 0.593 | 0.627 | 436 | 114 |
| cucumber_hammer | p_slip_current | 0.500 | 0.681 | 0.937 | 0.809 | 0.907 | 37 | 79 |
| cucumber_hammer | p_instability_H1 | 0.848 | 0.250 | 0.961 | 0.606 | 0.848 | 87 | 29 |
| cucumber_hammer | p_instability_H3 | 0.728 | 0.095 | 1.000 | 0.547 | 0.844 | 105 | 11 |
| cucumber_hammer | p_instability_H5 | 0.784 | 0.095 | 1.000 | 0.547 | 0.844 | 105 | 11 |

## Diagnostic best thresholds on real data (not for model selection)

| subset | score | best threshold | balanced acc | F1 | stable acc | slip recall |
|---|---|---:|---:|---:|---:|---:|
| overall | p_instability_H1 | 0.970 | 0.882 | 0.863 | 0.933 | 0.832 |
| overall | p_instability_H3 | 0.950 | 0.881 | 0.860 | 0.929 | 0.832 |
| overall | p_instability_H5 | 0.950 | 0.880 | 0.859 | 0.913 | 0.848 |
| cucumber_hammer | p_instability_H1 | 0.980 | 0.923 | 0.916 | 1.000 | 0.846 |
| cucumber_hammer | p_instability_H3 | 0.950 | 0.936 | 0.953 | 0.940 | 0.933 |
| cucumber_hammer | p_instability_H5 | 0.960 | 0.913 | 0.915 | 0.974 | 0.853 |

## Cucumber/Hammer window means

| sequence | side | window | range | n | pSlip | H1 | H3 | H5 |
|---|---|---|---|---:|---:|---:|---:|---:|
| cucumber_1 | left | slip | 35-75 | 41 | 0.838 | 0.957 | 0.973 | 0.974 |
| cucumber_1 | right | slip | 48-73 | 26 | 0.984 | 0.991 | 0.993 | 0.991 |
| cucumber_2 | left | slip | 39-52 | 14 | 0.862 | 0.939 | 0.959 | 0.961 |
| cucumber_2 | right | slip | 38-51 | 14 | 0.996 | 0.979 | 0.981 | 0.978 |
| cucumber_3 | left | slip | 40-62 | 23 | 0.591 | 0.957 | 0.968 | 0.967 |
| cucumber_3 | left | stable | 80-110 | 31 | 0.025 | 0.907 | 0.942 | 0.944 |
| cucumber_3 | right | slip | 41-64 | 24 | 0.936 | 0.965 | 0.979 | 0.978 |
| cucumber_3 | right | stable | 80-110 | 31 | 0.039 | 0.579 | 0.716 | 0.762 |
| hammer_1 | left | slip | 50-80 | 31 | 1.000 | 0.990 | 0.967 | 0.968 |
| hammer_1 | right | slip | 50-80 | 31 | 1.000 | 0.998 | 0.994 | 0.993 |
| hammer_2 | left | slip | 34-44 | 11 | 0.995 | 0.987 | 0.963 | 0.965 |
| hammer_2 | left | stable | 90-116 | 27 | 0.723 | 0.971 | 0.935 | 0.943 |
| hammer_2 | right | slip | 34-41 | 8 | 1.000 | 0.995 | 0.986 | 0.985 |
| hammer_2 | right | stable | 90-116 | 27 | 0.419 | 0.959 | 0.932 | 0.939 |
| hammer_3 | left | slip | 40-70 | 31 | 1.000 | 0.991 | 0.975 | 0.975 |
| hammer_3 | right | slip | 40-70 | 31 | 1.000 | 0.998 | 0.994 | 0.994 |

## Main observations

- `p_slip_current` remains the most deployment-ready signal: it has a large stable/slip score gap and good F1 at the default 0.5 threshold.
- The future-instability head has useful ranking separation, but H1/H3/H5 raw probabilities remain shifted upward on real images; this is why stable windows still look unstable at threshold 0.5.
- Cucumber and hammer slip windows are clearer/longer, so they are better for qualitative figures. However hammer stable windows still produce high future-instability scores, highlighting the remaining domain-calibration gap.
- For paper use without training on real data, report AUROC/AUPRC/gap and original-validation threshold results, not a threshold tuned on real validation labels.

## Generated files

- threshold_sweep_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/threshold_sweep_metrics.csv`
- object_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/object_window_means.csv`
- sequence_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/sequence_window_means.csv`
- cucumber_hammer_window_means_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/cucumber_hammer_window_means.csv`
- markdown: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/quick_real_future_analysis.md`
- plots: `['/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/score_distributions_stable_vs_slip.png', '/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/future_threshold_sweep_balanced_accuracy.png', '/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/quick_analysis/20260530_025816/cucumber_hammer_window_scores.png']`
