# Real Force-Slip Calibration Report

- generated_at: `2026-05-29T22:15:35`
- input_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_per_frame_predictions.csv`
- report_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_calibration/20260529_221531`
- features: `p_slip_current, p_instability_H1, p_instability_H3, p_instability_H5, Fn_pred_N, Ft_pred_N, Fmag_pred_N, Ft_over_Fn_pred, dFx_causal_N, dFy_causal_N, dFz_causal_N`
- label rule: `stable_windows -> 0`, `slip_windows -> 1`.
- This is real-world calibration only: no Sparsh backbone, encoder, decoder, or checkpoint weights were retrained or modified.
- Purpose: bridge the domain gap between controlled flat/sharp/sphere training data and real banana/cucumber/hammer/lemon grasp trials.

## Headline results

| split | best method | level | n | threshold | AUROC | AUPRC | F1 | Acc | BalAcc | stable mean | slip mean | gap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sequence_split | window_p_slip_current_mean_thresholded | window | 12 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.804 | 0.804 |
| object_holdout | window_p_slip_current_max_thresholded | window | 16 | 0.970 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.468 | 1.000 | 0.532 |

## Frame-level test comparison

### sequence_split

| method | threshold | AUROC | AUPRC | F1 | Acc | BalAcc | stable mean | slip mean | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_p_slip_current@0.5 | 0.500 | 0.978 | 0.985 | 0.932 | 0.932 | 0.937 | 0.000 | 0.867 | 0.866 |
| threshold_p_slip_current | 0.962 | 0.978 | 0.985 | 0.886 | 0.891 | 0.898 | 0.000 | 0.867 | 0.866 |
| raw_p_instability_H1@0.5 | 0.500 | 0.380 | 0.476 | 0.696 | 0.534 | 0.500 | 1.000 | 0.995 | -0.005 |
| threshold_p_instability_H1 | 0.946 | 0.380 | 0.476 | 0.686 | 0.523 | 0.489 | 1.000 | 0.995 | -0.005 |
| raw_p_instability_H3@0.5 | 0.500 | 0.380 | 0.475 | 0.696 | 0.534 | 0.500 | 1.000 | 1.000 | -0.000 |
| raw_p_instability_H5@0.5 | 0.500 | 0.423 | 0.483 | 0.696 | 0.534 | 0.500 | 1.000 | 1.000 | -0.000 |
| logistic_calibration@0.5 | 0.500 | 0.938 | 0.966 | 0.940 | 0.940 | 0.944 | 0.225 | 0.893 | 0.668 |
| logistic_calibration_thresholded | 0.938 | 0.938 | 0.966 | 0.908 | 0.910 | 0.915 | 0.225 | 0.893 | 0.668 |

### object_holdout

| method | threshold | AUROC | AUPRC | F1 | Acc | BalAcc | stable mean | slip mean | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_p_slip_current@0.5 | 0.500 | 0.910 | 0.973 | 0.890 | 0.839 | 0.789 | 0.283 | 0.870 | 0.587 |
| threshold_p_slip_current | 0.970 | 0.910 | 0.973 | 0.840 | 0.799 | 0.862 | 0.283 | 0.870 | 0.587 |
| raw_p_instability_H1@0.5 | 0.500 | 0.622 | 0.762 | 0.862 | 0.771 | 0.594 | 0.792 | 0.964 | 0.173 |
| threshold_p_instability_H1 | 0.946 | 0.622 | 0.762 | 0.822 | 0.724 | 0.600 | 0.792 | 0.964 | 0.173 |
| raw_p_instability_H3@0.5 | 0.500 | 0.584 | 0.766 | 0.849 | 0.741 | 0.522 | 0.951 | 0.998 | 0.047 |
| raw_p_instability_H5@0.5 | 0.500 | 0.592 | 0.767 | 0.846 | 0.734 | 0.509 | 0.981 | 1.000 | 0.018 |
| logistic_calibration@0.5 | 0.500 | 0.918 | 0.975 | 0.870 | 0.799 | 0.694 | 0.598 | 0.931 | 0.333 |
| logistic_calibration_thresholded | 0.990 | 0.918 | 0.975 | 0.791 | 0.748 | 0.827 | 0.598 | 0.931 | 0.333 |

## Split details

### sequence_split

- test: frames=266, stable=124, slip=142, windows=12, sequences=['cucumber_2', 'hammer_3', 'lemon_4', 'lemon_5']
- train: frames=494, stable=248, slip=246, windows=20, sequences=['banana_1', 'cucumber_1', 'cucumber_3', 'hammer_1', 'lemon_1', 'lemon_2']
- val: frames=272, stable=178, slip=94, windows=12, sequences=['banana_2', 'hammer_2', 'lemon_3']

### object_holdout

- test: frames=428, stable=116, slip=312, windows=16, sequences=['cucumber_1', 'cucumber_2', 'cucumber_3', 'hammer_1', 'hammer_2', 'hammer_3']
- train: frames=428, stable=310, slip=118, windows=20, sequences=['banana_1', 'lemon_1', 'lemon_2', 'lemon_4', 'lemon_5']
- val: frames=176, stable=124, slip=52, windows=8, sequences=['banana_2', 'lemon_3']

## Output files

- json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_calibration/20260529_221531/real_force_slip_calibration_report.json`
- markdown: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_calibration/20260529_221531/real_force_slip_calibration_report.md`
- sequence_split_frame_metric_bars: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_calibration/20260529_221531/sequence_split_frame_metric_bars.png`
- object_holdout_frame_metric_bars: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_calibration/20260529_221531/object_holdout_frame_metric_bars.png`

## Practical recommendation

- sequence_split: raw p_slip_current remains the strongest uncalibrated real-data signal (F1=0.932, AUROC=0.978).
- sequence_split: raw H1 future-instability is poorly calibrated on real data (F1=0.696, AUROC=0.380), consistent with the high stable-window probabilities observed earlier.
- sequence_split: threshold fitting p_slip_current gives a lightweight deployment rule (threshold=0.962, F1=0.886).
- sequence_split: logistic calibration combines slip/future/force proxies (threshold=0.938, F1=0.908, BalAcc=0.915); use it as a real-world calibration layer only if test metrics exceed the simpler threshold baseline.
- object_holdout: raw p_slip_current remains the strongest uncalibrated real-data signal (F1=0.890, AUROC=0.910).
- object_holdout: raw H1 future-instability is poorly calibrated on real data (F1=0.862, AUROC=0.622), consistent with the high stable-window probabilities observed earlier.
- object_holdout: threshold fitting p_slip_current gives a lightweight deployment rule (threshold=0.970, F1=0.840).
- object_holdout: logistic calibration combines slip/future/force proxies (threshold=0.990, F1=0.791, BalAcc=0.827); use it as a real-world calibration layer only if test metrics exceed the simpler threshold baseline.
