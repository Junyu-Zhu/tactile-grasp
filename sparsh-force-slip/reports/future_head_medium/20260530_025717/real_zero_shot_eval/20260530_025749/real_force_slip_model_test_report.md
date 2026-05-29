# Real Force-Slip Model Test Report (Stable/Slip Windows)

- generated_at: `2026-05-30T02:58:14`
- dataset: `/vla1/zjy/tactile_dataset/real-force-slip-model-test-dataset`
- stage_i_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- stage_ii_future_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_future_medium/20260530_025717/heads/medium_hard_aux_balanced/checkpoints/best.pth`
- label_source: `future_labels`; stable/future-safe windows -> target_instability=0; slip/future-slip windows -> target_instability=1
- evaluated_frames: `931`; sequence_sides: `26`; windows: `44`
- preprocessing: first frame background subtraction = `True`, context stride = `5` frames, resized to `[320, 240]`.

## Interpretation notes

- This run uses manually annotated local stable/slip windows and reports left/right sensors separately.
- The labels are diagnostic deployment labels from the real image sequences, not calibrated force ground truth.
- `p_slip_current` is the Stage-I slip probability; `p_instability_H*` is the Stage-II future-instability probability.

## Binary separation metrics

| score | n | stable mean | slip mean | gap | AUROC | AUPRC | F1@0.5 | Acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p_slip_current | 931 | 0.122 | 0.877 | 0.755 | 0.937 | 0.948 | 0.857 | 0.879 |
| p_instability_H1 | 931 | 0.822 | 0.970 | 0.148 | 0.931 | 0.941 | 0.599 | 0.455 |
| p_instability_H3 | 931 | 0.811 | 0.962 | 0.151 | 0.924 | 0.925 | 0.590 | 0.432 |
| p_instability_H5 | 931 | 0.835 | 0.964 | 0.130 | 0.927 | 0.928 | 0.585 | 0.419 |

## Stable vs slip window summary

| window | sides | windows | frames | pSlip mean | H1 inst mean | H5 inst mean | Fmag mean | Ft/Fn mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| slip | 26 | 26 | 381 | 0.877 | 0.970 | 0.964 | 1.458 | 0.942 |
| stable | 18 | 18 | 550 | 0.122 | 0.822 | 0.835 | 1.520 | 0.872 |

## Object-level summary

| object | window | sides | windows | frames | pSlip mean | H1 inst mean | H5 inst mean | Fmag mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| banana | slip | 4 | 4 | 29 | 0.777 | 0.907 | 0.944 | 0.460 |
| banana | stable | 4 | 4 | 124 | 0.276 | 0.694 | 0.828 | 0.505 |
| cucumber | slip | 6 | 6 | 142 | 0.859 | 0.965 | 0.976 | 0.475 |
| cucumber | stable | 2 | 2 | 62 | 0.032 | 0.743 | 0.853 | 0.633 |
| hammer | slip | 6 | 6 | 143 | 1.000 | 0.994 | 0.981 | 2.355 |
| hammer | stable | 2 | 2 | 54 | 0.571 | 0.965 | 0.941 | 2.395 |
| lemon | slip | 10 | 10 | 67 | 0.697 | 0.956 | 0.913 | 2.056 |
| lemon | stable | 10 | 10 | 310 | 0.001 | 0.864 | 0.816 | 1.951 |

## Sequence / side / window summary

| sequence | side | window | range | frames | pSlip mean | slip rate | H1 inst mean | H5 inst mean | Fmag mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| banana_1 | left | slip | 30-33 | 4 | 0.310 | 0.500 | 0.743 | 0.869 | 1.060 |
| banana_1 | left | stable | 40-70 | 31 | 0.001 | 0.000 | 0.343 | 0.642 | 1.002 |
| banana_1 | right | slip | 28-32 | 5 | 0.640 | 0.600 | 0.941 | 0.960 | 0.409 |
| banana_1 | right | stable | 40-70 | 31 | 0.089 | 0.065 | 0.782 | 0.856 | 0.387 |
| banana_2 | left | slip | 33-40 | 8 | 0.934 | 1.000 | 0.955 | 0.964 | 0.197 |
| banana_2 | left | stable | 60-90 | 31 | 0.720 | 0.806 | 0.916 | 0.948 | 0.246 |
| banana_2 | right | slip | 29-40 | 12 | 0.886 | 0.917 | 0.915 | 0.948 | 0.457 |
| banana_2 | right | stable | 60-90 | 31 | 0.294 | 0.226 | 0.735 | 0.864 | 0.383 |
| cucumber_1 | left | slip | 35-75 | 41 | 0.838 | 0.854 | 0.957 | 0.974 | 0.520 |
| cucumber_1 | right | slip | 48-73 | 26 | 0.984 | 1.000 | 0.991 | 0.991 | 0.414 |
| cucumber_2 | left | slip | 39-52 | 14 | 0.862 | 0.857 | 0.939 | 0.961 | 0.210 |
| cucumber_2 | right | slip | 38-51 | 14 | 0.996 | 1.000 | 0.979 | 0.978 | 0.259 |
| cucumber_3 | left | slip | 40-62 | 23 | 0.591 | 0.609 | 0.957 | 0.967 | 0.533 |
| cucumber_3 | left | stable | 80-110 | 31 | 0.025 | 0.000 | 0.907 | 0.944 | 0.550 |
| cucumber_3 | right | slip | 41-64 | 24 | 0.936 | 0.958 | 0.965 | 0.978 | 0.686 |
| cucumber_3 | right | stable | 80-110 | 31 | 0.039 | 0.000 | 0.579 | 0.762 | 0.716 |
| hammer_1 | left | slip | 50-80 | 31 | 1.000 | 1.000 | 0.990 | 0.968 | 2.379 |
| hammer_1 | right | slip | 50-80 | 31 | 1.000 | 1.000 | 0.998 | 0.993 | 2.324 |
| hammer_2 | left | slip | 34-44 | 11 | 0.995 | 1.000 | 0.987 | 0.965 | 2.312 |
| hammer_2 | left | stable | 90-116 | 27 | 0.723 | 1.000 | 0.971 | 0.943 | 2.409 |
| hammer_2 | right | slip | 34-41 | 8 | 1.000 | 1.000 | 0.995 | 0.985 | 2.334 |
| hammer_2 | right | stable | 90-116 | 27 | 0.419 | 0.370 | 0.959 | 0.939 | 2.382 |
| hammer_3 | left | slip | 40-70 | 31 | 1.000 | 1.000 | 0.991 | 0.975 | 2.377 |
| hammer_3 | right | slip | 40-70 | 31 | 1.000 | 1.000 | 0.998 | 0.994 | 2.362 |
| lemon_1 | left | slip | 34-37 | 4 | 0.604 | 0.750 | 0.927 | 0.955 | 0.596 |
| lemon_1 | left | stable | 50-80 | 31 | 0.001 | 0.000 | 0.667 | 0.808 | 0.641 |
| lemon_1 | right | slip | 31-34 | 4 | 0.500 | 0.500 | 0.854 | 0.931 | 0.641 |
| lemon_1 | right | stable | 50-80 | 31 | 0.007 | 0.000 | 0.769 | 0.899 | 0.644 |
| lemon_2 | left | slip | 31-37 | 7 | 0.602 | 0.714 | 0.959 | 0.882 | 2.266 |
| lemon_2 | left | stable | 60-90 | 31 | 0.000 | 0.000 | 0.912 | 0.804 | 2.336 |
| lemon_2 | right | slip | 28-35 | 8 | 0.577 | 0.625 | 0.947 | 0.867 | 2.339 |
| lemon_2 | right | stable | 60-90 | 31 | 0.000 | 0.000 | 0.904 | 0.793 | 2.396 |
| lemon_3 | left | slip | 28-35 | 8 | 0.661 | 0.625 | 0.934 | 0.871 | 2.342 |
| lemon_3 | left | stable | 70-100 | 31 | 0.000 | 0.000 | 0.839 | 0.764 | 2.373 |
| lemon_3 | right | slip | 28-35 | 8 | 0.497 | 0.500 | 0.956 | 0.903 | 2.088 |
| lemon_3 | right | stable | 70-100 | 31 | 0.000 | 0.000 | 0.927 | 0.852 | 2.109 |
| lemon_4 | left | slip | 36-41 | 6 | 0.951 | 1.000 | 0.990 | 0.961 | 2.328 |
| lemon_4 | left | stable | 70-100 | 31 | 0.000 | 0.000 | 0.894 | 0.802 | 2.363 |
| lemon_4 | right | slip | 36-41 | 6 | 0.885 | 0.833 | 0.982 | 0.937 | 2.142 |
| lemon_4 | right | stable | 70-100 | 31 | 0.001 | 0.000 | 0.849 | 0.738 | 2.223 |
| lemon_5 | left | slip | 34-41 | 8 | 0.647 | 0.625 | 0.971 | 0.906 | 2.290 |
| lemon_5 | left | stable | 70-100 | 31 | 0.000 | 0.000 | 0.951 | 0.857 | 2.233 |
| lemon_5 | right | slip | 34-41 | 8 | 0.999 | 1.000 | 0.993 | 0.962 | 2.210 |
| lemon_5 | right | stable | 70-100 | 31 | 0.000 | 0.000 | 0.932 | 0.837 | 2.191 |

## Automatic conclusion

- p_slip_current: stable_mean=0.122, slip_mean=0.877, gap=0.755, AUROC=0.937.
- p_instability_H1: stable_mean=0.822, slip_mean=0.970, gap=0.148, AUROC=0.931.
- p_instability_H3: stable_mean=0.811, slip_mean=0.962, gap=0.151, AUROC=0.924.
- p_instability_H5: stable_mean=0.835, slip_mean=0.964, gap=0.130, AUROC=0.927.

## Output files

- json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_model_test_report.json`
- markdown: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_model_test_report.md`
- per_frame_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_per_frame_predictions.csv`
- sequence_side_window_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_sequence_side_window_summary.csv`
- window_type_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_window_type_summary.csv`
- object_window_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_object_window_summary.csv`
- sequence_side_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_sequence_side_summary.csv`
- side_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_side_summary.csv`
- binary_metrics_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_binary_metrics.csv`
- stable_slip_metrics_png: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_medium/20260530_025717/real_zero_shot_eval/20260530_025749/real_force_slip_stable_slip_metrics.png`
