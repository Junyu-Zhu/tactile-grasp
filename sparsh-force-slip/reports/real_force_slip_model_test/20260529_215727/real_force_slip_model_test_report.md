# Real Force-Slip Model Test Report (Stable/Slip Windows)

- generated_at: `2026-05-29T21:57:53`
- dataset: `/vla1/zjy/tactile_dataset/real-force-slip-model-test-dataset`
- stage_i_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- stage_ii_future_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth`
- labels: `stable_windows -> target_instability=0`, `slip_windows -> target_instability=1`
- evaluated_frames: `1032`; sequence_sides: `26`; windows: `44`
- preprocessing: first frame background subtraction = `True`, context stride = `5` frames, resized to `[320, 240]`.

## Interpretation notes

- This run uses manually annotated local stable/slip windows and reports left/right sensors separately.
- The labels are diagnostic deployment labels from the real image sequences, not calibrated force ground truth.
- `p_slip_current` is the Stage-I slip probability; `p_instability_H*` is the Stage-II future-instability probability.

## Binary separation metrics

| score | n | stable mean | slip mean | gap | AUROC | AUPRC | F1@0.5 | Acc@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p_slip_current | 1032 | 0.122 | 0.773 | 0.650 | 0.891 | 0.915 | 0.822 | 0.838 |
| p_instability_H1 | 1032 | 0.846 | 0.954 | 0.108 | 0.565 | 0.535 | 0.649 | 0.508 |
| p_instability_H3 | 1032 | 0.967 | 0.995 | 0.028 | 0.555 | 0.570 | 0.643 | 0.483 |
| p_instability_H5 | 1032 | 0.980 | 0.997 | 0.018 | 0.568 | 0.523 | 0.640 | 0.477 |

## Stable vs slip window summary

| window | sides | windows | frames | pSlip mean | H1 inst mean | H5 inst mean | Fmag mean | Ft/Fn mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| slip | 26 | 26 | 482 | 0.773 | 0.954 | 0.997 | 1.481 | 0.916 |
| stable | 18 | 18 | 550 | 0.122 | 0.846 | 0.980 | 1.520 | 0.872 |

## Object-level summary

| object | window | sides | windows | frames | pSlip mean | H1 inst mean | H5 inst mean | Fmag mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| banana | slip | 4 | 4 | 52 | 0.650 | 0.865 | 0.977 | 0.498 |
| banana | stable | 4 | 4 | 124 | 0.276 | 0.693 | 0.947 | 0.505 |
| cucumber | slip | 6 | 6 | 146 | 0.769 | 0.924 | 0.999 | 0.471 |
| cucumber | stable | 2 | 2 | 62 | 0.032 | 0.610 | 0.965 | 0.633 |
| hammer | slip | 6 | 6 | 166 | 0.960 | 1.000 | 1.000 | 2.360 |
| hammer | stable | 2 | 2 | 54 | 0.571 | 1.000 | 1.000 | 2.395 |
| lemon | slip | 10 | 10 | 118 | 0.570 | 0.965 | 1.000 | 1.926 |
| lemon | stable | 10 | 10 | 310 | 0.001 | 0.928 | 0.992 | 1.951 |

## Sequence / side / window summary

| sequence | side | window | range | frames | pSlip mean | slip rate | H1 inst mean | H5 inst mean | Fmag mean |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| banana_1 | left | slip | 27-37 | 11 | 0.251 | 0.364 | 0.798 | 1.000 | 1.055 |
| banana_1 | left | stable | 40-70 | 31 | 0.001 | 0.000 | 0.473 | 1.000 | 1.002 |
| banana_1 | right | slip | 27-37 | 11 | 0.541 | 0.545 | 0.879 | 0.996 | 0.406 |
| banana_1 | right | stable | 40-70 | 31 | 0.089 | 0.065 | 0.627 | 0.801 | 0.387 |
| banana_2 | left | slip | 26-40 | 15 | 0.960 | 1.000 | 0.955 | 0.994 | 0.194 |
| banana_2 | left | stable | 60-90 | 31 | 0.720 | 0.806 | 0.889 | 0.990 | 0.246 |
| banana_2 | right | slip | 26-40 | 15 | 0.711 | 0.733 | 0.814 | 0.930 | 0.460 |
| banana_2 | right | stable | 60-90 | 31 | 0.294 | 0.226 | 0.782 | 0.997 | 0.383 |
| cucumber_1 | left | slip | 30-60 | 31 | 0.683 | 0.710 | 0.920 | 1.000 | 0.564 |
| cucumber_1 | right | slip | 30-60 | 31 | 0.686 | 0.774 | 0.895 | 0.996 | 0.421 |
| cucumber_2 | left | slip | 38-54 | 17 | 0.830 | 0.824 | 0.961 | 1.000 | 0.204 |
| cucumber_2 | right | slip | 38-54 | 17 | 0.997 | 1.000 | 0.998 | 0.999 | 0.239 |
| cucumber_3 | left | slip | 40-64 | 25 | 0.618 | 0.640 | 0.988 | 1.000 | 0.543 |
| cucumber_3 | left | stable | 80-110 | 31 | 0.025 | 0.000 | 0.915 | 1.000 | 0.550 |
| cucumber_3 | right | slip | 40-64 | 25 | 0.932 | 0.960 | 0.825 | 0.999 | 0.688 |
| cucumber_3 | right | stable | 80-110 | 31 | 0.039 | 0.000 | 0.305 | 0.929 | 0.716 |
| hammer_1 | left | slip | 50-80 | 31 | 1.000 | 1.000 | 1.000 | 1.000 | 2.379 |
| hammer_1 | right | slip | 50-80 | 31 | 1.000 | 1.000 | 1.000 | 1.000 | 2.324 |
| hammer_2 | left | slip | 34-54 | 21 | 0.892 | 1.000 | 1.000 | 1.000 | 2.357 |
| hammer_2 | left | stable | 90-116 | 27 | 0.723 | 1.000 | 1.000 | 1.000 | 2.409 |
| hammer_2 | right | slip | 34-54 | 21 | 0.789 | 0.857 | 1.000 | 1.000 | 2.356 |
| hammer_2 | right | stable | 90-116 | 27 | 0.419 | 0.370 | 1.000 | 1.000 | 2.382 |
| hammer_3 | left | slip | 40-70 | 31 | 1.000 | 1.000 | 1.000 | 1.000 | 2.377 |
| hammer_3 | right | slip | 40-70 | 31 | 1.000 | 1.000 | 1.000 | 1.000 | 2.362 |
| lemon_1 | left | slip | 28-39 | 12 | 0.356 | 0.417 | 0.906 | 1.000 | 0.587 |
| lemon_1 | left | stable | 50-80 | 31 | 0.001 | 0.000 | 0.589 | 0.921 | 0.641 |
| lemon_1 | right | slip | 28-39 | 12 | 0.417 | 0.417 | 0.748 | 1.000 | 0.640 |
| lemon_1 | right | stable | 50-80 | 31 | 0.007 | 0.000 | 0.686 | 1.000 | 0.644 |
| lemon_2 | left | slip | 28-40 | 13 | 0.539 | 0.615 | 1.000 | 1.000 | 2.280 |
| lemon_2 | left | stable | 60-90 | 31 | 0.000 | 0.000 | 1.000 | 1.000 | 2.336 |
| lemon_2 | right | slip | 28-40 | 13 | 0.588 | 0.615 | 1.000 | 1.000 | 2.343 |
| lemon_2 | right | stable | 60-90 | 31 | 0.000 | 0.000 | 1.000 | 1.000 | 2.396 |
| lemon_3 | left | slip | 27-37 | 11 | 0.663 | 0.636 | 1.000 | 1.000 | 2.345 |
| lemon_3 | left | stable | 70-100 | 31 | 0.000 | 0.000 | 1.000 | 1.000 | 2.373 |
| lemon_3 | right | slip | 27-37 | 11 | 0.543 | 0.545 | 1.000 | 1.000 | 2.070 |
| lemon_3 | right | stable | 70-100 | 31 | 0.000 | 0.000 | 1.000 | 1.000 | 2.109 |
| lemon_4 | left | slip | 31-41 | 11 | 0.519 | 0.545 | 1.000 | 1.000 | 2.347 |
| lemon_4 | left | stable | 70-100 | 31 | 0.000 | 0.000 | 1.000 | 1.000 | 2.363 |
| lemon_4 | right | slip | 31-41 | 11 | 0.710 | 0.727 | 1.000 | 1.000 | 2.155 |
| lemon_4 | right | stable | 70-100 | 31 | 0.001 | 0.000 | 1.000 | 1.000 | 2.223 |
| lemon_5 | left | slip | 31-42 | 12 | 0.515 | 0.500 | 1.000 | 1.000 | 2.302 |
| lemon_5 | left | stable | 70-100 | 31 | 0.000 | 0.000 | 1.000 | 1.000 | 2.233 |
| lemon_5 | right | slip | 31-42 | 12 | 0.861 | 0.917 | 1.000 | 1.000 | 2.225 |
| lemon_5 | right | stable | 70-100 | 31 | 0.000 | 0.000 | 1.000 | 1.000 | 2.191 |

## Automatic conclusion

- p_slip_current: stable_mean=0.122, slip_mean=0.773, gap=0.650, AUROC=0.891.
- p_instability_H1: stable_mean=0.846, slip_mean=0.954, gap=0.108, AUROC=0.565.
- p_instability_H3: stable_mean=0.967, slip_mean=0.995, gap=0.028, AUROC=0.555.
- p_instability_H5: stable_mean=0.980, slip_mean=0.997, gap=0.018, AUROC=0.568.

## Output files

- json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_model_test_report.json`
- markdown: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_model_test_report.md`
- per_frame_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_per_frame_predictions.csv`
- sequence_side_window_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_sequence_side_window_summary.csv`
- window_type_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_window_type_summary.csv`
- object_window_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_object_window_summary.csv`
- sequence_side_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_sequence_side_summary.csv`
- side_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_side_summary.csv`
- binary_metrics_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_binary_metrics.csv`
- stable_slip_metrics_png: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_215727/real_force_slip_stable_slip_metrics.png`
