# Real Force-Slip Model Test Report (Sidecar Window)

- generated_at: `2026-05-29T20:11:06`
- dataset: `/vla1/zjy/tactile_dataset/real-force-slip-model-test-dataset`
- stage_i_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- stage_ii_future_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth`
- window_source: `sidecar eval_start/eval_end`
- frame_rate: `60 fps`; evaluated_frames: `2209`; sequence_sides: `28`
- preprocessing: first frame background subtraction = `True`, context stride = `5` frames, resized to `[320, 240]`.

## Interpretation notes

- This run uses manually annotated sidecar windows, so it avoids most no-contact and irrelevant release frames.
- The real dataset still has weak sequence-level labels, not per-frame force/slip ground truth; results are deployment diagnostics rather than calibrated accuracy/RMSE.
- Stage-II future-instability uses predicted force deltas on real data, so treat it as exploratory unless it agrees with slip/force trends.

## Outcome-level summary

| outcome | sides | frames | pSlip mean | pSlip max | slip rate | H1 inst mean | H5 inst mean | Fmag mean | Ft/Fn mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| slip_fail | 10 | 553 | 0.634 | 1.000 | 0.646 | 0.799 | 0.901 | 0.774 | 0.561 |
| slip_success | 6 | 578 | 0.553 | 1.000 | 0.596 | 0.883 | 0.965 | 1.651 | 1.084 |
| success | 12 | 1078 | 0.132 | 0.980 | 0.129 | 0.898 | 0.988 | 1.682 | 1.030 |

## Sequence-side summary

| sequence | side | outcome | window | frames | pSlip mean | slip rate | H1 inst mean | H5 inst mean | Fmag mean | first pSlip>=0.5 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| banana_slip_fail | left | slip_fail | 7-43 | 37 | 0.879 | 0.892 | 0.865 | 0.944 | 0.174 | 7 |
| banana_slip_fail | right | slip_fail | 4-43 | 40 | 0.648 | 0.675 | 0.567 | 0.747 | 0.424 | 4 |
| banana_success | left | success | 33-124 | 92 | 0.028 | 0.033 | 0.457 | 1.000 | 1.005 | 33 |
| banana_success | right | success | 32-119 | 88 | 0.143 | 0.091 | 0.666 | 0.866 | 0.380 | 32 |
| banana_success_2 | left | success | 23-117 | 95 | 0.719 | 0.789 | 0.894 | 0.991 | 0.236 | 23 |
| banana_success_2 | right | success | 42-121 | 80 | 0.254 | 0.188 | 0.758 | 0.997 | 0.384 | 42 |
| cucumber_slip_fail | left | slip_fail | 22-77 | 56 | 0.792 | 0.804 | 0.869 | 0.911 | 0.504 | 22 |
| cucumber_slip_fail | right | slip_fail | 22-74 | 53 | 0.770 | 0.830 | 0.839 | 0.922 | 0.409 | 22 |
| cucumber_slip_fail_2 | left | slip_fail | 27-52 | 26 | 0.767 | 0.808 | 0.787 | 0.841 | 0.227 | 27 |
| cucumber_slip_fail_2 | right | slip_fail | 25-51 | 27 | 0.852 | 0.815 | 0.883 | 0.945 | 0.260 | 25 |
| cucumber_slip_success | left | slip_success | 29-126 | 98 | 0.240 | 0.235 | 0.966 | 1.000 | 0.548 | 30 |
| cucumber_slip_success | right | slip_success | 22-126 | 105 | 0.382 | 0.371 | 0.464 | 0.903 | 0.704 | 22 |
| hammer_slip_fail | left | slip_fail | 43-130 | 88 | 0.589 | 0.591 | 0.931 | 0.932 | 2.315 | 43 |
| hammer_slip_fail | right | slip_fail | 41-128 | 88 | 0.698 | 0.693 | 0.943 | 0.951 | 2.198 | 41 |
| hammer_slip_success_2 | left | slip_success | 25-118 | 94 | 0.719 | 0.926 | 1.000 | 1.000 | 2.333 | 32 |
| hammer_slip_success_2 | right | slip_success | 17-116 | 100 | 0.581 | 0.650 | 1.000 | 1.000 | 2.291 | 21 |
| hammer_slip_success_3 | left | slip_success | 31-120 | 90 | 0.790 | 0.789 | 0.924 | 0.933 | 2.227 | 31 |
| hammer_slip_success_3 | right | slip_success | 28-118 | 91 | 0.604 | 0.604 | 0.945 | 0.955 | 1.800 | 28 |
| lemon_slip_fail | left | slip_fail | 22-90 | 69 | 0.163 | 0.174 | 0.655 | 0.896 | 0.609 | 22 |
| lemon_slip_fail | right | slip_fail | 19-87 | 69 | 0.178 | 0.174 | 0.653 | 0.927 | 0.622 | 19 |
| lemon_success | left | success | 39-127 | 89 | 0.039 | 0.045 | 1.000 | 1.000 | 2.331 | 39 |
| lemon_success | right | success | 34-124 | 91 | 0.055 | 0.055 | 1.000 | 1.000 | 2.390 | 34 |
| lemon_success_2 | left | success | 35-124 | 90 | 0.044 | 0.044 | 1.000 | 1.000 | 2.372 | 35 |
| lemon_success_2 | right | success | 35-126 | 92 | 0.065 | 0.065 | 1.000 | 1.000 | 2.098 | 35 |
| lemon_success_3 | left | success | 40-130 | 91 | 0.041 | 0.044 | 1.000 | 1.000 | 2.360 | 40 |
| lemon_success_3 | right | success | 38-128 | 91 | 0.059 | 0.055 | 1.000 | 1.000 | 2.213 | 38 |
| lemon_success_4 | left | success | 40-129 | 90 | 0.067 | 0.067 | 1.000 | 1.000 | 2.235 | 40 |
| lemon_success_4 | right | success | 38-126 | 89 | 0.066 | 0.067 | 1.000 | 1.000 | 2.186 | 38 |

## Automatic conclusion

- Sidecar-window pSlip mean: success=0.132, slip_fail=0.634.
- Sidecar-window slip-rate: success=0.129, slip_fail=0.646.
- Slip-success vs slip-fail pSlip mean: 0.553 vs 0.634.
- Mean force magnitude: success=1.682 N, slip_fail=0.774 N.

## Output files

- json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_200932/real_force_slip_model_test_report.json`
- markdown: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_200932/real_force_slip_model_test_report.md`
- per_frame_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_200932/real_force_slip_per_frame_predictions.csv`
- sequence_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_200932/real_force_slip_sequence_summary.csv`
- outcome_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_200932/real_force_slip_outcome_summary.csv`
- object_outcome_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_200932/real_force_slip_object_outcome_summary.csv`
- outcome_bars: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_200932/real_force_slip_outcome_bars.png`
