# Real Force-Slip Model Test Report

- generated_at: `2026-05-29T03:30:34`
- dataset: `/vla1/zjy/tactile_dataset/real-force-slip-model-test-dataset`
- stage_i_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- stage_ii_future_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth`
- frame_rate: `60 fps`; input_resolution: `640x480`; evaluated_frames: `4238`
- preprocessing: first frame background subtraction = `True`, context stride = `5` frames (~0.083s), resized to `[320, 240]`.

## Important interpretation notes

- The real dataset has sequence-level outcome names but no per-frame force/slip labels, so this is a weak-label deployment sanity check, not a calibrated accuracy/RMSE benchmark.
- `hard_fail`/`*_fail` may include no contact; a contact-level slip model can output low slip for no-touch frames, so hard failures should be interpreted as out-of-distribution/no-contact cases rather than ordinary slip failures.
- The future head was trained with force-delta inputs; here the unavailable real force delta is approximated by causal differences of Stage-I predicted force.

## Outcome-level summary

| outcome | seq-sides | pSlip mean | pSlip max | pSlip top10 | slip rate | H1 inst mean | H1 inst max | H5 inst mean | Fmag mean | Ft/Fn mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_fail | 6 | 0.890 | 1.000 | 1.000 | 0.922 | 0.906 | 0.994 | 0.949 | 0.176 | 0.258 |
| slip_fail | 6 | 0.817 | 1.000 | 1.000 | 0.824 | 0.883 | 1.000 | 0.946 | 0.287 | 0.311 |
| slip_success | 8 | 0.727 | 1.000 | 1.000 | 0.745 | 0.874 | 1.000 | 0.930 | 1.114 | 0.794 |
| success | 8 | 0.378 | 1.000 | 1.000 | 0.380 | 0.956 | 1.000 | 0.959 | 1.722 | 1.166 |

## Object-level summary

| object | outcome | seq-sides | pSlip mean | pSlip top10 | H1 inst mean | H5 inst mean | Fmag mean |
|---|---|---:|---:|---:|---:|---:|---:|
| banana | hard_fail | 4 | 0.853 | 1.000 | 0.888 | 0.945 | 0.176 |
| banana | slip_fail | 2 | 0.981 | 1.000 | 0.967 | 0.981 | 0.167 |
| cucumber | hard_fail | 2 | 0.965 | 1.000 | 0.941 | 0.956 | 0.175 |
| cucumber | slip_fail | 2 | 0.917 | 1.000 | 0.917 | 0.943 | 0.281 |
| cucumber | slip_success | 2 | 0.484 | 1.000 | 0.738 | 0.920 | 0.527 |
| hammer | slip_success | 6 | 0.808 | 1.000 | 0.920 | 0.933 | 1.309 |
| lemon | slip_fail | 2 | 0.552 | 1.000 | 0.765 | 0.912 | 0.411 |
| lemon | success | 8 | 0.378 | 1.000 | 0.956 | 0.959 | 1.722 |

## Sequence-side summary

| sequence | side | outcome | frames | pSlip mean | pSlip max | pSlip top10 | slip rate | H1 inst mean | H5 inst mean | Fmag mean | first slip frame |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| banana_hard_fail | left | hard_fail | 179 | 1.000 | 1.000 | 1.000 | 1.000 | 0.946 | 0.956 | 0.156 | 5 |
| banana_hard_fail | right | hard_fail | 179 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 | 0.954 | 0.153 | 5 |
| banana_hard_fail_2 | left | hard_fail | 155 | 0.836 | 1.000 | 1.000 | 0.923 | 0.900 | 0.952 | 0.218 | 5 |
| banana_hard_fail_2 | right | hard_fail | 155 | 0.576 | 1.000 | 1.000 | 0.677 | 0.755 | 0.918 | 0.177 | 5 |
| banana_slip_fail | left | slip_fail | 112 | 0.962 | 1.000 | 1.000 | 0.973 | 0.944 | 0.966 | 0.162 | 5 |
| banana_slip_fail | right | slip_fail | 112 | 1.000 | 1.000 | 1.000 | 1.000 | 0.990 | 0.996 | 0.173 | 5 |
| cucumber_fail | left | hard_fail | 143 | 0.958 | 1.000 | 1.000 | 0.965 | 0.935 | 0.949 | 0.169 | 5 |
| cucumber_fail | right | hard_fail | 143 | 0.972 | 1.000 | 1.000 | 0.965 | 0.946 | 0.963 | 0.182 | 5 |
| cucumber_slip_fail | left | slip_fail | 143 | 0.919 | 1.000 | 1.000 | 0.923 | 0.901 | 0.922 | 0.291 | 5 |
| cucumber_slip_fail | right | slip_fail | 143 | 0.915 | 1.000 | 1.000 | 0.937 | 0.932 | 0.965 | 0.272 | 5 |
| cucumber_slip_success | left | slip_success | 138 | 0.439 | 1.000 | 1.000 | 0.435 | 0.925 | 0.954 | 0.464 | 5 |
| cucumber_slip_success | right | slip_success | 138 | 0.530 | 1.000 | 1.000 | 0.522 | 0.551 | 0.887 | 0.590 | 5 |
| hammer_slip_success | left | slip_success | 193 | 0.786 | 1.000 | 1.000 | 0.788 | 0.915 | 0.929 | 1.215 | 5 |
| hammer_slip_success | right | slip_success | 193 | 0.841 | 1.000 | 1.000 | 0.839 | 0.904 | 0.919 | 1.136 | 5 |
| hammer_slip_success_2 | left | slip_success | 174 | 0.831 | 1.000 | 1.000 | 0.943 | 0.926 | 0.937 | 1.420 | 5 |
| hammer_slip_success_2 | right | slip_success | 174 | 0.759 | 1.000 | 1.000 | 0.799 | 0.953 | 0.963 | 1.471 | 5 |
| hammer_slip_success_3 | left | slip_success | 151 | 0.871 | 1.000 | 1.000 | 0.874 | 0.889 | 0.910 | 1.449 | 5 |
| hammer_slip_success_3 | right | slip_success | 151 | 0.759 | 1.000 | 1.000 | 0.762 | 0.931 | 0.941 | 1.165 | 5 |
| lemon_slip_fail | left | slip_fail | 129 | 0.552 | 1.000 | 1.000 | 0.558 | 0.774 | 0.913 | 0.412 | 5 |
| lemon_slip_fail | right | slip_fail | 129 | 0.553 | 1.000 | 1.000 | 0.550 | 0.757 | 0.911 | 0.411 | 5 |
| lemon_success | left | success | 160 | 0.409 | 1.000 | 1.000 | 0.419 | 0.964 | 0.965 | 1.675 | 5 |
| lemon_success | right | success | 160 | 0.394 | 1.000 | 1.000 | 0.394 | 0.942 | 0.956 | 1.709 | 5 |
| lemon_success_2 | left | success | 152 | 0.379 | 1.000 | 1.000 | 0.375 | 0.956 | 0.956 | 1.780 | 5 |
| lemon_success_2 | right | success | 152 | 0.381 | 1.000 | 1.000 | 0.382 | 0.964 | 0.964 | 1.593 | 5 |
| lemon_success_3 | left | success | 164 | 0.432 | 1.000 | 1.000 | 0.433 | 0.956 | 0.957 | 1.631 | 5 |
| lemon_success_3 | right | success | 164 | 0.445 | 1.000 | 1.000 | 0.445 | 0.961 | 0.963 | 1.518 | 5 |
| lemon_success_4 | left | success | 126 | 0.279 | 1.000 | 1.000 | 0.278 | 0.954 | 0.955 | 1.937 | 5 |
| lemon_success_4 | right | success | 126 | 0.306 | 1.000 | 1.000 | 0.317 | 0.951 | 0.956 | 1.932 | 5 |

## Deployment sanity-check conclusion

- Current slip branch shows useful weak-label ordering by sequence mean: stable `success` has much lower mean pSlip (0.378) than `slip_success` (0.727), `slip_fail` (0.817), and `hard_fail` (0.890). For real sequences, mean/slip-rate are more useful than max/top-10%, because all classes contain short saturated high-risk frames.
- Predicted force magnitude separates contact strength: hard/no-contact failures are low (~0.176 N mean Fmag), while stable lemon success is high (~1.722 N). This suggests the force head is responding to contact intensity, although it is not a calibrated force benchmark without real force labels.
- The Stage-II future-instability head is not reliable as a standalone real-world score here: it assigns high instability to stable success as well. Likely causes are real-domain force-ratio shift and the fact that real force deltas are approximated from predicted force. For the current real experiment, use Stage-I pSlip + force/contact trends as the main evidence and treat future-instability scores as exploratory.
- Dataset limitation: stable success examples are currently only lemon, while other objects mostly cover slip/fail outcomes, so object identity and outcome are confounded. For stronger paper evidence, add same-object success/slip-fail pairs for banana, cucumber, and hammer.

## Output files

- json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_032848/real_force_slip_model_test_report.json`
- markdown: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_032848/real_force_slip_model_test_report.md`
- per_frame_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_032848/real_force_slip_per_frame_predictions.csv`
- sequence_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_032848/real_force_slip_sequence_summary.csv`
- outcome_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_032848/real_force_slip_outcome_summary.csv`
- object_outcome_csv: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_032848/real_force_slip_object_outcome_summary.csv`
- outcome_bars: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/real_force_slip_model_test/20260529_032848/real_force_slip_outcome_bars.png`
