# Phase7 Summary

- generated_at: `2026-05-19T02:38:55`
- raw_data_modified: `False`

## step1_future_baselines

# Phase7 Step1 Future Baselines

- generated_at: `2026-05-19T01:30:49`
- raw_data_modified: `False`

| condition | n | H1 F1 | H1 AUPRC | H1 AUROC | H1 ECE | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_slip_persistence | 14328 | 0.9544 | 0.9825 | 0.9895 | 0.0118 | 0.9117 | 0.9492 | 0.8628 | 0.9148 |
| friction_q_threshold_score | 14328 | 0.9172 | 0.9657 | 0.9730 | 0.0370 | 0.8573 | 0.9137 | 0.8055 | 0.8776 |
| friction_ratio_hard_threshold | 14328 | 0.9172 | 0.8825 | 0.9235 | 0.0355 | 0.8573 | 0.8157 | 0.8055 | 0.7693 |
| slip_only_temporal_logreg | 14328 | 0.9486 | 0.9826 | 0.9895 | 0.0184 | 0.9061 | 0.9489 | 0.8619 | 0.9143 |
| force_only_logreg | 14328 | 0.9552 | 0.9921 | 0.9964 | 0.0170 | 0.9287 | 0.9874 | 0.9104 | 0.9734 |
| full_dynamics_existing | 14328 | 0.9666 | 0.9952 | 0.9979 | 0.0052 | 0.9564 | 0.9936 | 0.9338 | 0.9805 |
| full_dynamics_plus_q_existing | 14328 | 0.9668 | 0.9959 | 0.9983 | 0.0053 | 0.9593 | 0.9945 | 0.9381 | 0.9815 |

## Full H1/H3/H5 metric table

| condition | H1 F1 | H1 AUPRC | H1 AUROC | H1 ECE | H3 F1 | H3 AUPRC | H3 AUROC | H3 ECE | H5 F1 | H5 AUPRC | H5 AUROC | H5 ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_slip_persistence | 0.9544 | 0.9825 | 0.9895 | 0.0118 | 0.9117 | 0.9492 | 0.9629 | 0.0297 | 0.8628 | 0.9148 | 0.9292 | 0.0586 |
| friction_q_threshold_score | 0.9172 | 0.9657 | 0.9730 | 0.0370 | 0.8573 | 0.9137 | 0.9258 | 0.0669 | 0.8055 | 0.8776 | 0.8903 | 0.0964 |
| friction_ratio_hard_threshold | 0.9172 | 0.8825 | 0.9235 | 0.0355 | 0.8573 | 0.8157 | 0.8751 | 0.0655 | 0.8055 | 0.7693 | 0.8372 | 0.0950 |
| slip_only_temporal_logreg | 0.9486 | 0.9826 | 0.9895 | 0.0184 | 0.9061 | 0.9489 | 0.9624 | 0.0522 | 0.8619 | 0.9143 | 0.9285 | 0.0707 |
| force_only_logreg | 0.9552 | 0.9921 | 0.9964 | 0.0170 | 0.9287 | 0.9874 | 0.9926 | 0.0245 | 0.9104 | 0.9734 | 0.9817 | 0.0344 |
| full_dynamics_existing | 0.9666 | 0.9952 | 0.9979 | 0.0052 | 0.9564 | 0.9936 | 0.9969 | 0.0079 | 0.9338 | 0.9805 | 0.9864 | 0.0130 |
| full_dynamics_plus_q_existing | 0.9668 | 0.9959 | 0.9983 | 0.0053 | 0.9593 | 0.9945 | 0.9975 | 0.0054 | 0.9381 | 0.9815 | 0.9872 | 0.0078 |

- CSV: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step1_future_baselines/step1_future_baselines_full_metrics.csv`


## step2_leave_one_geometry_out

# Phase7 Step2 Leave-One-Contact-Geometry-Out

- raw_data_modified: `False`

## New split: train sharp+sphere -> test flat

| condition | current force RMSE | current slip F1 | H1 F1 | H3 F1 | H5 F1 | H5 AUPRC | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| separate_late_fusion | 0.0339 | 0.9498 | 0.9569 | 0.9296 | 0.8904 | 0.9560 | 34 |
| decoupled_static | 0.0350 | 0.9338 | 0.9582 | 0.9160 | 0.8659 | 0.9556 | 34 |
| decoupled_dynamics | 0.0350 | 0.9338 | 0.9539 | 0.9285 | 0.9120 | 0.9820 | 40 |
| decoupled_dynamics_friction | 0.0350 | 0.9338 | 0.9596 | 0.9415 | 0.9183 | 0.9834 | 37 |

## Combined three held-out splits

| split | condition | H3 F1 | H5 F1 | H5 AUPRC |
|---|---|---:|---:|---:|
| flat+sharp_to_sphere | separate_late_fusion | 0.9089 | 0.8525 | 0.9145 |
| flat+sharp_to_sphere | decoupled_static | 0.9185 | 0.8636 | 0.9164 |
| flat+sharp_to_sphere | decoupled_dynamics | 0.8922 | 0.7392 | 0.9543 |
| flat+sharp_to_sphere | decoupled_dynamics_friction | 0.9400 | 0.9169 | 0.9573 |
| flat+sphere_to_sharp | separate_late_fusion | 0.9031 | 0.8614 | 0.9174 |
| flat+sphere_to_sharp | decoupled_static | 0.8914 | 0.8494 | 0.9211 |
| flat+sphere_to_sharp | decoupled_dynamics | 0.9397 | 0.8675 | 0.9661 |
| flat+sphere_to_sharp | decoupled_dynamics_friction | 0.9447 | 0.8756 | 0.9665 |
| sharp+sphere_to_flat | separate_late_fusion | 0.9296 | 0.8904 | 0.9560 |
| sharp+sphere_to_flat | decoupled_static | 0.9160 | 0.8659 | 0.9556 |
| sharp+sphere_to_flat | decoupled_dynamics | 0.9285 | 0.9120 | 0.9820 |
| sharp+sphere_to_flat | decoupled_dynamics_friction | 0.9415 | 0.9183 | 0.9834 |

## New split full H1/H3/H5 metrics

| condition | current force RMSE | current slip F1 | H1 F1 | H1 AUPRC | H1 AUROC | H3 F1 | H3 AUPRC | H3 AUROC | H5 F1 | H5 AUPRC | H5 AUROC | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separate_late_fusion | 0.0339 | 0.9498 | 0.9569 | 0.9872 | 0.9868 | 0.9296 | 0.9725 | 0.9691 | 0.8904 | 0.9560 | 0.9456 | 34 |
| decoupled_static | 0.0350 | 0.9338 | 0.9582 | 0.9869 | 0.9887 | 0.9160 | 0.9715 | 0.9728 | 0.8659 | 0.9556 | 0.9522 | 34 |
| decoupled_dynamics | 0.0350 | 0.9338 | 0.9539 | 0.9938 | 0.9957 | 0.9285 | 0.9916 | 0.9937 | 0.9120 | 0.9820 | 0.9836 | 40 |
| decoupled_dynamics_friction | 0.0350 | 0.9338 | 0.9596 | 0.9966 | 0.9979 | 0.9415 | 0.9933 | 0.9953 | 0.9183 | 0.9834 | 0.9851 | 37 |

- CSV: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step2_leave_one_geometry_out/step2_sharp_sphere_to_flat_full_metrics.csv`


## step3_early_warning_threshold_sweep

# Phase7 Step3 Early-Warning Threshold Sweep

| threshold | early recall | false alarm rate | late ratio | missed ratio | mean lead | median lead |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.6698 | 0.1538 | 0.3302 | 0.0000 | 6.7292 | 1.5000 |
| 0.4 | 0.5814 | 0.1538 | 0.4186 | 0.0000 | 5.6400 | 1.0000 |
| 0.5 | 0.5023 | 0.0000 | 0.4977 | 0.0000 | 5.6019 | 1.0000 |
| 0.6 | 0.4326 | 0.0000 | 0.5674 | 0.0000 | 4.2796 | 1.0000 |
| 0.7 | 0.3860 | 0.0000 | 0.6140 | 0.0000 | 1.8434 | 1.0000 |
| 0.8 | 0.3256 | 0.0000 | 0.6744 | 0.0000 | 1.1714 | 1.0000 |

## Figures

- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step3_early_warning_threshold_sweep/step3_lead_time_cdf_all_thresholds.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step3_early_warning_threshold_sweep/step3_lead_time_hist_threshold_0p5.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step3_early_warning_threshold_sweep/step3_threshold_early_warning_recall.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step3_early_warning_threshold_sweep/step3_threshold_false_alarm_rate.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step3_early_warning_threshold_sweep/step3_threshold_lead_time_to_slip_onset_steps_mean.png`


## step4_force_axis_decomposition

# Phase7 Step4 Force Per-Axis / Physical Decomposition

| condition | Fx | Fy | Fz | Fn | Ft | Fmag | mean | slip F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| separate_baseline | 0.0369 | 0.0345 | 0.0253 | 0.0253 | 0.0380 | 0.0346 | 0.0322 | 0.9731 |
| naive_shared_multitask | 0.0402 | 0.0333 | 0.0357 | 0.0357 | 0.0359 | 0.0432 | 0.0364 | 0.9784 |
| partially_shared_lambda_0.25 | 0.0331 | 0.0323 | 0.0278 | 0.0278 | 0.0365 | 0.0378 | 0.0311 | 0.9771 |
| consistency_decoder | 0.0346 | 0.0324 | 0.0336 | 0.0336 | 0.0335 | 0.0405 | 0.0335 | 0.9819 |
| decoupled_multitask | 0.0293 | 0.0324 | 0.0295 | 0.0295 | 0.0363 | 0.0397 | 0.0304 | 0.9652 |
| joint_lightweight_force_slip_future | 0.0707 | 0.0828 | 0.0665 | 0.0665 | 0.0790 | 0.0849 | 0.0734 | 0.9816 |

## Error distribution across validation datasets

- CSV: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step4_force_axis_decomposition/step4_force_error_distribution_by_dataset.csv`
- Boxplot: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step4_force_axis_decomposition/step4_force_error_distribution_by_dataset.png`
- Contact-geometry grouped plot: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step4_force_axis_decomposition/step4_force_error_by_contact_geometry.png`
- Joint lightweight Fn/Ft/Fmag metrics were recomputed from its saved checkpoint and Phase6 cached validation features.


## step5_multiseed_stability

# Phase7 Step5 Main Results Multi-Seed Stability

| condition | source | force RMSE mean±std | slip F1 mean±std | H3 F1 mean±std | H5 F1 mean±std | H5 AUPRC mean±std | note |
|---|---|---:|---:|---:|---:|---:|---|
| separate_baseline | phase4_1 n=3 | 0.0315±0.0009 (n=3) | 0.9716±0.0033 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a |  |
| decoupled_multitask | phase4_1 n=3 | 0.0317±0.0012 (n=3) | 0.9729±0.0067 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a |  |
| naive_shared_multitask | phase4_4 single seed only | 0.0364±0.0000 (n=1) | 0.9784±0.0000 (n=1) | n/a±n/a | n/a±n/a | n/a±n/a | only seed42 available; missing seeds are recorded, not fabricated |
| full_dynamics_future_head | phase6_1 n=3 | n/a±n/a (n=) | n/a±n/a (n=) | 0.9562±0.0008 | 0.9337±0.0013 | 0.9810±0.0005 |  |
| full_dynamics_plus_q | phase6_1 n=3 | n/a±n/a (n=) | n/a±n/a (n=) | 0.9546±0.0045 | 0.9313±0.0067 | 0.9813±0.0003 |  |
| naive_shared_multitask_quick5_supplemental | Phase7 matched 5-epoch checkpoints, seeds 42/43/44 | 0.0554±0.0003 (n=3) | 0.9747±0.0024 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a | 3-seed quick stability check; not a full 51-epoch replacement for the original seed42 reference. |

## Naive shared quick 5-epoch supplemental seeds

This supplemental row closes the naive-shared 3-seed stability gap with matched 5-epoch checkpoints. It is useful for seed sensitivity, but the 5-epoch statistics are not directly comparable to the full 51-epoch single-seed reference row.

| seed | run_id | epoch | force RMSE | slip F1 | slip accuracy | checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 42 | phase2_b_gsmini_20260513_163448 | 5 | 0.0555 | 0.9720 | 0.9861 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_gsmini_20260513_163448/mae_shared_multitask/checkpoints/epoch-0005.pth` |
| 43 | phase7_naive_shared_quick_seed43_20260519 | 5 | 0.0550 | 0.9742 | 0.9872 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed43_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth` |
| 44 | phase7_naive_shared_quick_seed44_20260519 | 5 | 0.0557 | 0.9779 | 0.9891 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed44_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth` |

## Best / worst seeds

| metric | best seed | best value | worst seed | worst value | lower is better |
|---|---|---:|---|---:|---|
| separate_force_rmse | seed1 | 0.0305 | seed0 | 0.0322 | True |
| separate_slip_f1 | seed1 | 0.9738 | seed2 | 0.9678 | False |
| decoupled_force_rmse | seed0 | 0.0304 | seed2 | 0.0327 | True |
| decoupled_slip_f1 | seed2 | 0.9775 | seed0 | 0.9652 | False |
| full_dynamics_baseline_H3_f1 | 43 | 0.9569 | 44 | 0.9553 | False |
| full_dynamics_baseline_H5_f1 | 43 | 0.9348 | 44 | 0.9323 | False |
| full_plus_q_H3_f1 | 42 | 0.9593 | 43 | 0.9503 | False |
| full_plus_q_H5_f1 | 42 | 0.9381 | 43 | 0.9248 | False |
| naive_shared_quick5_force_rmse | 43 | 0.0550 | 44 | 0.0557 | True |
| naive_shared_quick5_slip_f1 | 44 | 0.9779 | 42 | 0.9720 | False |


## step6_case_visualization

# Phase7 Step6 Failure / Success Case Visualization

- copied_figures: `10`

## Figures
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/heldout_heldout_contact_case_sharp_batch_1_val_0.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/main_false_alarm_absence_check_sphere_batch_3_val_7.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/main_late_warning_flat_batch_1_val_17.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/main_missed_warning_absence_check_flat_batch_2_val_15.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/main_success_early_warning_sharp_batch_1_val_9.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/high_force_no_current_slip_sphere_batch_4_val_8.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/sharp_contact_case_sharp_batch_1_val_0.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/slip_with_low_friction_ratio_failure_sharp_batch_1_val_14.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/sphere_contact_case_sphere_batch_1_val_0.png`
- `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase7/step6_case_visualization/figures/success_early_warning_flat_batch_1_val_2.png`

## CSV manifest

- `step6_case_visualization.csv`


## step7_runtime_model_size

# Phase7 Step7 Runtime / Model Size / Inference Cost

## Parameter counts

| model | parameters |
|---|---:|
| separate_force_head | 7236867 |
| separate_slip_head | 7386437 |
| naive_shared_total | 93716357 |
| naive_shared_downstream_excl_encoder | 7460357 |
| decoupled_total | 101175749 |
| decoupled_downstream_excl_encoder | 14919749 |
| future_head_full_plus_q | 531475 |

## Measured current force/slip inference latency

| model | device | batch | encoder feature extraction ms/batch | force/slip heads ms/batch | total ms/batch | total ms/sample | trainable params |
|---|---|---:|---:|---:|---:|---:|---:|
| naive_shared_mae | cuda:0 | 64 | 123.477918 | 1.768546 | 134.942540 | 2.108477 | 7460357 |
| decoupled_mae | cuda:0 | 64 | 120.939902 | 4.342638 | 137.251879 | 2.144561 | 14919749 |

## Future head cached-feature latency
- device: `cuda:0`
- batch_size: `512`
- ms_per_batch: `0.098856`
- ms_per_sample: `0.000193`
- trainable_params: `531475`

## Separate probing deployment estimate

Separate force/slip probing checkpoints are independent downstream tasks. For a naive real-time deployment without encoder-feature reuse, the estimate is two frozen-encoder passes plus head cost; with feature reuse, it reduces to one encoder pass plus two lightweight heads.

- two-pass total estimate: `248.724382` ms/batch64
- shared-encoder lower-bound estimate: `127.015010` ms/batch64

## GPU memory snapshot

- `0, NVIDIA GeForce RTX 4090, 0 %, 7502 MiB, 49140 MiB`
- `1, NVIDIA GeForce RTX 4090, 0 %, 7462 MiB, 49140 MiB`
- `2, NVIDIA GeForce RTX 4090, 75 %, 6752 MiB, 49140 MiB`
- `3, NVIDIA GeForce RTX 4090, 73 %, 4875 MiB, 49140 MiB`

## Notes
- Encoder feature extraction time is measured as the frozen Sparsh MAE encoder forward on a validation batch; no training data were modified.
- Force/slip head time is measured by running the decoder on the cached encoder tokens from the same batch.
- Separate probing latency is reported as a deployment estimate because the force and slip probes are independent checkpoints rather than one combined callable model.
- Measurements were taken on a shared server, so absolute latency can vary with concurrent jobs; relative head/encoder scale is the intended evidence.

