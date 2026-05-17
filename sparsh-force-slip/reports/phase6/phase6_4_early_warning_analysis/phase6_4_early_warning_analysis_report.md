# Phase6-4 Early-warning Analysis

- generated_at: `2026-05-18T02:07:39`
- selected_condition: `full_plus_q` seed `42`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/heads/phase6_1_full_plus_q_seed42_20260518_0130/checkpoints/best.pth`
- raw_data_modified: `False`

## Early-warning summary

| metric | value |
|---|---:|
| trajectory_count | 228 |
| slip_trajectories | 215 |
| detected_pre_slip_trajectories | 108 |
| late_warning_trajectories | 107 |
| missed_slip_trajectories | 0 |
| false_alarm_trajectories | 0 |
| early_warning_recall | 0.5023 |
| lead_time_to_slip_onset_steps_mean | 5.6019 |
| lead_time_to_slip_onset_steps_median | 1.0000 |

## Cases

| label | status | dataset | trajectory | onset | first warning | note | figure |
|---|---|---|---|---:|---:|---|---|
| success_early_warning | detected_pre | sharp_batch_1_val | 9 | 74 | 1 |  | `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_4_early_warning_analysis/figures/main_success_early_warning_sharp_batch_1_val_9.png` |
| late_warning | late | flat_batch_1_val | 17 | 39 | 39 |  | `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_4_early_warning_analysis/figures/main_late_warning_flat_batch_1_val_17.png` |
| missed_warning_absence_check | late | flat_batch_2_val | 15 | 14 | 17 | No true missed-warning trajectory at threshold 0.5; this is the lowest-risk slip trajectory. | `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_4_early_warning_analysis/figures/main_missed_warning_absence_check_flat_batch_2_val_15.png` |
| false_alarm_absence_check | true_negative | sphere_batch_3_val | 7 | None | None | No true false-alarm trajectory at threshold 0.5; this is the highest-risk no-slip trajectory. | `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_4_early_warning_analysis/figures/main_false_alarm_absence_check_sphere_batch_3_val_7.png` |
| heldout_contact_case | late | sharp_batch_1_val | 0 | 58 | 58 |  | `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_4_early_warning_analysis/figures/heldout_heldout_contact_case_sharp_batch_1_val_0.png` |
