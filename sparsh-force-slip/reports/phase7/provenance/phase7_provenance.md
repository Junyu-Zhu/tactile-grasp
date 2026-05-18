# Phase7 Provenance

```json
{
  "generated_at": "2026-05-19T02:36:42",
  "repo": "/home/zjy/document/tactile-grasp",
  "branch": "sparsh-force-slip",
  "commit_before_final_audit_commit": "efe3c63e36af4a8def42c8487800249f7cf73d53",
  "status_short_before_final_audit_commit": "M sparsh-force-slip/reports/phase7/phase7_all_results.md\n M sparsh-force-slip/reports/phase7/phase7_summary.json\n M sparsh-force-slip/reports/phase7/phase7_summary.md\n M sparsh-force-slip/reports/phase7/step1_future_baselines/step1_future_baselines.json\n M sparsh-force-slip/reports/phase7/step1_future_baselines/step1_future_baselines.md\n M sparsh-force-slip/reports/phase7/step2_leave_one_geometry_out/step2_leave_one_geometry_out.json\n M sparsh-force-slip/reports/phase7/step2_leave_one_geometry_out/step2_leave_one_geometry_out.md\n M sparsh-force-slip/reports/phase7/step3_early_warning_threshold_sweep/step3_early_warning_threshold_sweep.json\n M sparsh-force-slip/reports/phase7/step3_early_warning_threshold_sweep/step3_early_warning_threshold_sweep.md\n M sparsh-force-slip/reports/phase7/step4_force_axis_decomposition/step4_force_axis_decomposition.json\n M sparsh-force-slip/reports/phase7/step4_force_axis_decomposition/step4_force_axis_decomposition.md\n M sparsh-force-slip/reports/phase7/step5_multiseed_stability/step5_multiseed_stability.json\n M sparsh-force-slip/reports/phase7/step5_multiseed_stability/step5_multiseed_stability.md\n M sparsh-force-slip/reports/phase7/step7_runtime_model_size/step7_runtime_model_size.json\n M sparsh-force-slip/reports/phase7/step7_runtime_model_size/step7_runtime_model_size.md\n M sparsh-force-slip/reports/training_results.md\n?? sparsh-force-slip/reports/phase7/step1_future_baselines/step1_future_baselines_full_metrics.csv\n?? sparsh-force-slip/reports/phase7/step2_leave_one_geometry_out/step2_sharp_sphere_to_flat_full_metrics.csv\n?? sparsh-force-slip/reports/phase7/step3_early_warning_threshold_sweep/step3_lead_time_cdf_all_thresholds.png\n?? sparsh-force-slip/reports/phase7/step4_force_axis_decomposition/step4_force_error_by_contact_geometry.png\n?? sparsh-force-slip/reports/phase7/step4_force_axis_decomposition/step4_force_error_distribution_by_dataset.csv\n?? sparsh-force-slip/reports/phase7/step4_force_axis_decomposition/step4_force_error_distribution_by_dataset.png\n?? sparsh-force-slip/reports/phase7/step5_multiseed_stability/naive_shared_quick5_multiseed.json\n?? sparsh-force-slip/reports/phase7/step5_multiseed_stability/naive_shared_quick5_multiseed.md\n?? sparsh-force-slip/scripts/phase7_add_required_plots.py\n?? sparsh-force-slip/scripts/phase7_add_step1_full_table.py\n?? sparsh-force-slip/scripts/phase7_add_step2_full_table.py\n?? sparsh-force-slip/scripts/phase7_collect_naive_shared_quick5.py\n?? sparsh-force-slip/scripts/phase7_runtime_measurements.py",
  "raw_dataset": "/vla1/zjy/tactile_datasets (read-only by protocol)",
  "raw_data_modified": false,
  "derived_dataset_root": "/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini",
  "feature_runs": {
    "phase4_decoupled": "/vla1/zjy/sparsh_runs/force_slip_phase4/phase4_3_reuse_p3_features_20260517_171500",
    "phase5_friction": "/vla1/zjy/sparsh_runs/force_slip_phase5/phase5_2_friction_features_20260518_0035",
    "phase6_friction": "/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130"
  },
  "phase7_checkpoint_paths": {
    "naive_shared_seed42_epoch5": "/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_gsmini_20260513_163448/mae_shared_multitask/checkpoints/epoch-0005.pth",
    "naive_shared_seed43_epoch5": "/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed43_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth",
    "naive_shared_seed44_epoch5": "/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed44_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth",
    "future_head_full_plus_q_seed42": "/vla1/zjy/sparsh_runs/force_slip_phase6/phase6_1_friction_features_20260518_0130/heads/phase6_1_full_plus_q_seed42_20260518_0130/checkpoints/best.pth"
  },
  "new_phase7_scripts": [
    "sparsh-force-slip/scripts/phase7_supplemental_experiments.py",
    "sparsh-force-slip/scripts/phase7_add_step1_full_table.py",
    "sparsh-force-slip/scripts/phase7_add_step2_full_table.py",
    "sparsh-force-slip/scripts/phase7_add_required_plots.py",
    "sparsh-force-slip/scripts/phase7_runtime_measurements.py",
    "sparsh-force-slip/scripts/phase7_collect_naive_shared_quick5.py"
  ],
  "commands_executed_or_reused": [
    "python sparsh-force-slip/scripts/phase7_supplemental_experiments.py summary",
    "python sparsh-force-slip/scripts/phase7_add_step1_full_table.py",
    "python sparsh-force-slip/scripts/phase7_add_step2_full_table.py",
    "python sparsh-force-slip/scripts/phase7_add_required_plots.py",
    "CUDA_VISIBLE_DEVICES=2 python sparsh-force-slip/scripts/phase7_runtime_measurements.py",
    "CUDA_VISIBLE_DEVICES=0 python sparsh-force-slip/scripts/phase2_b_multitask.py train --encoder mae --run-id phase7_naive_shared_quick_seed43_20260519 --max-epochs 5 --batch-size 100 --num-workers 2 --validation-frequency 5 --lambda-slip 1.0 --slip-horizon 0 --wandb-mode disabled --decoder-variant shared --seed 43",
    "CUDA_VISIBLE_DEVICES=1 python sparsh-force-slip/scripts/phase2_b_multitask.py train --encoder mae --run-id phase7_naive_shared_quick_seed44_20260519 --max-epochs 5 --batch-size 100 --num-workers 2 --validation-frequency 5 --lambda-slip 1.0 --slip-horizon 0 --wandb-mode disabled --decoder-variant shared --seed 44",
    "python sparsh-force-slip/scripts/phase7_collect_naive_shared_quick5.py"
  ],
  "notes": [
    "Phase7 reuses Phase1-Phase6 derived datasets, feature caches, checkpoints, eval caches and reports wherever available.",
    "Only minimal quick 5-epoch naive-shared seed43/44\u8865\u8dd1 was added to provide 3-seed stability evidence for Step5; W&B disabled.",
    "No automatic push/pull was performed."
  ]
}
```
