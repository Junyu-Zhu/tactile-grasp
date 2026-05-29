# Future Head Horizon Fix Report

- generated_at: `2026-05-29T22:38:30`
- report_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_horizon_fix/20260529_223000`
- feature_run: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000`
- old_head: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth`
- real data policy: banana/cucumber/hammer/lemon data was used only for final zero-shot validation, not training, calibration, threshold fitting, or model selection.

## Label audit

- Existing cached labels are future-horizon labels generated from `labels[sample+1:sample+h+1]`; they are not direct copies of current slip labels.
- H1/H3/H5 therefore mean whether slip/instability occurs within the next 1/3/5 sampled steps in the original flat/sharp/sphere dataset.

## Original-data diagnostics and trained heads

| tag | mean AUPRC | mean AUROC | mean F1 | mean BalAcc | stable saturation>=0.99 | ckpt |
|---|---:|---:|---:|---:|---:|---|
| old_head_val_all | 0.9896 | 0.9935 | 0.9524 | 0.9608 | 0.0001 | `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth` |
| all_source_baseline_bce | 0.9564 | 0.9652 | 0.9120 | 0.9224 | 0.0001 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/all_source_baseline_bce/checkpoints/best.pth` |
| all_source_balanced_bce | 0.9572 | 0.9666 | 0.9178 | 0.9331 | 0.0006 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/all_source_balanced_bce/checkpoints/best.pth` |
| all_source_balanced_bce_smooth | 0.9591 | 0.9678 | 0.9155 | 0.9300 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/all_source_balanced_bce_smooth/checkpoints/best.pth` |
| flat_sharp_to_sphere_baseline_bce | 0.9409 | 0.9535 | 0.9093 | 0.9197 | 0.0001 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/flat_sharp_to_sphere_baseline_bce/checkpoints/best.pth` |
| flat_sharp_to_sphere_balanced_bce | 0.9395 | 0.9528 | 0.9062 | 0.9269 | 0.0001 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/flat_sharp_to_sphere_balanced_bce/checkpoints/best.pth` |
| flat_sharp_to_sphere_balanced_bce_smooth | 0.9410 | 0.9546 | 0.8961 | 0.9261 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/flat_sharp_to_sphere_balanced_bce_smooth/checkpoints/best.pth` |
| flat_sphere_to_sharp_baseline_bce | 0.9422 | 0.9473 | 0.9134 | 0.9245 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/flat_sphere_to_sharp_baseline_bce/checkpoints/best.pth` |
| flat_sphere_to_sharp_balanced_bce | 0.9443 | 0.9508 | 0.8463 | 0.8988 | 0.0003 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/flat_sphere_to_sharp_balanced_bce/checkpoints/best.pth` |
| flat_sphere_to_sharp_balanced_bce_smooth | 0.9466 | 0.9528 | 0.9117 | 0.9255 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/flat_sphere_to_sharp_balanced_bce_smooth/checkpoints/best.pth` |
| sharp_sphere_to_flat_baseline_bce | 0.9655 | 0.9593 | 0.9135 | 0.9245 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/sharp_sphere_to_flat_baseline_bce/checkpoints/best.pth` |
| sharp_sphere_to_flat_balanced_bce | 0.9714 | 0.9677 | 0.9173 | 0.9277 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/sharp_sphere_to_flat_balanced_bce/checkpoints/best.pth` |
| sharp_sphere_to_flat_balanced_bce_smooth | 0.9751 | 0.9732 | 0.7803 | 0.7926 | 0.0000 | `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/sharp_sphere_to_flat_balanced_bce_smooth/checkpoints/best.pth` |

- selected_best_head: `/vla1/zjy/sparsh_runs/force_slip_future_head_horizon_fix/20260529_223000/heads/sharp_sphere_to_flat_balanced_bce_smooth/checkpoints/best.pth`
- selected_reason: `selected on original flat/sharp/sphere validation/heldout metrics only; real data not used`

## Real zero-shot summary

| object | windows | accuracy | stable_acc | inst_acc | stable H1 sat old/new | H1 stable mean old/new | H1 inst mean old/new |
|---|---:|---:|---:|---:|---:|---:|---:|
| banana | 8 | 0.5000 | 0.0000 | 1.0000 | 0.0000/0.0000 | 0.6930/0.8617 | 0.8616/0.9337 |
| cucumber | 8 | 0.7500 | 0.0000 | 1.0000 | 0.0000/0.0000 | 0.6100/0.9311 | 0.9313/0.9719 |
| hammer | 8 | 0.7500 | 0.0000 | 1.0000 | 1.0000/1.0000 | 1.0000/0.9968 | 1.0000/0.9993 |
| lemon | 20 | 0.5000 | 0.0000 | 1.0000 | 0.8000/0.0000 | 0.9275/0.9197 | 0.9654/0.9662 |

## Files

- model_selection_summary: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_horizon_fix/20260529_223000/model_selection_summary.csv`
- real_window_table: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_horizon_fix/20260529_223000/real_zero_shot_old_vs_new_window_table.csv`
- real_object_summary: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/future_head_horizon_fix/20260529_223000/real_zero_shot_object_summary.csv`

## Original-val thresholds applied to real zero-shot windows

These thresholds were fitted only on the original flat/sharp/sphere validation features, not on real data.

| horizon | threshold | val BalAcc | val F1 | val AUROC | val AUPRC |
|---|---:|---:|---:|---:|---:|
| H1 | 0.848 | 0.9687 | 0.9574 | 0.9912 | 0.9856 |
| H3 | 0.728 | 0.9354 | 0.9103 | 0.9684 | 0.9580 |
| H5 | 0.784 | 0.9009 | 0.8738 | 0.9399 | 0.9286 |

### Real zero-shot object results with original-val thresholds

| object | rule | windows | Acc | stable Acc | instability Acc |
|---|---|---:|---:|---:|---:|
| banana | H1_pred | 8 | 0.7500 | 0.5000 | 1.0000 |
| banana | H1H3H5_majority_pred | 8 | 0.5000 | 0.0000 | 1.0000 |
| banana | H_any_pred | 8 | 0.5000 | 0.0000 | 1.0000 |
| cucumber | H1_pred | 8 | 0.7500 | 0.0000 | 1.0000 |
| cucumber | H1H3H5_majority_pred | 8 | 0.7500 | 0.0000 | 1.0000 |
| cucumber | H_any_pred | 8 | 0.7500 | 0.0000 | 1.0000 |
| hammer | H1_pred | 8 | 0.7500 | 0.0000 | 1.0000 |
| hammer | H1H3H5_majority_pred | 8 | 0.7500 | 0.0000 | 1.0000 |
| hammer | H_any_pred | 8 | 0.7500 | 0.0000 | 1.0000 |
| lemon | H1_pred | 20 | 0.6000 | 0.2000 | 1.0000 |
| lemon | H1H3H5_majority_pred | 20 | 0.5000 | 0.0000 | 1.0000 |
| lemon | H_any_pred | 20 | 0.5000 | 0.0000 | 1.0000 |

Interpretation: the new head improves real-data H-score ranking/AUROC versus the old H head, and reduces some H1 stable-window saturation (notably lemon), but the default H thresholds still over-predict instability on stable real windows. Every object has instability-window recall of 1.0 under H1 thresholding, but full stable-vs-slip window accuracy is below 0.8 for several objects. Therefore this is useful diagnostic progress, not yet a paper-ready replacement for pSlip-based real deployment detection.

## Per-object one-dataset success check

Using original-val H1 threshold and treating each sequence side as an independent real validation dataset, every object has at least one dataset/side with >=0.8 window accuracy. This is a weak deployment criterion; object-level aggregate stable-vs-slip accuracy remains lower than 0.8 for several objects.

| object | sequence | side | windows | H1 accuracy | H1 instability recall |
|---|---|---|---:|---:|---:|
| banana | banana_1 | left | 2 | 1.0000 | 1.0000 |
| cucumber | cucumber_1 | left | 1 | 1.0000 | 1.0000 |
| hammer | hammer_1 | left | 1 | 1.0000 | 1.0000 |
| lemon | lemon_1 | left | 2 | 1.0000 | 1.0000 |
