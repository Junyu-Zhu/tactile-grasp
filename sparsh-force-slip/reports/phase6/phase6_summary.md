# Phase6 Summary

- generated_at: `2026-05-18T02:07:43`
- raw_data_modified: `False`

## Reports

- phase6_1: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_1_multiseed_stability/phase6_1_multiseed_stability_report.md`
- phase6_2: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_2_heldout_sharp_generalization/phase6_2_heldout_sharp_generalization_report.md`
- phase6_3: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_3_frozen_vs_joint_ablation/phase6_3_frozen_vs_joint_ablation_report.md`
- phase6_4: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase6/phase6_4_early_warning_analysis/phase6_4_early_warning_analysis_report.md`

## Recommended paper method

Two-stage MAE decoupled force/slip perception + friction-aware tactile dynamics head (full+q), with multi-seed stability and two contact-geometry holdout stress tests.

## Main table candidates

- Phase6-1 multi-seed full dynamics vs full+q mean±std
- Phase6-2 flat+sphere→sharp heldout comparison
- Phase6-3 frozen two-stage vs lightweight joint-head ablation

## Supplement candidates

- Phase6-4 early-warning/failure-case trajectory figures
- Per-seed W&B logs and checkpoint paths

## Safe claims

- full+q is evaluated across three random seeds while keeping data split and cached features fixed.
- friction-aware dynamics is stress-tested on both sphere-heldout and sharp-heldout contact geometry splits.
- a lightweight joint-head proxy is included to justify the two-stage design without claiming full end-to-end world-model finetuning.

## Avoid overclaiming

- Do not claim real-robot grasp success or full physical world modeling.
- Do not describe tau/q as a measured friction coefficient; it is train-derived empirical conditioning.
- Do not present the lightweight joint ablation as exhaustive architecture search.

## Remaining risks

- All experiments remain on derived GSmini datasets, not live robot trials.
- Phase6-3 joint training is a frozen-feature proxy, not a full SPARSH backbone finetune.
