# Phase4-4 MAE Architecture Ablation

- generated_at: `2026-05-17T14:55:03`
- backbone: `mae`
- split: Phase1 derived all-source train/val split reused from Phase2/Phase3
- raw_data_modified: `False`

## Current-task architecture table

| condition | force RMSE ↓ | Δ force vs separate | slip F1 ↑ | Δ slip F1 | slip acc ↑ | H1 future F1 | H3 future F1 | H5 future F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| separate_baseline | 0.0322 | 0.00% | 0.9731 | 0.00 pp | 0.9866 | n/a | n/a | n/a |
| naive_shared_multitask | 0.0364 | 12.88% | 0.9784 | 0.53 pp | 0.9893 | n/a | n/a | n/a |
| partially_shared_lambda_0.25 | 0.0311 | -3.61% | 0.9771 | 0.40 pp | 0.9887 | n/a | n/a | n/a |
| partially_shared_lambda_0.50 | 0.0319 | -0.98% | 0.9799 | 0.68 pp | 0.9901 | n/a | n/a | n/a |
| consistency_decoder | 0.0335 | 3.99% | 0.9819 | 0.88 pp | 0.9911 | n/a | n/a | n/a |
| decoupled_multitask | 0.0304 | -5.75% | 0.9652 | -0.79 pp | 0.9825 | n/a | n/a | n/a |
| decoupled_plus_world_model | 0.0304 | -5.75% | 0.9652 | -0.79 pp | 0.9825 | 0.9645 | 0.9506 | 0.9222 |

## Interpretation

- Naive shared multitask improves slip F1 but increases MAE force RMSE by about 12.9%, showing negative transfer for force regression.
- Partially shared decoders mitigate this conflict: lambda=0.25 improves force RMSE versus separate while keeping slip F1 above separate; lambda=0.50 favors slip slightly more.
- The consistency decoder reaches the highest current-slip F1 among reused MAE rows but worsens force RMSE versus separate, so it is a useful ablation rather than the safest main method.
- Decoupled multitask gives the best force RMSE among these rows, but current slip F1 is lower than separate; therefore the paper should not claim decoupled dominates every current-task metric.
- Decoupled + world model keeps decoupled current-task metrics and adds multi-horizon future instability prediction, which is the clearest architecture-level novelty beyond SPARSH-style separate probing.

## Source artifacts

- separate: `sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/eval_cache/a_mae_allsource_val.json`
- shared: `sparsh-force-slip/reports/phase2/phase2_b_gsmini_20260513_163448/phase2_b_diagnostic_report.json`
- partially_shared_025: `sparsh-force-slip/reports/phase2/phase2_b_ps_lam025_gsmini_20260514_063052/eval_cache/b_mae_partially_shared_allsource_val.json`
- partially_shared_050: `sparsh-force-slip/reports/phase2/phase2_b_ps_lam050_gsmini_20260514_063052/eval_cache/b_mae_partially_shared_allsource_val.json`
- consistency: `sparsh-force-slip/reports/phase2/phase2_c_consistency_gsmini_20260514_161845/eval_cache/c_mae_consistency_phase2_c_consistency_gsmini_20260514_161845_allsource_val.json`
- decoupled: `sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/eval_cache/decoupled_mae_phase3_1_decoupled_gsmini_20260516_154730_allsource_val.json`
- world_model: `sparsh-force-slip/reports/phase3/phase3_2_20260517_014013/phase3_world_model_summary.json`

## Paper-use recommendation

- Use `separate_baseline` as the SPARSH-style baseline, `naive_shared_multitask` as the negative-transfer ablation, `partially_shared` as mitigation ablation, and `decoupled_plus_world_model` as the proposed route for future stability.
- Phrase the claim as: decoupling protects force regression and enables force-slip-conditioned future-stability prediction; it does not universally outperform separate probing on current slip F1.
