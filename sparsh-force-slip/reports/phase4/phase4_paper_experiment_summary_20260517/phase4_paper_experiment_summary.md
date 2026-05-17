# Phase4 Paper Experiment Summary

- generated_at: `2026-05-17T20:32:22`
- branch: `sparsh-force-slip`
- raw_data_modified: `False`

## Recommended paper method

**MAE decoupled multitask + causal-dynamics future-stability head**. Use SPARSH-style separate force/slip as the primary current-task baseline, and frame the proposed method as a future-stability/tactile-dynamics extension rather than a claim that joint prediction dominates every current metric.

## Key results

| Evidence | Result |
|---|---|
| finding | Phase4-1: MAE route is stable across 3 seeds; world H1 future-slip F1 mean 0.9655±0.0013, H3 0.9548±0.0041, H5 0.9311±0.0078. |
| finding | Phase4-2: Separate late fusion H1 F1 0.9644, decoupled joint H1 F1 0.9626, decoupled+dynamics H1 F1 0.9676; dynamics gives the best H3/H5 future-slip F1. |
| finding | Phase4-3: z-only H1 F1 0.9450; force/slip H1 F1 0.9626; full dynamics H1 F1 0.9676, H5 F1 0.9353. |
| finding | Phase4-4: naive shared multitask worsens force RMSE by 12.88% vs separate; decoupled gives the best force RMSE among reused MAE current-task rows but lower current-slip F1 than separate. |

## Phase4-1 multi-seed summary

| metric | mean | std | n |
|---|---:|---:|---:|
| separate_force_rmse_mean_N | 0.0315 | 0.0009 | 3 |
| separate_slip_f1 | 0.9716 | 0.0033 | 3 |
| decoupled_force_rmse_mean_N | 0.0317 | 0.0012 | 3 |
| decoupled_slip_f1 | 0.9729 | 0.0067 | 3 |
| world_H1_future_slip_f1 | 0.9655 | 0.0013 | 3 |
| world_H1_future_slip_auroc | 0.9981 | 0.0006 | 3 |
| world_H1_future_slip_auprc | 0.9954 | 0.0007 | 3 |
| world_H3_future_slip_f1 | 0.9548 | 0.0041 | 3 |
| world_H5_future_slip_f1 | 0.9311 | 0.0078 | 3 |
| world_H1_stability_ece | 0.0051 | 0.0012 | 3 |

## Safe claims

- SPARSH-style separate probing is a strong current-task baseline and should remain in the main comparison.
- Naive force/slip multitask can cause force negative transfer, motivating decoder decoupling.
- Force/slip-conditioned dynamics improves future instability prediction over z-only and over current force/slip late fusion alone on the derived validation split.
- The proposed contribution is best framed as tactile future-stability/world-model extension rather than merely merging two decoder heads.

## Avoid overclaiming

- Do not claim decoupled multitask universally beats separate probing on current slip F1.
- Do not call the future-stability head a full physical world model; describe it as lightweight tactile dynamics/future-instability prediction.
- Do not claim grasp success prediction unless a separate grasp-outcome dataset/evaluation is added.

## Remaining risks

- All reported Phase4 metrics use the Phase1 derived validation split, not a separate held-out test set.
- Feature caches and heads are derived from the Gelsight-mini force/slip data distribution; cross-sensor or cross-object generalization still needs explicit testing.
- World-head labels are future slip/instability proxies, not direct grasp success labels.
- Some Phase4-1 base trainings overlapped temporally with later unrelated GPU jobs; no failures occurred, but wall-clock speed was affected.

## Suggested paper tables

- Main comparison: SPARSH separate baseline, naive shared multitask, decoupled multitask, decoupled + dynamics future-stability head
- Input ablation: z_t only, z_t + p_slip, z_t + force scalars, z_t + force + slip, z_t + force + slip + delta_force
- Stability: seed0, seed1, seed2, mean±std for H1/H3/H5 future-slip metrics

## Source reports

- phase4_1: `sparsh-force-slip/reports/phase4/phase4_1_20260517_144424/phase4_1_mae_multiseed_report.json`
- phase4_2: `sparsh-force-slip/reports/phase4/phase4_2_20260517_171500/phase4_2_late_fusion_vs_joint_report.json`
- phase4_3: `sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_world_model_input_ablation_report.json`
- phase4_4: `sparsh-force-slip/reports/phase4/phase4_4_architecture_ablation_20260517/phase4_4_architecture_ablation_report.json`
