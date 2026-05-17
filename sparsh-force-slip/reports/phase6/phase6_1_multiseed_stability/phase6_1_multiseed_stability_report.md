# Phase6-1 Multi-seed Stability

- generated_at: `2026-05-18T01:44:10`
- feature_run_id: `phase6_1_friction_features_20260518_0130`
- seeds: `[42, 43, 44]`
- raw_data_modified: `False`

## Per-seed metrics

| condition | seed | H1 F1 | H1 AUPRC | H1 ECE | H1 high recall | H1 lead mean | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_dynamics_baseline | 42 | 0.9666 | 0.9952 | 0.0052 | 0.9950 | 4.5856 | 0.9564 | 0.9936 | 0.9338 | 0.9805 |
| full_plus_q | 42 | 0.9668 | 0.9959 | 0.0053 | 0.9981 | 5.6019 | 0.9593 | 0.9945 | 0.9381 | 0.9815 |
| full_dynamics_baseline | 43 | 0.9643 | 0.9954 | 0.0064 | 0.9950 | 3.9674 | 0.9569 | 0.9940 | 0.9348 | 0.9815 |
| full_plus_q | 43 | 0.9646 | 0.9956 | 0.0085 | 0.9973 | 3.5556 | 0.9503 | 0.9941 | 0.9248 | 0.9814 |
| full_dynamics_baseline | 44 | 0.9634 | 0.9952 | 0.0074 | 0.9954 | 5.2151 | 0.9553 | 0.9936 | 0.9323 | 0.9809 |
| full_plus_q | 44 | 0.9644 | 0.9958 | 0.0084 | 0.9973 | 4.1548 | 0.9542 | 0.9942 | 0.9312 | 0.9810 |

## Mean ± std

| condition | metric | mean | std | n |
|---|---|---:|---:|---:|
| full_dynamics_baseline | H1_f1 | 0.9648 | 0.0017 | 3 |
| full_dynamics_baseline | H1_auroc | 0.9979 | 0.0001 | 3 |
| full_dynamics_baseline | H1_auprc | 0.9953 | 0.0001 | 3 |
| full_dynamics_baseline | H1_ece | 0.0063 | 0.0011 | 3 |
| full_dynamics_baseline | H1_high_recall | 0.9951 | 0.0002 | 3 |
| full_dynamics_baseline | H1_lead_mean | 4.5893 | 0.6238 | 3 |
| full_dynamics_baseline | H3_f1 | 0.9562 | 0.0008 | 3 |
| full_dynamics_baseline | H3_auprc | 0.9937 | 0.0002 | 3 |
| full_dynamics_baseline | H3_ece | 0.0081 | 0.0002 | 3 |
| full_dynamics_baseline | H5_f1 | 0.9337 | 0.0013 | 3 |
| full_dynamics_baseline | H5_auprc | 0.9810 | 0.0005 | 3 |
| full_dynamics_baseline | H5_ece | 0.0113 | 0.0016 | 3 |
| full_plus_q | H1_f1 | 0.9653 | 0.0014 | 3 |
| full_plus_q | H1_auroc | 0.9983 | 0.0001 | 3 |
| full_plus_q | H1_auprc | 0.9958 | 0.0001 | 3 |
| full_plus_q | H1_ece | 0.0074 | 0.0018 | 3 |
| full_plus_q | H1_high_recall | 0.9976 | 0.0004 | 3 |
| full_plus_q | H1_lead_mean | 4.4374 | 1.0520 | 3 |
| full_plus_q | H3_f1 | 0.9546 | 0.0045 | 3 |
| full_plus_q | H3_auprc | 0.9942 | 0.0002 | 3 |
| full_plus_q | H3_ece | 0.0106 | 0.0054 | 3 |
| full_plus_q | H5_f1 | 0.9313 | 0.0067 | 3 |
| full_plus_q | H5_auprc | 0.9813 | 0.0003 | 3 |
| full_plus_q | H5_ece | 0.0152 | 0.0103 | 3 |

## Interpretation

Across seeds [42, 43, 44], full+q H3 F1 mean=0.9546 versus baseline=0.9562; the result tests whether the Phase5 friction gain is initialization-stable.
