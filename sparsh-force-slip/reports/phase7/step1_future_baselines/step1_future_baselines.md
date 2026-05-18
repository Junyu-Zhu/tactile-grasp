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
