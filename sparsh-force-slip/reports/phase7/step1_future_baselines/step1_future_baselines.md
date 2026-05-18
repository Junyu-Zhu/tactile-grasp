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
