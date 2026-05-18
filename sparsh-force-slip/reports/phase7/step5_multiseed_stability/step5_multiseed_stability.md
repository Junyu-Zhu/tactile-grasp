# Phase7 Step5 Main Results Multi-Seed Stability

| condition | source | force RMSE mean±std | slip F1 mean±std | H3 F1 mean±std | H5 F1 mean±std | H5 AUPRC mean±std | note |
|---|---|---:|---:|---:|---:|---:|---|
| separate_baseline | phase4_1 n=3 | 0.0315±0.0009 (n=3) | 0.9716±0.0033 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a |  |
| decoupled_multitask | phase4_1 n=3 | 0.0317±0.0012 (n=3) | 0.9729±0.0067 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a |  |
| naive_shared_multitask | phase4_4 single seed only | 0.0364±0.0000 (n=1) | 0.9784±0.0000 (n=1) | n/a±n/a | n/a±n/a | n/a±n/a | only seed42 available; missing seeds are recorded, not fabricated |
| full_dynamics_future_head | phase6_1 n=3 | n/a±n/a (n=) | n/a±n/a (n=) | 0.9562±0.0008 | 0.9337±0.0013 | 0.9810±0.0005 |  |
| full_dynamics_plus_q | phase6_1 n=3 | n/a±n/a (n=) | n/a±n/a (n=) | 0.9546±0.0045 | 0.9313±0.0067 | 0.9813±0.0003 |  |

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
