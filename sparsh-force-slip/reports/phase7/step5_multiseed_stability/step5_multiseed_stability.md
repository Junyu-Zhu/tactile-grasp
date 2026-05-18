# Phase7 Step5 Main Results Multi-Seed Stability

| condition | source | force RMSE mean±std | slip F1 mean±std | H3 F1 mean±std | H5 F1 mean±std | H5 AUPRC mean±std | note |
|---|---|---:|---:|---:|---:|---:|---|
| separate_baseline | phase4_1 n=3 | 0.0315±0.0009 (n=3) | 0.9716±0.0033 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a |  |
| decoupled_multitask | phase4_1 n=3 | 0.0317±0.0012 (n=3) | 0.9729±0.0067 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a |  |
| naive_shared_multitask | phase4_4 single seed only | 0.0364±0.0000 (n=1) | 0.9784±0.0000 (n=1) | n/a±n/a | n/a±n/a | n/a±n/a | only seed42 available; missing seeds are recorded, not fabricated |
| full_dynamics_future_head | phase6_1 n=3 | n/a±n/a (n=) | n/a±n/a (n=) | 0.9562±0.0008 | 0.9337±0.0013 | 0.9810±0.0005 |  |
| full_dynamics_plus_q | phase6_1 n=3 | n/a±n/a (n=) | n/a±n/a (n=) | 0.9546±0.0045 | 0.9313±0.0067 | 0.9813±0.0003 |  |
