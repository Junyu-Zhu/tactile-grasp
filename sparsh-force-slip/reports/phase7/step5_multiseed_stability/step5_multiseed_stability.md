# Phase7 Step5 Main Results Multi-Seed Stability

| condition | source | force RMSE mean±std | slip F1 mean±std | H3 F1 mean±std | H5 F1 mean±std | H5 AUPRC mean±std | note |
|---|---|---:|---:|---:|---:|---:|---|
| separate_baseline | phase4_1 n=3 | 0.0315±0.0009 (n=3) | 0.9716±0.0033 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a |  |
| decoupled_multitask | phase4_1 n=3 | 0.0317±0.0012 (n=3) | 0.9729±0.0067 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a |  |
| naive_shared_multitask | phase4_4 single seed only | 0.0364±0.0000 (n=1) | 0.9784±0.0000 (n=1) | n/a±n/a | n/a±n/a | n/a±n/a | only seed42 available; missing seeds are recorded, not fabricated |
| full_dynamics_future_head | phase6_1 n=3 | n/a±n/a (n=) | n/a±n/a (n=) | 0.9562±0.0008 | 0.9337±0.0013 | 0.9810±0.0005 |  |
| full_dynamics_plus_q | phase6_1 n=3 | n/a±n/a (n=) | n/a±n/a (n=) | 0.9546±0.0045 | 0.9313±0.0067 | 0.9813±0.0003 |  |
| naive_shared_multitask_quick5_supplemental | Phase7 matched 5-epoch checkpoints, seeds 42/43/44 | 0.0554±0.0003 (n=3) | 0.9747±0.0024 (n=3) | n/a±n/a | n/a±n/a | n/a±n/a | 3-seed quick stability check; not a full 51-epoch replacement for the original seed42 reference. |

## Naive shared quick 5-epoch supplemental seeds

This supplemental row closes the naive-shared 3-seed stability gap with matched 5-epoch checkpoints. It is useful for seed sensitivity, but the 5-epoch statistics are not directly comparable to the full 51-epoch single-seed reference row.

| seed | run_id | epoch | force RMSE | slip F1 | slip accuracy | checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 42 | phase2_b_gsmini_20260513_163448 | 5 | 0.0555 | 0.9720 | 0.9861 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_gsmini_20260513_163448/mae_shared_multitask/checkpoints/epoch-0005.pth` |
| 43 | phase7_naive_shared_quick_seed43_20260519 | 5 | 0.0550 | 0.9742 | 0.9872 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed43_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth` |
| 44 | phase7_naive_shared_quick_seed44_20260519 | 5 | 0.0557 | 0.9779 | 0.9891 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed44_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth` |

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
| naive_shared_quick5_force_rmse | 43 | 0.0550 | 44 | 0.0557 | True |
| naive_shared_quick5_slip_f1 | 44 | 0.9779 | 42 | 0.9720 | False |
