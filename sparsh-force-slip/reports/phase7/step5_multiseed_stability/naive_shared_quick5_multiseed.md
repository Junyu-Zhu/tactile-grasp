# Naive Shared Quick 5-Epoch Multi-Seed Stability

- raw_data_modified: `False`
- purpose: supplement Phase7 Step5 with 3 seeds for the naive shared decoder.
- caveat: matched 5-epoch checkpoints measure seed sensitivity but are not directly comparable to the full 51-epoch seed42 reference.

| seed | run_id | epoch | force RMSE | slip F1 | slip accuracy | checkpoint |
|---:|---|---:|---:|---:|---:|---|
| 42 | phase2_b_gsmini_20260513_163448 | 5 | 0.0555 | 0.9720 | 0.9861 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_gsmini_20260513_163448/mae_shared_multitask/checkpoints/epoch-0005.pth` |
| 43 | phase7_naive_shared_quick_seed43_20260519 | 5 | 0.0550 | 0.9742 | 0.9872 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed43_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth` |
| 44 | phase7_naive_shared_quick_seed44_20260519 | 5 | 0.0557 | 0.9779 | 0.9891 | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase7_naive_shared_quick_seed44_20260519/mae_shared_multitask/checkpoints/epoch-0005.pth` |

## Summary

- force RMSE: 0.0554±0.0003 (n=3)
- slip F1: 0.9747±0.0024 (n=3)
