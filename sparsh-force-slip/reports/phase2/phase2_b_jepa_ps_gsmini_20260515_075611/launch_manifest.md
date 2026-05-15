# Phase2-B JEPA partially_shared launch phase2_b_jepa_ps_gsmini_20260515_075611

- created_at: `2026-05-15T07:56:11`
- branch: `sparsh-force-slip`
- run_ids: lambda 0.25 `phase2_b_jepa_lam025_gsmini_20260515_075611`, lambda 0.50 `phase2_b_jepa_lam050_gsmini_20260515_075611`
- constraint: Each run uses CUDA_VISIBLE_DEVICES=<one gpu> and +trainer.devices=1; no --data-parallel.

## GPU snapshot before launch
```
0, NVIDIA GeForce RTX 4090, 3676 MiB, 49140 MiB, 0 %
1, NVIDIA GeForce RTX 4090, 3636 MiB, 49140 MiB, 33 %
2, NVIDIA GeForce RTX 4090, 3640 MiB, 49140 MiB, 6 %
3, NVIDIA GeForce RTX 4090, 3639 MiB, 49140 MiB, 67 %
```

## Tasks
| session | encoder | lambda | GPU | W&B name | log | expected checkpoint |
|---|---:|---:|---:|---|---|---|
| `p2b_jepa_20260515_075611_ijepa_l025_g0` | ijepa | 0.25 | 0 | `phase2_b_jepa_lam025_gsmini_20260515_075611_ijepa_partially_shared_lambda025` | `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_b_jepa_ps_gsmini_20260515_075611/p2b_jepa_20260515_075611_ijepa_l025_g0.log` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |
| `p2b_jepa_20260515_075611_ijepa_l050_g1` | ijepa | 0.50 | 1 | `phase2_b_jepa_lam050_gsmini_20260515_075611_ijepa_partially_shared_lambda050` | `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_b_jepa_ps_gsmini_20260515_075611/p2b_jepa_20260515_075611_ijepa_l050_g1.log` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/ijepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |
| `p2b_jepa_20260515_075611_vjepa_l025_g2` | vjepa | 0.25 | 2 | `phase2_b_jepa_lam025_gsmini_20260515_075611_vjepa_partially_shared_lambda025` | `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_b_jepa_ps_gsmini_20260515_075611/p2b_jepa_20260515_075611_vjepa_l025_g2.log` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam025_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |
| `p2b_jepa_20260515_075611_vjepa_l050_g3` | vjepa | 0.50 | 3 | `phase2_b_jepa_lam050_gsmini_20260515_075611_vjepa_partially_shared_lambda050` | `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_b_jepa_ps_gsmini_20260515_075611/p2b_jepa_20260515_075611_vjepa_l050_g3.log` | `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_jepa_lam050_gsmini_20260515_075611/vjepa_partially_shared_multitask/checkpoints/epoch-0051.pth` |
