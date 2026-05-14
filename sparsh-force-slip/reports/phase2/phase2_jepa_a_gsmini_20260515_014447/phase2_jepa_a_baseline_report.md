# Phase2 JEPA A Baseline Report

- Generated: `2026-05-15T07:45:57`
- Run ID: `phase2_jepa_a_gsmini_20260515_014447`
- Derived dataset: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- All epoch-0051 checkpoints present: `True`

## Summary metrics

| encoder | task | val force RMSE mean | val rmse Fx | val rmse Fy | val rmse Fz | val slip F1 | val slip acc | checkpoint | W&B |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| ijepa | force | 0.068671 | 0.059684 | 0.072353 | 0.073975 |  |  | `epoch-0051.pth` | [run](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_force_0) |
| ijepa | slip |  |  |  |  | 0.878049 | 0.950000 | `epoch-0051.pth` | [run](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_slip_0) |
| vjepa | force | 0.047615 | 0.058772 | 0.055289 | 0.028785 |  |  | `epoch-0051.pth` | [run](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_force_0) |
| vjepa | slip |  |  |  |  | 0.918919 | 0.970000 | `epoch-0051.pth` | [run](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_slip_0) |

## Experiment paths

### ijepa force

- Experiment: `2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_force`
- Directory: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_force`
- Checkpoint: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_force/checkpoints/epoch-0051.pth`
- W&B: `https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_force_0`
- Runtime seconds: `14217`

### ijepa slip

- Experiment: `2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_slip`
- Directory: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_slip`
- Checkpoint: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_slip/checkpoints/epoch-0051.pth`
- W&B: `https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_ijepa_slip_0`
- Runtime seconds: `10970`

### vjepa force

- Experiment: `2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_force`
- Directory: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_force`
- Checkpoint: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_force/checkpoints/epoch-0051.pth`
- W&B: `https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_force_0`
- Runtime seconds: `20986`

### vjepa slip

- Experiment: `2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_slip`
- Directory: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_slip`
- Checkpoint: `/vla1/zjy/sparsh_runs/experiments/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_slip/checkpoints/epoch-0051.pth`
- W&B: `https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/2026.05.15_01-44_phase2_jepa_a_gsmini_20260515_014447_vjepa_slip_0`
- Runtime seconds: `19237`

## Next step

Update `phase2_b_multitask.py` with IJEPA/VJEPA checkpoints and these A experiment ids, then run Phase2-B partially_shared lambda sweeps.
