# Phase2 JEPA A Baseline Launch Manifest

- Run ID: phase2_jepa_a_gsmini_20260515_014447
- Created: 2026-05-15T01:44:48
- Derived dataset: /vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini
- Rule: independent tmux session per training; one GPU per training via CUDA_VISIBLE_DEVICES and +trainer.devices=1

## nvidia-smi before launch
```
0, 3676, 49140, 0, 0, 88.86
1, 3636, 49140, 0, 0, 103.24
2, 3640, 49140, 0, 0, 115.70
3, 3639, 49140, 68, 21, 86.83
```

## Tasks
| tmux session | GPU | encoder | task | experiment_name | runbook | log |
|---|---:|---|---|---|---|---|
| phase2_jepa_a_gsmini_20260515_014447_ijepa_force_gpu0 | 0 | ijepa | force | phase2_jepa_a_gsmini_20260515_014447_ijepa_force | /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/phase2_jepa_a_phase2_jepa_a_gsmini_20260515_014447/run_ijepa_force.sh | /home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_jepa_a_gsmini_20260515_014447/ijepa_force_gpu0.log |
| phase2_jepa_a_gsmini_20260515_014447_ijepa_slip_gpu1 | 1 | ijepa | slip | phase2_jepa_a_gsmini_20260515_014447_ijepa_slip | /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/phase2_jepa_a_phase2_jepa_a_gsmini_20260515_014447/run_ijepa_slip.sh | /home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_jepa_a_gsmini_20260515_014447/ijepa_slip_gpu1.log |
| phase2_jepa_a_gsmini_20260515_014447_vjepa_force_gpu2 | 2 | vjepa | force | phase2_jepa_a_gsmini_20260515_014447_vjepa_force | /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/phase2_jepa_a_phase2_jepa_a_gsmini_20260515_014447/run_vjepa_force.sh | /home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_jepa_a_gsmini_20260515_014447/vjepa_force_gpu2.log |
| phase2_jepa_a_gsmini_20260515_014447_vjepa_slip_gpu3 | 3 | vjepa | slip | phase2_jepa_a_gsmini_20260515_014447_vjepa_slip | /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/phase2_jepa_a_phase2_jepa_a_gsmini_20260515_014447/run_vjepa_slip.sh | /home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_jepa_a_gsmini_20260515_014447/vjepa_slip_gpu3.log |
