# Phase4-2 Separate Late Fusion vs Decoupled Joint

- generated_at: `2026-05-17T17:28:29`
- raw_data_modified: `False`

| condition | feature kind | input mode | force RMSE | slip F1 | slip acc | H1 F1 | H1 AUROC | H1 AUPRC | H1 ECE | H3 F1 | H5 F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separate_late_fusion | separate_late_fusion | z_force_slip | 0.0300 | 0.9684 | 0.9861 | 0.9644 | 0.9902 | 0.9849 | 0.0068 | 0.9203 | 0.8800 |
| decoupled_joint | decoupled_joint | z_force_slip | 0.0284 | 0.9593 | 0.9819 | 0.9626 | 0.9907 | 0.9842 | 0.0078 | 0.9160 | 0.8710 |
| decoupled_joint_dynamics | decoupled_joint | full | 0.0284 | 0.9593 | 0.9819 | 0.9676 | 0.9980 | 0.9953 | 0.0061 | 0.9562 | 0.9353 |

## Notes

- Separate late fusion uses the SPARSH-style force-only and slip-only MAE tasks for current predictions, then trains only a lightweight future-stability head.
- Decoupled joint uses one MAE decoupled multitask checkpoint; dynamics additionally includes causal delta_force.

## Artifacts

- separate_late_fusion: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_2_20260517_171500/phase4_2_separate_late_fusion_20260517_171500_report.json`
- decoupled_joint: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_2_20260517_171500/phase4_2_decoupled_joint_20260517_171500_report.json`
- decoupled_joint_dynamics: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_2_20260517_171500/phase4_2_decoupled_joint_dynamics_20260517_171500_report.json`
