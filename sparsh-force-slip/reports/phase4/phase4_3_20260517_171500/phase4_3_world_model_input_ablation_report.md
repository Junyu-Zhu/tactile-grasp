# Phase4-3 World-model Input Ablation

- generated_at: `2026-05-17T17:20:01`
- raw_data_modified: `False`

| condition | feature kind | input mode | force RMSE | slip F1 | slip acc | H1 F1 | H1 AUROC | H1 AUPRC | H1 ECE | H3 F1 | H5 F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| z_only | decoupled_joint | z_only | 0.0284 | 0.9593 | 0.9819 | 0.9450 | 0.9752 | 0.9655 | 0.0136 | 0.8890 | 0.8401 |
| z_p_slip | decoupled_joint | z_p_slip | 0.0284 | 0.9593 | 0.9819 | 0.9608 | 0.9892 | 0.9833 | 0.0057 | 0.9175 | 0.8790 |
| z_force | decoupled_joint | z_force | 0.0284 | 0.9593 | 0.9819 | 0.9572 | 0.9851 | 0.9778 | 0.0165 | 0.9028 | 0.8656 |
| z_force_slip | decoupled_joint | z_force_slip | 0.0284 | 0.9593 | 0.9819 | 0.9626 | 0.9907 | 0.9842 | 0.0078 | 0.9160 | 0.8710 |
| full | decoupled_joint | full | 0.0284 | 0.9593 | 0.9819 | 0.9676 | 0.9980 | 0.9953 | 0.0061 | 0.9562 | 0.9353 |

## Notes

- Same MAE decoupled checkpoint, split, horizons, optimizer, and epoch budget are used for all input modes.
- full adds causal delta_force to z_t, force-derived scalars, and current slip probability.

## Artifacts

- z_only: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_z_only_20260517_171500_report.json`
- z_p_slip: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_z_p_slip_20260517_171500_report.json`
- z_force: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_z_force_20260517_171500_report.json`
- z_force_slip: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_z_force_slip_20260517_171500_report.json`
- full: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_full_20260517_171500_report.json`
