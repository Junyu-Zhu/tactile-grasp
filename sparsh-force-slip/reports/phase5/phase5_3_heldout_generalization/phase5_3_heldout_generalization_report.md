# Phase5-3 Held-out Contact Generalization

- generated_at: `2026-05-18T00:46:30`
- raw_data_modified: `False`

## Results

| condition | train n | val/test n | H1 F1 | H1 AUROC | H1 AUPRC | H1 ECE | H1 high recall | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC | current force RMSE | current slip F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separate_late_fusion | 21938 | 9718 | 0.9623 | 0.9865 | 0.9791 | 0.0071 | 0.9900 | 0.9089 | 0.9436 | 0.8525 | 0.9145 | 0.0232 | 0.9739 |
| decoupled_static | 21938 | 9718 | 0.9617 | 0.9861 | 0.9787 | 0.0049 | 1.0000 | 0.9185 | 0.9456 | 0.8636 | 0.9164 | 0.0229 | 0.9672 |
| decoupled_dynamics | 21938 | 9718 | 0.9552 | 0.9973 | 0.9921 | 0.0223 | 0.9980 | 0.8922 | 0.9778 | 0.7392 | 0.9543 | 0.0229 | 0.9672 |
| decoupled_dynamics_friction | 21938 | 9718 | 0.9620 | 0.9980 | 0.9934 | 0.0054 | 0.9967 | 0.9400 | 0.9814 | 0.9169 | 0.9573 | 0.0229 | 0.9672 |

## Recommendation

Held-out split train=['flat', 'sharp'], test=['sphere']. Best H5 AUPRC condition: `decoupled_dynamics_friction`. Interpret this as contact-geometry stress testing, not final robot deployment evidence.

## Notes

- Training uses only train split samples from the selected train groups; evaluation uses val split samples from held-out groups, avoiding test-threshold tuning.
- No raw datasets are modified; generated heads/checkpoints are under /vla1/zjy/sparsh_runs/force_slip_phase5 when friction features are used and Phase4 feature roots otherwise.
