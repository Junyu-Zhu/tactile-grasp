# Phase6-2 Held-out Sharp Generalization

- generated_at: `2026-05-18T01:51:58`
- raw_data_modified: `False`

## Results

| condition | train n | val/test n | H1 F1 | H1 AUROC | H1 AUPRC | H1 ECE | H1 high recall | H1 lead mean | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC | current force RMSE | current slip F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| separate_late_fusion | 52081 | 2799 | 0.9556 | 0.9805 | 0.9765 | 0.0241 | 1.0000 | 2.8571 | 0.9031 | 0.9435 | 0.8614 | 0.9174 | 0.0429 | 0.9701 |
| decoupled_static | 52081 | 2799 | 0.9597 | 0.9836 | 0.9784 | 0.0130 | 1.0000 | 12.5000 | 0.8914 | 0.9475 | 0.8494 | 0.9211 | 0.0372 | 0.9606 |
| decoupled_dynamics | 52081 | 2799 | 0.9659 | 0.9978 | 0.9948 | 0.0066 | 1.0000 | 11.6667 | 0.9397 | 0.9866 | 0.8675 | 0.9661 | 0.0372 | 0.9606 |
| decoupled_dynamics_friction | 52081 | 2799 | 0.9644 | 0.9980 | 0.9951 | 0.0048 | 1.0000 | 8.7000 | 0.9447 | 0.9870 | 0.8756 | 0.9665 | 0.0372 | 0.9606 |

## Interpretation

Best H5 AUPRC on held-out sharp is `decoupled_dynamics_friction`. This complements Phase5's flat+sharp→sphere stress test by holding out a localized sharp contact geometry.

## Notes

- Only lightweight heads are trained; SPARSH encoder and force/slip estimators are reused from Phase4/5.
- The split uses train samples from flat+sphere and validation samples from sharp.
