# Phase5-2 Friction Proxy Future Stability Head

- generated_at: `2026-05-18T00:39:20`
- raw_data_modified: `False`

## Results

| condition | train n | val/test n | H1 F1 | H1 AUROC | H1 AUPRC | H1 ECE | H1 high recall | H3 F1 | H3 AUPRC | H5 F1 | H5 AUPRC | current force RMSE | current slip F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_dynamics_baseline | 65938 | 14328 | 0.9666 | 0.9979 | 0.9952 | 0.0052 | 0.9950 | 0.9564 | 0.9936 | 0.9338 | 0.9805 | 0.0284 | 0.9593 |
| full_plus_r | 65938 | 14328 | 0.9640 | 0.9977 | 0.9950 | 0.0061 | 0.9950 | 0.9562 | 0.9935 | 0.9343 | 0.9804 | 0.0284 | 0.9593 |
| full_plus_r_dr | 65938 | 14328 | 0.9652 | 0.9979 | 0.9953 | 0.0068 | 0.9969 | 0.9545 | 0.9937 | 0.9292 | 0.9807 | 0.0284 | 0.9593 |
| full_plus_q | 65938 | 14328 | 0.9668 | 0.9983 | 0.9959 | 0.0053 | 0.9981 | 0.9593 | 0.9945 | 0.9381 | 0.9815 | 0.0284 | 0.9593 |
| full_plus_q_dq | 65938 | 14328 | 0.9639 | 0.9983 | 0.9958 | 0.0077 | 0.9977 | 0.9543 | 0.9943 | 0.9298 | 0.9813 | 0.0284 | 0.9593 |

## Recommendation

Recommended condition by H3 AUPRC: `full_plus_q`. Baseline H3 F1=0.9564; best H3 F1=0.9593.

## Notes

- Phase4 full dynamics already included a raw Ft/Fn column; Phase5 adds train-clipped r, temporal delta-r, calibrated q, and delta-q to test whether friction-aware conditioning adds value beyond the existing ratio.
- Selection uses mean AUPRC across H1/H3/H5 while the report highlights H3/H5 because longer-horizon warning is the paper-facing target.
