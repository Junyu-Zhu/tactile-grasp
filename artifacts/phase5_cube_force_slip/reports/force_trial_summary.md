# Force trial summary

Exported frames: 2290  
Trajectories: 8  
Force-valid frames: 1308  
Force-invalid/masked frames: 982  
Max normal force (N): 2.539855

Phase 5 force regimes are represented by `force_regime` in `sample_index.csv`.
The categories include `no_contact`, light/medium/firm/over contact bands, and
stage roles such as `lift_and_hold_*` and `contact_release` when present.

## Regime counts

| item | count |
| --- | ---: |
| `no_contact` | 1066 |
| `contact_geometry_no_force_invalid` | 770 |
| `contact_release` | 214 |
| `release_no_contact` | 118 |
| `firm` | 32 |
| `over` | 32 |
| `hold_over` | 20 |
| `hold_firm` | 18 |
| `hold_medium` | 8 |
| `medium` | 8 |
| `light` | 4 |


## Source roots

- `artifacts/phase3`
- `sim_dataset/phase4_collected/bridge_robustness`
- `sim_dataset/phase4_collected/force_oriented`
- `sim_dataset/phase4_collected/slip_oriented`
- `sim_dataset/phase5_cube_force_slip/raw/force_v2`
- `sim_dataset/phase5_cube_force_slip/raw/slip_v2`
