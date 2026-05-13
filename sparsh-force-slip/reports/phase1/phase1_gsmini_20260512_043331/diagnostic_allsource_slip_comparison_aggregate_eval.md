# Slip Diagnostic Training Comparison

- generated_at: `2026-05-13T15:38:09`
- run_id: `phase1_gsmini_20260512_043331`
- Scope: aggregate validation evaluation from saved `epoch-0051.pth` checkpoints, not W&B final-batch summaries.
- Eval sets:
  - `sphere_val`: sphere_batch_1_val, sphere_batch_2_val, sphere_batch_3_val, sphere_batch_4_val, sphere_batch_5_val, sphere_batch_6_val
  - `allsource_val`: flat_batch_1_val, flat_batch_2_val, sharp_batch_1_val, sharp_batch_2_val, sphere_batch_1_val, sphere_batch_2_val, sphere_batch_3_val, sphere_batch_4_val, sphere_batch_5_val, sphere_batch_6_val

## Main aggregate comparison

| encoder | train data | eval set | n | pos ratio | acc | F1 slip | recall slip | ΔF RMSE mean N |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DINOv2 | sphere-only | sphere_val | 9855 | 0.1999 | 0.9786 | 0.9479 | 0.9736 | 0.0430 |
| DINOv2 | all-source | sphere_val | 9855 | 0.1999 | 0.9777 | 0.9461 | 0.9797 | 0.0440 |
| DINOv2 | sphere-only | allsource_val | 14920 | 0.2468 | 0.9589 | 0.9213 | 0.9747 | 0.0544 |
| DINOv2 | all-source | allsource_val | 14920 | 0.2468 | 0.9735 | 0.9480 | 0.9777 | 0.0493 |
| MAE | sphere-only | sphere_val | 9855 | 0.1999 | 0.9916 | 0.9788 | 0.9726 | 0.0328 |
| MAE | all-source | sphere_val | 9855 | 0.1999 | 0.9902 | 0.9756 | 0.9838 | 0.0316 |
| MAE | sphere-only | allsource_val | 14920 | 0.2468 | 0.9884 | 0.9763 | 0.9680 | 0.0434 |
| MAE | all-source | allsource_val | 14920 | 0.2468 | 0.9864 | 0.9727 | 0.9826 | 0.0367 |

## Direct deltas

### DINOv2: all-source training vs sphere-only training on sphere_val

- accuracy: `0.9786` -> `0.9777` (-0.0009)
- F1 slip: `0.9479` -> `0.9461` (-0.0018)
- recall slip: `0.9736` -> `0.9797` (+0.0061)
- mean Δforce RMSE N: `0.0430` -> `0.0440` (+2.1%)

### DINOv2: all-source training vs sphere-only training on allsource_val

- accuracy: `0.9589` -> `0.9735` (+0.0146)
- F1 slip: `0.9213` -> `0.9480` (+0.0267)
- recall slip: `0.9747` -> `0.9777` (+0.0030)
- mean Δforce RMSE N: `0.0544` -> `0.0493` (-9.5%)

### MAE: all-source training vs sphere-only training on sphere_val

- accuracy: `0.9916` -> `0.9902` (-0.0014)
- F1 slip: `0.9788` -> `0.9756` (-0.0032)
- recall slip: `0.9726` -> `0.9838` (+0.0112)
- mean Δforce RMSE N: `0.0328` -> `0.0316` (-3.9%)

### MAE: all-source training vs sphere-only training on allsource_val

- accuracy: `0.9884` -> `0.9864` (-0.0020)
- F1 slip: `0.9763` -> `0.9727` (-0.0036)
- recall slip: `0.9680` -> `0.9826` (+0.0147)
- mean Δforce RMSE N: `0.0434` -> `0.0367` (-15.4%)

## Conclusion

All-source slip training is useful as a diagnostic but should not automatically replace the sphere-only baseline. On allsource_val, DINOv2 F1 changes from 0.9213 to 0.9480, while MAE F1 changes from 0.9763 to 0.9727. Use the detailed table to decide the tradeoff; if the formal goal is broad flat+sharp+sphere coverage, all-source training is the relevant candidate, but if the formal goal is maximum sphere-only slip performance, keep the existing sphere-only baseline.

## Notes

- W&B `wandb-summary.json` values are final logged batch values in this code path; the table above re-evaluates full validation sets from checkpoints for a fairer diagnostic.
- The all-source run is still diagnostic. Do not replace the sphere-only slip baseline unless the chosen evaluation policy accepts the all-source tradeoff.
