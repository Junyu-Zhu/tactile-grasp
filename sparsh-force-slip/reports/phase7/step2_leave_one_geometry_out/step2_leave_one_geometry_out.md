# Phase7 Step2 Leave-One-Contact-Geometry-Out

- raw_data_modified: `False`

## New split: train sharp+sphere -> test flat

| condition | current force RMSE | current slip F1 | H1 F1 | H3 F1 | H5 F1 | H5 AUPRC | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| separate_late_fusion | 0.0339 | 0.9498 | 0.9569 | 0.9296 | 0.8904 | 0.9560 | 34 |
| decoupled_static | 0.0350 | 0.9338 | 0.9582 | 0.9160 | 0.8659 | 0.9556 | 34 |
| decoupled_dynamics | 0.0350 | 0.9338 | 0.9539 | 0.9285 | 0.9120 | 0.9820 | 40 |
| decoupled_dynamics_friction | 0.0350 | 0.9338 | 0.9596 | 0.9415 | 0.9183 | 0.9834 | 37 |

## Combined three held-out splits

| split | condition | H3 F1 | H5 F1 | H5 AUPRC |
|---|---|---:|---:|---:|
| flat+sharp_to_sphere | separate_late_fusion | 0.9089 | 0.8525 | 0.9145 |
| flat+sharp_to_sphere | decoupled_static | 0.9185 | 0.8636 | 0.9164 |
| flat+sharp_to_sphere | decoupled_dynamics | 0.8922 | 0.7392 | 0.9543 |
| flat+sharp_to_sphere | decoupled_dynamics_friction | 0.9400 | 0.9169 | 0.9573 |
| flat+sphere_to_sharp | separate_late_fusion | 0.9031 | 0.8614 | 0.9174 |
| flat+sphere_to_sharp | decoupled_static | 0.8914 | 0.8494 | 0.9211 |
| flat+sphere_to_sharp | decoupled_dynamics | 0.9397 | 0.8675 | 0.9661 |
| flat+sphere_to_sharp | decoupled_dynamics_friction | 0.9447 | 0.8756 | 0.9665 |
| sharp+sphere_to_flat | separate_late_fusion | 0.9296 | 0.8904 | 0.9560 |
| sharp+sphere_to_flat | decoupled_static | 0.9160 | 0.8659 | 0.9556 |
| sharp+sphere_to_flat | decoupled_dynamics | 0.9285 | 0.9120 | 0.9820 |
| sharp+sphere_to_flat | decoupled_dynamics_friction | 0.9415 | 0.9183 | 0.9834 |
