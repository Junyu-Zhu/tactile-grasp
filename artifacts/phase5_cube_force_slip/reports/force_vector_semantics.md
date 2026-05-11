# Force vector semantics

- Frame: `phase5_gripper_scalar_normal_v1`.
- Order: `[tangent_a_n, tangent_b_n, normal_n]`.
- Unit: Newtons.
- Sign: positive `normal_n` means compressive contact along the gripper scalar
  normal; tangential force components are intentionally zero because Phase 3/4
  recordings expose scalar contact magnitudes, not signed shear vectors.
- Valid metric vector: `[0, 0, normal_force_n]`, where `normal_force_n` is
  `max(abs(left_force_n), abs(right_force_n), abs(max_force_n))` from simulation
  contact sensors when nonzero.
- Valid no-contact vector: `[0, 0, 0]` with `force_label_source=sim_no_contact_zero`.
- Explicitly invalid for Phase 5 metrics: `[left_force_n, right_force_n, max_force_n]`.
- Invalid mask: geometry-contact frames with zero sensor force are exported but
  marked `force_valid_for_metrics=False`.

## Label-source counts

| item | count |
| --- | ---: |
| `sim_no_contact_zero` | 1184 |
| `invalid_geometry_contact_without_force_scalar` | 982 |
| `sim_contact_sensor_scalar_normal` | 124 |


## Force-regime counts

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
