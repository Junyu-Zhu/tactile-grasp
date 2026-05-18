# Phase7 Step7 Runtime / Model Size / Inference Cost

## Parameter counts

| model | parameters |
|---|---:|
| separate_force_head | 7236867 |
| separate_slip_head | 7386437 |
| naive_shared_total | 93716357 |
| naive_shared_downstream_excl_encoder | 7460357 |
| decoupled_total | 101175749 |
| decoupled_downstream_excl_encoder | 14919749 |
| future_head_full_plus_q | 531475 |

## Future head cached-feature latency
- device: `cpu`
- batch_size: `512`
- ms_per_batch: `2.257983`
- ms_per_sample: `0.004410`

## Notes
- CPU latency measured for future head on cached features; encoder extraction time is cache-based/not remeasured to avoid new raw-data passes.
- GPU memory was not actively stressed; use nvidia-smi during training for deployment-grade numbers.
