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

## GPU memory snapshot

- `0, NVIDIA GeForce RTX 4090, 0 %, 6672 MiB, 49140 MiB`
- `1, NVIDIA GeForce RTX 4090, 0 %, 6632 MiB, 49140 MiB`
- `2, NVIDIA GeForce RTX 4090, 0 %, 6610 MiB, 49140 MiB`
- `3, NVIDIA GeForce RTX 4090, 0 %, 6610 MiB, 49140 MiB`

## Encoder feature extraction status

- End-to-end encoder extraction was not remeasured in Phase7; this phase reused cached Sparsh features to avoid additional raw-data passes.
