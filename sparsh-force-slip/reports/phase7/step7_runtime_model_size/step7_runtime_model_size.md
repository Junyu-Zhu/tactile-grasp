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

## Measured current force/slip inference latency

| model | device | batch | encoder feature extraction ms/batch | force/slip heads ms/batch | total ms/batch | total ms/sample | trainable params |
|---|---|---:|---:|---:|---:|---:|---:|
| naive_shared_mae | cuda:0 | 64 | 123.477918 | 1.768546 | 134.942540 | 2.108477 | 7460357 |
| decoupled_mae | cuda:0 | 64 | 120.939902 | 4.342638 | 137.251879 | 2.144561 | 14919749 |

## Future head cached-feature latency
- device: `cuda:0`
- batch_size: `512`
- ms_per_batch: `0.098856`
- ms_per_sample: `0.000193`
- trainable_params: `531475`

## Separate probing deployment estimate

Separate force/slip probing checkpoints are independent downstream tasks. For a naive real-time deployment without encoder-feature reuse, the estimate is two frozen-encoder passes plus head cost; with feature reuse, it reduces to one encoder pass plus two lightweight heads.

- two-pass total estimate: `248.724382` ms/batch64
- shared-encoder lower-bound estimate: `127.015010` ms/batch64

## GPU memory snapshot

- `0, NVIDIA GeForce RTX 4090, 0 %, 7502 MiB, 49140 MiB`
- `1, NVIDIA GeForce RTX 4090, 0 %, 7462 MiB, 49140 MiB`
- `2, NVIDIA GeForce RTX 4090, 75 %, 6752 MiB, 49140 MiB`
- `3, NVIDIA GeForce RTX 4090, 73 %, 4875 MiB, 49140 MiB`

## Notes
- Encoder feature extraction time is measured as the frozen Sparsh MAE encoder forward on a validation batch; no training data were modified.
- Force/slip head time is measured by running the decoder on the cached encoder tokens from the same batch.
- Separate probing latency is reported as a deployment estimate because the force and slip probes are independent checkpoints rather than one combined callable model.
- Measurements were taken on a shared server, so absolute latency can vary with concurrent jobs; relative head/encoder scale is the intended evidence.
