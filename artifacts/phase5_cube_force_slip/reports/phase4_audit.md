# Phase 4 audit / proxy-boundary freeze

Phase 5 does **not** silently upgrade Phase 4 proxy labels.  Existing Phase 4
bridge outputs remain usable for format/dataloader smoke only unless a Phase 5
row-level valid mask marks a newly derived label as simulation-valid.

## Observed Phase 4 bridge gates

| item | count |
| --- | ---: |
| `FORCE_LABEL_VALIDITY` | sim-valid |
| `FORCE_SMOKE_PASS` | PASS_DATALOADER_ONLY |
| `FORMAT_BRIDGE_PASS` | PASS |
| `METRIC_VALIDITY` | limited-proxy |
| `SLIP_LABEL_VALIDITY` | proxy |
| `SLIP_SMOKE_PASS` | PASS_DATALOADER_ONLY |


## Phase 5 boundary rule

- Every exported row has `force_label_source`, `slip_label_source`, and
  `label_valid_for_metrics` in `sample_index.csv`.
- Phase 4's invalid/proxy force vector `[left_force_n, right_force_n, max_force_n]`
  is not reused as a metric vector.
- Geometry-only contact rows with zero force are kept for provenance and format
  continuity but have `force_valid_for_metrics=False`.
- Release/end rows are masked for slip metrics unless a future explicit release-slip
  protocol is introduced.
