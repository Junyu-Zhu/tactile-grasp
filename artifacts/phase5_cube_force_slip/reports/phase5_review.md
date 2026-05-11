# Phase 5 review

## Scope check

- Cube-only simulation: satisfied.
- Forcefield/chips_can/cracker_box/real robot/real data/RL/GraspNet/closed-loop:
  not introduced.
- Heavy data location: `sim_dataset/phase5_cube_force_slip/sparsh_export_v2/cube_force_slip_v2` under repo-local `sim_dataset/`.
- Reviewable reports: `artifacts/phase5_cube_force_slip/reports`.

## Gate summary

| item | count |
| --- | ---: |
| `EVAL_GATE` | BLOCKED_NO_CHECKPOINT_NO_TRUSTED_METRICS |
| `FORCE_LABEL_VALIDITY` | sim-valid |
| `FORMAT_GATE` | PASS_DATALOADER_ONLY |
| `SLIP_LABEL_VALIDITY` | sim-valid |


## Counts

| item | count |
| --- | ---: |
| `frames` | 2290 |
| `slip_invalid_frames` | 1434 |
| `force_valid_frames` | 1308 |
| `in_contact_frames` | 1106 |
| `force_invalid_frames` | 982 |
| `slip_valid_frames` | 856 |
| `slip_negative_valid_frames` | 560 |
| `release_masked_slip_frames` | 332 |
| `slip_positive_valid_frames` | 296 |
| `trajectories` | 8 |


## Remaining limitation

Labels are valid only for limited cube simulation checks with masks.  No real
Sparsh checkpoint was bundled, and no benchmark metrics should be claimed unless
`eval_gate.md` is updated by a checkpoint-backed evaluation run.

## Sparsh smoke evidence

- FORMAT_GATE: PASS_DATALOADER_ONLY
- Label gate pass: True
- EVAL_GATE: BLOCKED_NO_CHECKPOINT_NO_TRUSTED_METRICS
- Trusted metrics: not run.

## Sparsh model/data effect diagnostic

- Report: `artifacts/phase5_cube_force_slip/reports/sparsh_model_data_effect.md`
- Local encoder used: `/home/zjy/Documents/dataset1/sparsh/sparsh_models/sparsh-dino-small/dino_vitsmall.ckpt`
- Normal-force frozen-encoder diagnostic on new Phase5 trajectories:
  RMSE 0.2118 N on all valid force samples, 0.4540 N on nonzero-force samples.
- Slip frozen-encoder diagnostic on new Phase5 trajectories:
  balanced accuracy 0.6461, F1 0.4630.
- This is not an official TacBench/Sparsh metric because the local model set
  contains encoder checkpoints but no trained force/slip task decoder checkpoint.
