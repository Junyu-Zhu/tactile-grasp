# Sparsh Force-Slip Workspace

This directory is the canonical workspace for force-slip stage code in `tactile_grasp`.

## Path contract

- Local development path: `/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip`
- Server execution path: `/documents/tactile_grasp/sparsh-force-slip`
- Training outputs/checkpoints stay outside the repo under `/vla1/zjy/sparsh_runs`
- Datasets and base models stay under `/vla1/zjy/{tactile_datasets,sparsh_models}`

## Sync contract

Do not develop directly on `zjy-4090` as the source of truth.

1. Edit force-slip code locally in this directory.
2. Commit and push from `/home/zjy/Documents/grasp/tactile_grasp`.
3. SSH to `zjy-4090`, `cd /documents/tactile_grasp`, and pull from GitHub.
4. Run training/evaluation from `/documents/tactile_grasp/sparsh-force-slip`.

Keep all force-slip models, configs, training entrypoints, metrics scripts, split/manifest tools, and run docs here unless a pipeline document explicitly states otherwise.
