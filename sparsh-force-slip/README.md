# Sparsh Force-Slip Workspace

This directory is the canonical workspace for force-slip stage code, scripts, runbooks, reports, and source records in `tactile-grasp`.

## Path contract

- Server project root: `/home/zjy/document`
- Server repository: `/home/zjy/document/tactile-grasp`
- Server force-slip code path: `/home/zjy/document/tactile-grasp/sparsh-force-slip`
- Working branch: `sparsh-force-slip`
- Sparsh training repo: `/home/zjy/document/sparsh`
- Raw datasets: `/vla1/zjy/tactile_datasets` (read-only for this workflow)
- Derived datasets / run outputs: `/vla1/zjy/sparsh_runs`
- Base models: `/vla1/zjy/sparsh_models`

## Execution contract

Future force-slip code edits, data derivation, training, evaluation, and run-record updates happen on `zjy-4090` under `/home/zjy/document/tactile-grasp`.

```bash
ssh zjy-4090
cd /home/zjy/document/tactile-grasp
git switch sparsh-force-slip
cd sparsh-force-slip
```

`git push` and `git pull` are manual operations performed by the user. Do not make training scripts or automation push/pull automatically.

## Data contract

Do not modify raw data under `/vla1/zjy/tactile_datasets`. Generate derived manifests, split-specific datasets, reports, and training outputs under `/vla1/zjy/sparsh_runs`.

## Phase commit rule

After each phase is complete and verified, make one overall commit on the `sparsh-force-slip` branch covering that phase's code, configs, docs, scripts, run records, metrics summaries, and source records. Commit messages should follow the AGENTS.md Lore Commit Protocol.
