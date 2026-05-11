# Phase 4 completion audit

This branch implements Phase 4 as a cube-only bridge, collection, and smoke-test
stage. Generated data is intentionally stored under ignored `sim_dataset/`.

## Source and output locations

- Existing cube artifacts: `artifacts/phase3/phase3_cube_*`
- New Phase 4 cube collection roots:
  - `sim_dataset/phase4_collected/bridge_robustness`
  - `sim_dataset/phase4_collected/force_oriented`
  - `sim_dataset/phase4_collected/slip_oriented`
- Derived Sparsh bridge:
  - `sim_dataset/phase4_sparsh_cube/cube_phase3_bridge`

## Prompt-to-artifact checklist

| Requirement | Evidence |
| --- | --- |
| Create git branch `phase4` | `git branch --show-current` reports `phase4`. |
| Commit key steps | Commits on branch: gitignore setup, then Phase4 bridge/smoke tooling. |
| Put `sim_dataset` in gitignore | `.gitignore` contains `sim_dataset/`. |
| Use TacEx environment | Collection and smoke commands used `conda run -n tacex`. |
| Install missing libraries if needed | Installed `lightning` in `tacex` after Sparsh loader import failed. |
| Save newly collected data under `sim_dataset` | Phase4 collection roots are under `sim_dataset/phase4_collected/*`. |
| Step 1 inventory | `sim_dataset/phase4_sparsh_cube/inventory.{json,csv}`. |
| Step 2 Sparsh-compatible bridge | `dataset_gelsight_cube.pkl`, `dataset_slip_forces.pkl`, `manifest.csv`, `README.md`. |
| Step 3 force dataloader smoke | `phase4_sparsh_smoke.py`; `bridge_report.json` reports `FORCE_SMOKE_PASS=PASS_DATALOADER_ONLY`. |
| Step 4 slip dataloader smoke | `phase4_sparsh_smoke.py`; `bridge_report.json` reports `SLIP_SMOKE_PASS=PASS_DATALOADER_ONLY`. |
| Step 5 bridge robustness batch | New TacEx/Isaac cube trial in `sim_dataset/phase4_collected/bridge_robustness`, reviewed as passed. |
| Step 6 force-oriented protocol | New TacEx/Isaac cube trial in `sim_dataset/phase4_collected/force_oriented`, reviewed as passed, with nonzero sim contact force. |
| Step 7 slip-oriented protocol | New TacEx/Isaac cube trial in `sim_dataset/phase4_collected/slip_oriented`, reviewed as passed, with relative-motion proxy slip labels. |
| Step 8 label validity gate | `bridge_report.json`: force `sim-valid`, slip `proxy`, metrics `limited-proxy`. |
| Step 9 summary/decision | `sim_dataset/phase4_sparsh_cube/summary.md` and `bridge_report.json`. |

## Latest verification snapshot

Commands run from `tactile_grasp/`:

```bash
conda run -n tacex python phase4_sim_dataset.py
PYTHONPATH=/home/zjy/Documents/grasp/sparsh conda run -n tacex python phase4_sparsh_smoke.py
conda run -n tacex python ur5_phase3_review.py --output_root sim_dataset/phase4_collected/bridge_robustness
conda run -n tacex python ur5_phase3_review.py --output_root sim_dataset/phase4_collected/force_oriented
conda run -n tacex python ur5_phase3_review.py --output_root sim_dataset/phase4_collected/slip_oriented
```

Observed bridge totals:

- source trials: 6
- frame rows: 1372
- trajectories: 6
- positive proxy slip labels: 218
- max force label: 1.792631 N
- force smoke: `PASS_DATALOADER_ONLY`
- slip smoke: `PASS_DATALOADER_ONLY`

## Important limits

- No Sparsh model-forward metrics were run because no Sparsh encoder/task
  checkpoint was supplied. The exact model-forward blocker is recorded in
  `sim_dataset/phase4_sparsh_cube/smoke_report.json`.
- Force labels include real simulation contact-sensor values where available,
  but remaining samples may be geometry/placeholder labels.
- Slip labels are simulation relative-motion proxy labels, not real-world
  controlled slip ground truth.
- Therefore metric validity is only `limited-proxy`, not final real evaluation.
