# Force-Slip Phase 1 GSmini Preparation Report

Generated: 2026-05-12T04:33:33.289830 on ps

## Paths

- Raw dataset root: `/vla1/zjy/tactile_datasets/Gelsight-mini/gelsight-force-estimation`
- Output root: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331`
- Derived dataset root: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- tactile-grasp repo: `/home/zjy/document/tactile-grasp`
- Sparsh repo: `/home/zjy/document/sparsh`

## Manifest

- Source datasets: 10
- Official training files use `dataset_gelsight_*.pkl` plus `dataset_slip_forces.pkl`.
- `org_dataset_gelsight_*.pkl` is recorded for domain diagnostics only and is not included in derived training directories.

## Split

- Train derived datasets: 10
- Val derived datasets: 10
- Test derived datasets: 10
- Split overlap issues: 0

## Smoke Test

- Passed: `True`
- Checked samples: 4

## Force Axis / Unit

- Inferred normal axis: `Fz`
- Confidence: `medium`
- Newton mapping if Fz normal: `{'Fn': 'abs(Fz_N)', 'Ft': 'sqrt(Fx_N^2 + Fy_N^2)', 'Fmag': 'sqrt(Fx_N^2 + Fy_N^2 + Fz_N^2)'}`

## Slip Alignment Audit

- Passed: `True`
- Issue count: 0

## Training Lists

Force task train/val/test lists are stored in `phase1_context.json` as `force_*_datasets`.
Slip task train/val/test lists are stored as `slip_*_datasets` and intentionally use `sphere_*` derived datasets.
Test lists are reserved for final evaluation, not hyperparameter tuning.
