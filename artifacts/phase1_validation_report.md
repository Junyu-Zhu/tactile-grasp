# Phase 1 Validation Report

## Source of truth
- Script: `tactile_grasp/ur5_sim.py`
- Control helper: `tactile_grasp/ur5_phase1_control.py`
- Reset helper: `tactile_grasp/ur5_phase1_reset.py`
- Checks helper: `tactile_grasp/ur5_phase1_checks.py`

## Runtime environment
- Conda env: `tacex`
- IsaacLab source injected via `PYTHONPATH`
- Mode: headless

## Validation command
```bash
PYTHONPATH="/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab:/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab_assets:/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab_tasks:/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab_mimic:/home/zjy/Documents/grasp/TacEx/IsaacLab/source/isaaclab_rl" \
conda run -s -n tacex python -u /home/zjy/Documents/grasp/tactile_grasp/ur5_sim.py \
  --headless --phase1-checks --gripper-cycles 20 --reset-trials 20 --pregrasp-trials 20
```

## Result
- Final line: `[RESULT] Phase 1 validation PASSED.`
- Arm joint actuation: PASS
- Robotiq open/close (20 cycles): PASS
- Deterministic reset (20 trials): PASS
- Fixed pre-grasp reach (20 trials): PASS

## Key numerical evidence
- Arm joint errors stayed within configured tolerance (`< 0.05 rad`)
- Reset arm max joint error: `0.007187366485595703 rad`
- Reset gripper max joint error: `0.02848396636545658 rad`
- Banana reset position error: `0.000616908073425293 m`
- Pre-grasp final worst joint error: `0.006471872329711914 rad`
- Example pre-grasp ee position: `[0.6002119779586792, 0.0004318580904509872, 0.9803544282913208]`

## Notes
- The validation process printed PASS before the wrapper shell later terminated the Python process; the PASS result and full JSON summary are preserved in `tactile_grasp/artifacts/phase1_validation_latest.log`.
- Screenshot / short video artifacts are still optional follow-up items if needed for presentation, but the Phase 1 functional gates are now satisfied.
