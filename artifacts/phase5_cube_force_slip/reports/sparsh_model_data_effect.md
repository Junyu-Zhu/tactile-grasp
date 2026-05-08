# Phase 5 Sparsh model/data effect diagnostic

## Local Sparsh assets

- Sparsh repo: `/home/zjy/Documents/grasp/sparsh`
- Sparsh data root: `/home/zjy/Documents/dataset1/sparsh/tactile_datasets`
- Sparsh model root: `/home/zjy/Documents/dataset1/sparsh/sparsh_models`
- Encoder used for diagnostic: `/home/zjy/Documents/dataset1/sparsh/sparsh_models/sparsh-dino-small/dino_vitsmall.ckpt`
- Encoder family/size: `sparsh-dino-small`

The supplied model directory contains SSL encoder checkpoints. It does not contain a task decoder checkpoint for `t1_force` or `t2_slip`, so this report does not claim official Sparsh/TacBench metrics.

## Phase 5 dataset coverage

- Dataset root: `sim_dataset/phase5_cube_force_slip/sparsh_export_v2/cube_force_slip_v2`
- Frames: 2290
- Trajectories: 8
- Force-valid frames: 1308 / 2290
- Slip-valid frames: 856 / 2290
- Valid slip positives / negatives: 296 / 560
- Gates: `{'FORMAT_GATE': 'PASS_DATALOADER_ONLY', 'FORCE_LABEL_VALIDITY': 'sim-valid', 'SLIP_LABEL_VALIDITY': 'sim-valid', 'EVAL_GATE': 'BLOCKED_NO_CHECKPOINT_NO_TRUSTED_METRICS'}`

## Frozen-Sparsh diagnostic probes

Train split: older Phase3/Phase4 cube trajectories. Test split: newly collected `phase5_force_v2` and `phase5_slip_v2` trajectories. These probes are small sklearn heads on frozen encoder mean-pooled patch tokens, not the Sparsh task decoders.

### Normal-force diagnostic

- Status: `PASS_DIAGNOSTIC_LINEAR_PROBE`
- Train/test samples: 741 / 547
- Test nonzero-force samples: 48
- All valid RMSE: 0.2118 N (baseline mean: 0.4485 N)
- All valid MAE: 0.0345 N (baseline mean: 0.2389 N)
- Nonzero-force RMSE: 0.4540 N (baseline: 1.4591 N)
- Test max normal force: 2.5399 N
- Interpretation: this only tests scalar normal force; Phase 5 still has no shear-force ground truth.

### Slip diagnostic

- Status: `PASS_DIAGNOSTIC_LINEAR_PROBE`
- Train positives/negatives: 214 / 292
- Test positives/negatives: 82 / 268
- Balanced accuracy: 0.6461 (majority baseline: 0.5000)
- F1: 0.4630 (majority baseline: 0.0000)
- Precision/recall: 0.3144 / 0.8780
- Confusion matrix [TN, FP, FN, TP]: `[111, 157, 10, 72]`

## Verdict

- Data format effect: good. The current Phase 5 export loads through Sparsh and has mask-aware labels.
- Label effect: improved over Phase 4. Force/slip labels are simulation-valid with masks, but force is scalar-normal only and slip is contact-frame simulation slip, not real TacBench sliding-probe ground truth.
- Model effect: limited. Encoder-only checkpoints can support representation diagnostics or future probe training, but official force/slip evaluation still requires a trained task decoder checkpoint or a new probe-training run.
- Domain-gap risk: high. Sparsh's documented force/slip datasets are real GelSight/DIGIT probe trajectories with ATI force and 2 mm shear slides; Phase 5 data is Isaac/TacEx cube grasp contact. Treat results as cube-simulation diagnostics only.
