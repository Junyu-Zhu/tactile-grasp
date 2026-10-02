# tactile-grasp

Research code for contact-force estimation, force-conditioned slip detection, and short-horizon contact-state prediction from vision-based tactile sensing.

The repository also contains UR5 grasping simulation and tactile data collection code. This README focuses on the [`sparsh-force-slip`](sparsh-force-slip/) component.

> **Release scope:** This is a research-source archive, not a one-command reproduction package. Datasets, pretrained weights, trained checkpoints, feature caches, training logs, experiment records, and internal pipeline plans are excluded from the current source snapshot. Some entry points require protocols and manifests generated during the original experiments.

## 1. Research tasks

- **Current contact-force estimation:** Estimate three-axis contact force from tactile representations and provide predicted force to downstream models.
- **Current slip detection:** Compare a visual baseline, visual–force concatenation, feature-level FiLM, and bounded FiLM, with emphasis on the trade-off between static false positives and gross-slip recall.
- **Limited encoder fine-tuning:** Compare visual training, auxiliary supervision from current ground-truth force, FiLM conditioning, and their combination. Auxiliary force supervision is distinct from updating the original physical-force output branch.
- **Short-horizon contact-state prediction:** Use the current basic visual representation, predicted force, or both to predict three-axis force changes at horizons of 1, 5, and 10 frames.

Experimental variants are stored in separate round directories. They are controlled comparisons, not components that automatically form one deployment model. Predicting future force does not by itself establish early slip warning or drop prediction.

## 2. Repository layout

```text
tactile-grasp/
├── sparsh-force-slip/
│   ├── scripts/                 # Earlier force/slip/future training and analysis
│   ├── runbooks/                # Retained shell launchers
│   └── experiments/htt_normalflow/
│       ├── adapters.py          # Dataset adapters
│       ├── prepare_splits.py    # Data-role split preparation
│       ├── round18_htt_force_conditioned_film/
│       ├── round19_htt_partial_encoder_finetuning/
│       ├── round20_htt_contact_state_transition/
│       ├── round22_921_g1_joint_frozen/
│       ├── round23_921_g2_force_aux_finetune/
│       └── ...                  # Other experimental variants and evaluation tools
├── protac/                      # Grasping, perception, and collection modules
├── isaacsim_test/               # Simulation scripts and checks
├── environment/                 # Robot and scene assets
└── PYTHON_CODE_LAYOUT.md        # Parent-repository code layout
```

Some rounds import modules from earlier rounds. Preserve this directory hierarchy rather than copying individual training files in isolation. Python and shell files under `experiments` are source code; generated results and records in the same directories are excluded by `.gitignore`.

## 3. Datasets and their roles

| Dataset or data source | Role in this research | Scope and limitations |
|---|---|---|
| **HTT** | Supervised force adaptation, static/gross slip detection, and future contact-force-change prediction | Primarily evaluated with trial-group-separated development protocols; incipient samples are excluded from the main binary supervision |
| **Sparsh / TacBench-related Gelsight-mini data** | Earlier force-estimation and slip tasks, including `gelsight-force-estimation` and `object_slide` | Inputs, labels, and splits must be checked for each task; metrics are not directly interchangeable with HTT results |
| **ToucHD-Force** | Force-supervised pre-adaptation on the verified sensor subset, followed by transfer evaluation on HTT | Force coordinates, offsets, and errors from the two domains must not be combined without validation |
| **NormalFlow** | Earlier contact-state, frozen-feature, and motion-assisted prediction experiments | Motion or pose is not treated as a slip label; this dataset is not required by the current HTT detection pipeline |
| **ToucHD-Mani** | Data-availability assessment and a candidate for future work | Not a required input for the main training pipeline; no completed supervised slip-training result is claimed here |
| **DeformableObjectsGrasping** | Availability audit for external real-object data | A data audit is not an external-domain performance evaluation |
| **Locally collected stable/slip tactile sequences** | Limited supplementary real-object analysis | Sequences without force ground truth are not used to report physical-force accuracy |

Obtain datasets and weights from their original providers and follow their terms of use. Relevant upstream projects:

- [Sparsh](https://github.com/facebookresearch/sparsh): pretrained tactile representations and downstream-task documentation.
- [AnyTouch2 / ToucHD](https://github.com/GeWu-Lab/AnyTouch2): the ToucHD dataset family and related information.

This repository does not redistribute these datasets. The main later experiments use a Sparsh MAE encoder; historical code also includes comparisons with DINO, I-JEPA, and other encoders.

### Suggested storage layout

Keep datasets and generated artifacts outside the repository, for example:

```text
<storage>/
├── tactile_dataset/
│   ├── HTT-dataset/
│   ├── NormalFlow-dataset/
│   └── ToucHD-Force/
├── tactile_datasets/Gelsight-mini/
│   ├── gelsight-force-estimation/
│   └── object_slide/
├── sparsh_models/               # Pretrained weights
└── sparsh_runs/                 # Feature caches, checkpoints, and results
```

This layout is a recommendation, not an automatically applied configuration. Historical scripts may retain machine-specific absolute paths. Review argument defaults, configuration files, and upstream dependencies before running them.

## 4. Environment

The original training environment used **Python 3.9.25**, **PyTorch 2.7.0+cu128**, and **CUDA runtime 12.8**. These are recorded experiment versions, not a claim that every script has been validated on other machines. The GPU driver must support the installed PyTorch CUDA build.

Core Python dependencies include `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `Pillow`, and `opencv-python`, together with the separate Sparsh `tactile_ssl` package and its dependencies.

```bash
git clone https://github.com/Junyu-Zhu/tactile-grasp.git
cd tactile-grasp

# Run in an environment with compatible PyTorch and Sparsh dependencies.
export XFORMERS_DISABLED=1
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import tactile_ssl; print(tactile_ssl.__file__)"
```

Follow the official Sparsh installation instructions. The original server also had local changes to Sparsh training entry points, environment configuration, and signal handling; changes to that external checkout are not included automatically when cloning this repository. A complete portable dependency lockfile is not provided.

Isaac Sim, Isaac Lab, and robot-control dependencies apply to the corresponding simulation modules. They are not required merely to inspect or use the cached-feature head implementations.

## 5. Training and evaluation entry points

Paths below are relative to `sparsh-force-slip/experiments/htt_normalflow/`.

| Task | Entry point | Notes |
|---|---|---|
| Data adaptation and splitting | `adapters.py`, `prepare_splits.py` | Check raw fields, trials, labels, and data roles |
| Frozen-encoder detection | `round22_921_g1_joint_frozen/train_frozen.py` | V: visual; C: concatenation; M: FiLM; MB: bounded FiLM |
| Short-horizon force-change prediction | `round22_921_g1_joint_frozen/train_f1.py` | K-V, K-F, and K-VF input comparisons |
| Limited encoder fine-tuning | `round22_921_g1_joint_frozen/train_e3.py` | The G2 training implementation remains in the R22 preparation directory |
| Fine-tuning queue | `round23_921_g2_force_aux_finetune/launch_g2_queue.py` | Requires a run inventory and matching authorization records; historical paths are not portable defaults |
| Frozen detection evaluation | `round22_921_g1_joint_frozen/evaluate_frozen_detection.py` | Fixed thresholds and calibration rules |
| Fine-tuned model prediction and evaluation | `round23_921_g2_force_aux_finetune/predict_e3.py`, `round23_921_g2_force_aux_finetune/evaluate_e3.py` | Requires compatible inputs and upstream model identities |

### Workflow

1. Prepare the selected dataset, pretrained encoder, and required force/visual upstream weights.
2. Verify complete trials and leakage groups, then construct fit, selection, calibration, and validation roles with valid temporal endpoints.
3. Use the corresponding preparation code to generate feature caches, support manifests, and protocols. Fit normalization only on the permitted training data.
4. Run a small smoke check for inputs, targets, gradients, frozen parameters, and checkpoint recovery before launching the full configuration.
5. Select checkpoints using the selection role, choose thresholds using calibration, and report development evaluation on validation.

Do not fabricate missing protocols, manifests, or identity hashes to bypass checks. Do not use test labels for model selection.

### Examples with prepared caches

These examples use actual CLI arguments; they are **not raw-data-to-training quick-start commands**. Set `DATA`, `SUPPORT`, and `OUT` to compatible, previously prepared paths with the required schema, fold, and seed identities. The examples omit `--formal`; formal dispatch additionally requires the locked protocol and authorization files expected by the entry point.

```bash
EXP=sparsh-force-slip/experiments/htt_normalflow/round22_921_g1_joint_frozen

# Frozen encoder: force-conditioned FiLM for current slip detection.
python "$EXP/train_frozen.py" \
  --data "$DATA" --support-inventory "$SUPPORT" \
  --output "$OUT" --group M --fold 1 --seed 20260914 --device cuda:0

# Short-horizon state prediction: visual representation plus predicted force.
# Replace DATA, SUPPORT, and OUT with this task's own cache, manifest, and output.
python "$EXP/train_f1.py" \
  --data "$DATA" --support-inventory "$SUPPORT" \
  --output "$OUT" --group K-VF --fold 1 --seed 20260914 --device cuda:0
```

A full comparison must retain all folds, seeds, and groups required by its protocol. A single example command does not reproduce the complete experiment. Original configurations, splits, run inventories, authorization records, and results are not distributed with this source archive and must be prepared separately.

## 6. Evaluation conventions

- The main current-slip task compares **static versus gross**. Incipient samples are analyzed separately and are not automatically treated as reliable early-warning labels.
- Detection metrics include pAUC, AP, observed static FPR, gross recall, balanced accuracy, macro-F1, trial-level false alarms, and event detection.
- A false-positive constraint met on calibration does not guarantee the same rate on validation.
- Short-horizon evaluation separates true force change, predicted change, current-force estimation error, and future absolute-force error, with persistence and linear baselines.
- The reference-relative, per-axis clipping to **±20 N** used in the relevant HTT experiments is a target convention. It is not a universal sensor range or a shared force definition across datasets.
- Development folds overlap, and historical upstream models have development-data exposure. These results are not independent blind tests or robot success rates.

## 7. Version control and release scope

The current force-slip snapshot tracks source code and usage documentation. Datasets, weights, training records, and pipeline plans are excluded through `.gitignore`. Files removed from Git tracking remain on their original machines.

Git history is retained at the project owner's request. **Earlier commits may still contain previously tracked logs, reports, and plans.** Ignore rules govern subsequent snapshots; they do not remove historical content.

Future commits should describe one coherent change and use a consistent subject, for example:

- `feat(force-slip): add a force-conditioned detection module`
- `fix(data): correct trial-to-frame alignment`
- `docs: update dataset and reproduction instructions`
- `chore(repo): exclude generated experiment artifacts`

Older commit subjects are preserved. Annotated release tags identify source snapshots without rewriting history. The `v0.1.0-source` tag marks the initial documented source archive, not a validated turnkey training package.

See the [component README](sparsh-force-slip/README.md) for additional context. Upstream code, datasets, and weights retain their respective licenses; this source cleanup does not assign or change those permissions.
