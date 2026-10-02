# Force-conditioned tactile slip and contact-state prediction

This directory contains research source code built on Sparsh tactile representations:
current force estimation, current static/gross slip detection, force-conditioned
fusion, limited encoder fine-tuning, and short-horizon contact-force prediction.

## Source layout

- `scripts/`: original force/slip and future-head implementations and analysis tools.
- `runbooks/*.sh`: historical launcher source; inspect arguments and paths before use.
- `experiments/htt_normalflow/`: later implementations, preserving cross-round imports.
- `experiments/htt_normalflow/round22_921_g1_joint_frozen/`: frozen visual,
  concatenation and FiLM detectors (`train_frozen.py`), contact-force-change
  prediction (`train_f1.py`), and limited fine-tuning implementation (`train_e3.py`).
- `experiments/htt_normalflow/round23_921_g2_force_aux_finetune/`: fine-tuning queue,
  prediction, evaluation and diagnostic code.

The `experiments` directory is not ignored wholesale: it contains real source code.
Generated reports, metrics, predictions, images, run inventories, checkpoints,
training logs, local pipeline plans and delivery copies are excluded from Git.
Existing excluded files remain on the owner's filesystem.

## External dependencies and inputs

The upstream Sparsh implementation is a separate dependency:
https://github.com/facebookresearch/sparsh

The original environment used Python 3.9 and PyTorch 2.7.0 with CUDA 12.8.
Other imports include torchvision, NumPy, SciPy, scikit-learn, Matplotlib,
Pillow and OpenCV. This is an environment record, not a verified portable lockfile.
The attention configuration used `XFORMERS_DISABLED=1`.

Datasets, pretrained weights, trained checkpoints and extracted feature caches
must be obtained separately under their respective terms. This repository does
not redistribute HTT, NormalFlow, ToucHD or Sparsh datasets or pretrained weights.
Robot/scene asset files elsewhere in the parent repository are not trained neural
model checkpoints and are outside this component's publication cleanup.

## Reproduction boundary

This is a research-source archive, not yet a standalone training distribution.
Some scripts retain historical machine paths and import modules from earlier
rounds. Preserve the directory hierarchy; do not flatten the Python files.

Training and evaluation scripts may require generated JSON protocols, data-role
splits, normalization records, input identities, run inventories and dispatch
permissions. These experiment records are deliberately excluded. Source scripts
that create them are retained where available. Reproduction requires preparing
compatible inputs and configurations and reviewing the original entry point.
Do not bypass provenance checks, replace data roles, or assume a missing file is
optional. Reusable public example configurations should be developed separately
from the private execution records, with a fresh small-scale validation.

Original force-slip weights and the upstream Sparsh repository have independent
provenance. Changes made to a local Sparsh checkout are not automatically included
here. A portable release must also document those changes and compatible versions.

## Scientific scope

The experiments concern development-set comparisons. Force estimates and force
changes use the target conventions of each experiment; they are not universally
interchangeable across sensors. Future-force prediction does not by itself establish
early slip warning, drop prediction, or a full action-conditioned world model.

See `PUBLICATION.md` for publication and Git-history handling.
