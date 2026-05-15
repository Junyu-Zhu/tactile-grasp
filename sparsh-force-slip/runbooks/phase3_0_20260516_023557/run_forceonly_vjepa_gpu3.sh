#!/usr/bin/env bash
set -euo pipefail
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=online
export WANDB_NAME="phase3_0_forceonly_gsmini_20260516_023557_vjepa_forceonly_lambda0"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder vjepa \
  --run-id "phase3_0_forceonly_gsmini_20260516_023557" \
  --decoder-variant partially_shared \
  --lambda-slip 0.0 \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --wandb-mode online \
  +trainer.devices=1
