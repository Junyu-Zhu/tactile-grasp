#!/usr/bin/env bash
set -euo pipefail
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=online
export WANDB_NAME="phase3_0_dino_b_lam050_gsmini_20260516_023557_dino_partially_shared_lambda050"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder dino \
  --run-id "phase3_0_dino_b_lam050_gsmini_20260516_023557" \
  --decoder-variant partially_shared \
  --lambda-slip 0.50 \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --wandb-mode online \
  +trainer.devices=1
