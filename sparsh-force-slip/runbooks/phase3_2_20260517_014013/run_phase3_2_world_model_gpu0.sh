#!/usr/bin/env bash
set -euo pipefail
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export WANDB_NAME="phase3_2_world_model_mae_20260517_014013_world_model"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase3_2_world_model.py run-all \
  --phase3-1-report "/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase3/phase3_1_20260516_154730/phase3_1_decoupled_multitask_report.json" \
  --stamp "20260517_014013" \
  --run-id "phase3_2_world_model_mae_20260517_014013" \
  --horizons 1 3 5 \
  --precompute-batch-size 128 \
  --train-batch-size 1024 \
  --num-workers 2 \
  --max-epochs 40 \
  --wandb-mode online
