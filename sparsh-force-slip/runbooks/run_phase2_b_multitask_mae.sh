#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-${PHASE2_B_RUN_ID:-phase2_b_gsmini_$(date +%Y%m%d_%H%M%S)}}"
GPU="${2:-${CUDA_DEVICE_ID:-1}}"
cd /home/zjy/document/tactile-grasp/sparsh-force-slip
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_NAME="${RUN_ID}_mae_b_shared_multitask"
export PYTHONPATH="/home/zjy/document/tactile-grasp/sparsh-force-slip:/home/zjy/document/sparsh:${PYTHONPATH:-}"
python scripts/phase2_b_multitask.py train \
  --encoder mae \
  --run-id "${RUN_ID}" \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --lambda-slip 1.0 \
  --slip-horizon 0 \
  --wandb-mode "${WANDB_MODE}"
