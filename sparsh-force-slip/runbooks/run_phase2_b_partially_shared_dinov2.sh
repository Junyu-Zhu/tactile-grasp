#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-${PHASE2_B_RUN_ID:-phase2_b_ps_gsmini_$(date +%Y%m%d_%H%M%S)}}"
GPU_LIST="${2:-${CUDA_DEVICE_LIST:-0,1}}"
BATCH_SIZE="${BATCH_SIZE:-200}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-25}"
cd /home/zjy/document/tactile-grasp/sparsh-force-slip
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_NAME="${RUN_ID}_dinov2_b_partially_shared_multitask"
export PYTHONPATH="/home/zjy/document/tactile-grasp/sparsh-force-slip:/home/zjy/document/sparsh:${PYTHONPATH:-}"
python scripts/phase2_b_multitask.py train \
  --encoder dinov2 \
  --run-id "${RUN_ID}" \
  --decoder-variant partially_shared \
  --data-parallel \
  --max-epochs 51 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 2 \
  --validation-frequency 5 \
  --lambda-slip 1.0 \
  --slip-horizon 0 \
  --log-every-steps "${LOG_EVERY_STEPS}" \
  --wandb-mode "${WANDB_MODE}"
