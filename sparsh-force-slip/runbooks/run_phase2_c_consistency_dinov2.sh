#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-${PHASE2_C_RUN_ID:-phase2_c_consistency_gsmini_$(date +%Y%m%d_%H%M%S)}}"
GPU="${2:-${CUDA_DEVICE_ID:-0}}"
LAMBDA_SLIP="${3:-${DINOV2_LAMBDA_SLIP:-0.20}}"
BETA_CONSISTENCY="${BETA_CONSISTENCY:-0.05}"
CONSISTENCY_ALPHA="${CONSISTENCY_ALPHA:-10}"
CONSISTENCY_TAU_SOURCE="${CONSISTENCY_TAU_SOURCE:-p65}"
BATCH_SIZE="${BATCH_SIZE:-100}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
cd /home/zjy/document/tactile-grasp/sparsh-force-slip
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_NAME="${RUN_ID}_dinov2_c_consistency_lam${LAMBDA_SLIP}_beta${BETA_CONSISTENCY}"
export PYTHONPATH="/home/zjy/document/tactile-grasp/sparsh-force-slip:/home/zjy/document/sparsh:${PYTHONPATH:-}"
python scripts/phase2_b_multitask.py train \
  --encoder dinov2 \
  --run-id "${RUN_ID}" \
  --decoder-variant consistency \
  --max-epochs 51 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 2 \
  --validation-frequency 5 \
  --lambda-slip "${LAMBDA_SLIP}" \
  --beta-consistency "${BETA_CONSISTENCY}" \
  --consistency-alpha "${CONSISTENCY_ALPHA}" \
  --consistency-tau-source "${CONSISTENCY_TAU_SOURCE}" \
  --slip-horizon 0 \
  --log-every-steps "${LOG_EVERY_STEPS}" \
  --wandb-mode "${WANDB_MODE}"
