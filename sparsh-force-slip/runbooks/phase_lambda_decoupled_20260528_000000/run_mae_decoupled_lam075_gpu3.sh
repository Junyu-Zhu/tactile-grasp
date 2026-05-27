#!/usr/bin/env bash
set -euo pipefail
SUMMARY="/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam075_20260528_000000/mae_decoupled_multitask/training_summary.json"
RUN_ID="phase_lambda_decoupled_mae_lam075_20260528_000000"
LAM="0.75"
TAG="lam075"
if [[ -f "$SUMMARY" ]]; then
  echo "SKIP existing summary: $SUMMARY"
  exit 0
fi
if ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "$RUN_ID" | grep -F -- "--encoder mae" | grep -F -- "--decoder-variant decoupled" | grep -q .; then
  echo "SKIP already running: $RUN_ID"
  exit 0
fi
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=online
export WANDB_NAME="${RUN_ID}_lambda_${TAG}"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder mae \
  --run-id "$RUN_ID" \
  --decoder-variant decoupled \
  --lambda-slip "$LAM" \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --wandb-mode online \
  +trainer.devices=1
