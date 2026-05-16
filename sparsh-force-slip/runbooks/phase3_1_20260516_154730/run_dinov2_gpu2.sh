#!/usr/bin/env bash
set -euo pipefail
SUMMARY="/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/dinov2_decoupled_multitask/training_summary.json"
if [[ -f "$SUMMARY" ]]; then
  echo "SKIP existing summary: $SUMMARY"
  exit 0
fi
if ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "phase3_1_decoupled_gsmini_20260516_154730" | grep -F -- "--encoder dinov2" | grep -F -- "--decoder-variant decoupled" | grep -q .; then
  echo "SKIP already running: phase3_1_decoupled_gsmini_20260516_154730 dinov2 decoupled"
  exit 0
fi
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=online
export WANDB_NAME="phase3_1_decoupled_gsmini_20260516_154730_dinov2_decoupled"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder dinov2 \
  --run-id "phase3_1_decoupled_gsmini_20260516_154730" \
  --decoder-variant decoupled \
  --lambda-slip 1.0 \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --wandb-mode online \
  +trainer.devices=1
