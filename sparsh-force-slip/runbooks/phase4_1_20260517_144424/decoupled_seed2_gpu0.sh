#!/usr/bin/env bash
set -euo pipefail
SUMMARY="/vla1/zjy/sparsh_runs/force_slip_phase2/phase4_1_mae_decoupled_seed2_20260517_144424/mae_decoupled_multitask/training_summary.json"
if [[ -f "$SUMMARY" ]]; then
  echo "SKIP existing decoupled summary phase4_1_mae_decoupled_seed2_20260517_144424"
  exit 0
fi
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export WANDB_NAME="phase4_1_mae_decoupled_seed2_20260517_144424_mae_decoupled"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train   --encoder mae   --run-id "phase4_1_mae_decoupled_seed2_20260517_144424"   --decoder-variant decoupled   --lambda-slip 1.0   --max-epochs 51   --batch-size 100   --num-workers 2   --validation-frequency 5   --seed 44   --wandb-mode online   +trainer.devices=1
