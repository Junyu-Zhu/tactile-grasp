#!/usr/bin/env bash
set -euo pipefail
SESSION="${SESSION:-force-slip-dino-lambda-probe}"
STAMP="${PHASE2_B_DINO_LAMBDA_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LAMBDA_A="${LAMBDA_A:-0.20}"
LAMBDA_B="${LAMBDA_B:-0.15}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
LOG_DIR="/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/dinov2_lambda_probe_${STAMP}"
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists" >&2
  exit 1
fi
run_one() {
  local window="$1"; shift
  local gpu="$1"; shift
  local lambda_slip="$1"; shift
  local tag="${lambda_slip/./}"
  local run_id="phase2_b_ps_dino_lam${tag}_gsmini_${STAMP}"
  local log_file="${LOG_DIR}/dinov2_${run_id}_gpu${gpu}.log"
  local cmd="set -euo pipefail; cd /home/zjy/document/tactile-grasp/sparsh-force-slip; source /home/zjy/miniconda3/etc/profile.d/conda.sh; conda activate sparsh; export CUDA_VISIBLE_DEVICES=${gpu}; export WANDB_MODE=online; export WANDB_NAME=${run_id}_dinov2_b_partially_shared_lambda${lambda_slip}; export PYTHONPATH=/home/zjy/document/tactile-grasp/sparsh-force-slip:/home/zjy/document/sparsh:\${PYTHONPATH:-}; echo RUN_ID=${run_id} GPU=${gpu} LAMBDA_SLIP=${lambda_slip}; python scripts/phase2_b_multitask.py train --encoder dinov2 --run-id ${run_id} --decoder-variant partially_shared --max-epochs 51 --batch-size 100 --num-workers 2 --validation-frequency 5 --lambda-slip ${lambda_slip} --slip-horizon 0 --log-every-steps 50 --wandb-mode online 2>&1 | tee ${log_file}; echo DINO_LAMBDA_DONE ${run_id} lambda=${lambda_slip}"
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -n "${window}"
  else
    tmux new-window -t "${SESSION}" -n "${window}"
  fi
  tmux send-keys -t "${SESSION}:${window}" "${cmd}" C-m
  echo "${run_id} ${gpu} ${lambda_slip} ${log_file}"
}
run_one dino_lam_a "${GPU_A}" "${LAMBDA_A}"
run_one dino_lam_b "${GPU_B}" "${LAMBDA_B}"
echo "Launched ${SESSION}; logs: ${LOG_DIR}"
