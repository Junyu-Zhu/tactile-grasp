#!/usr/bin/env bash
set -euo pipefail
SESSION="${SESSION:-force-slip-phase2-b-loss}"
STAMP="${PHASE2_B_LOSS_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID_L05="${PHASE2_B_LOSS_RUN_ID_L05:-phase2_b_ps_lam050_gsmini_${STAMP}}"
RUN_ID_L025="${PHASE2_B_LOSS_RUN_ID_L025:-phase2_b_ps_lam025_gsmini_${STAMP}}"
LOG_DIR="/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/loss_weight_sweep_${STAMP}"
BATCH_SIZE="${BATCH_SIZE:-100}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists" >&2
  exit 1
fi
run_cmd() {
  local window="$1"; shift
  local encoder="$1"; shift
  local run_id="$1"; shift
  local gpu="$1"; shift
  local lambda_slip="$1"; shift
  local log_file="${LOG_DIR}/${encoder}_${run_id}_gpu${gpu}.log"
  local cmd="set -euo pipefail; cd /home/zjy/document/tactile-grasp/sparsh-force-slip; source /home/zjy/miniconda3/etc/profile.d/conda.sh; conda activate sparsh; export CUDA_VISIBLE_DEVICES=${gpu}; export WANDB_MODE=online; export WANDB_NAME=${run_id}_${encoder}_b_partially_shared_lambda${lambda_slip}; export PYTHONPATH=/home/zjy/document/tactile-grasp/sparsh-force-slip:/home/zjy/document/sparsh:\${PYTHONPATH:-}; echo RUN_ID=${run_id} ENCODER=${encoder} GPU=${gpu} LAMBDA_SLIP=${lambda_slip}; python scripts/phase2_b_multitask.py train --encoder ${encoder} --run-id ${run_id} --decoder-variant partially_shared --max-epochs 51 --batch-size ${BATCH_SIZE} --num-workers 2 --validation-frequency 5 --lambda-slip ${lambda_slip} --slip-horizon 0 --log-every-steps ${LOG_EVERY_STEPS} --wandb-mode online 2>&1 | tee ${log_file}; echo LOSS_SWEEP_DONE ${run_id} ${encoder} lambda=${lambda_slip}"
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -n "${window}"
  else
    tmux new-window -t "${SESSION}" -n "${window}"
  fi
  tmux send-keys -t "${SESSION}:${window}" "${cmd}" C-m
}
run_cmd dinov2_l05 dinov2 "${RUN_ID_L05}" 0 0.5
run_cmd dinov2_l025 dinov2 "${RUN_ID_L025}" 1 0.25
run_cmd mae_l05 mae "${RUN_ID_L05}" 2 0.5
run_cmd mae_l025 mae "${RUN_ID_L025}" 3 0.25
cat <<MSG
Launched ${SESSION}
Loss scheme 1: RUN_ID=${RUN_ID_L05}, lambda_slip=0.5 (dinov2 GPU0, mae GPU2)
Loss scheme 2: RUN_ID=${RUN_ID_L025}, lambda_slip=0.25 (dinov2 GPU1, mae GPU3)
Logs: ${LOG_DIR}
After all finish, run both reports:
  source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && cd /home/zjy/document/tactile-grasp/sparsh-force-slip
  python scripts/phase2_b_multitask.py report --run-id ${RUN_ID_L05} --decoder-variant partially_shared --encoders dinov2 mae --batch-size 100 --num-workers 2 --refresh-eval
  python scripts/phase2_b_multitask.py report --run-id ${RUN_ID_L025} --decoder-variant partially_shared --encoders dinov2 mae --batch-size 100 --num-workers 2 --refresh-eval
MSG
