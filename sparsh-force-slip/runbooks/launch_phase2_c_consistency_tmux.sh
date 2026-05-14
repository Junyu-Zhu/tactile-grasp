#!/usr/bin/env bash
set -euo pipefail
SESSION="${SESSION:-force-slip-phase2-c-consistency}"
STAMP="${PHASE2_C_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${PHASE2_C_RUN_ID:-phase2_c_consistency_gsmini_${STAMP}}"
LOG_DIR="/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/${RUN_ID}"
DINOV2_GPU="${DINOV2_GPU:-2}"
MAE_GPU="${MAE_GPU:-3}"
DINOV2_LAMBDA_SLIP="${DINOV2_LAMBDA_SLIP:-0.20}"
MAE_LAMBDA_SLIP="${MAE_LAMBDA_SLIP:-0.50}"
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists" >&2
  exit 1
fi
launch_one() {
  local window="$1"; shift
  local script="$1"; shift
  local gpu="$1"; shift
  local lambda_slip="$1"; shift
  local log_file="${LOG_DIR}/${window}_${RUN_ID}_gpu${gpu}.log"
  local cmd="set -euo pipefail; cd /home/zjy/document/tactile-grasp/sparsh-force-slip; echo RUN_ID=${RUN_ID} WINDOW=${window} GPU=${gpu} LAMBDA_SLIP=${lambda_slip}; bash ${script} ${RUN_ID} ${gpu} ${lambda_slip} 2>&1 | tee ${log_file}; echo PHASE2_C_DONE ${RUN_ID} ${window}"
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux new-session -d -s "${SESSION}" -n "${window}"
  else
    tmux new-window -t "${SESSION}" -n "${window}"
  fi
  tmux send-keys -t "${SESSION}:${window}" "${cmd}" C-m
}
launch_one dinov2_c runbooks/run_phase2_c_consistency_dinov2.sh "${DINOV2_GPU}" "${DINOV2_LAMBDA_SLIP}"
launch_one mae_c runbooks/run_phase2_c_consistency_mae.sh "${MAE_GPU}" "${MAE_LAMBDA_SLIP}"
cat <<MSG
Launched ${SESSION}
RUN_ID=${RUN_ID}
DINOv2: GPU ${DINOV2_GPU}, lambda_slip=${DINOV2_LAMBDA_SLIP}
MAE:    GPU ${MAE_GPU}, lambda_slip=${MAE_LAMBDA_SLIP}
Logs: ${LOG_DIR}
After training finishes, evaluate with e.g.:
  python scripts/phase2_b_multitask.py report-c --c-run-id ${RUN_ID} --b-run-id phase2_b_ps_lam025_gsmini_20260514_063052 --b-run-id-mae phase2_b_ps_lam050_gsmini_20260514_063052 --b-run-id-dinov2 <best_dinov2_B_run> --reference-run-id phase2_b_ps_lam025_gsmini_20260514_063052 --encoders dinov2 mae --batch-size 100 --num-workers 2 --refresh-eval
MSG
