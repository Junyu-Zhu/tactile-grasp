#!/usr/bin/env bash
set -euo pipefail
SESSION="${SESSION:-force-slip-phase2-b}"
RUN_ID="${PHASE2_B_RUN_ID:-phase2_b_gsmini_$(date +%Y%m%d_%H%M%S)}"
GPU="${CUDA_DEVICE_ID:-1}"
LOG_DIR="/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/${RUN_ID}"
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists" >&2
  exit 1
fi
tmux new-session -d -s "${SESSION}" -n b_single_gpu
CMD="set -euo pipefail; export PHASE2_B_RUN_ID=${RUN_ID}; export CUDA_DEVICE_ID=${GPU}; cd /home/zjy/document/tactile-grasp/sparsh-force-slip; echo RUN_ID=${RUN_ID} GPU=${GPU}; bash runbooks/run_phase2_b_multitask_dinov2.sh ${RUN_ID} ${GPU} 2>&1 | tee ${LOG_DIR}/dinov2_b_shared_multitask.log; bash runbooks/run_phase2_b_multitask_mae.sh ${RUN_ID} ${GPU} 2>&1 | tee ${LOG_DIR}/mae_b_shared_multitask.log; source /home/zjy/miniconda3/etc/profile.d/conda.sh; conda activate sparsh; export CUDA_VISIBLE_DEVICES=${GPU}; export PYTHONPATH=/home/zjy/document/tactile-grasp/sparsh-force-slip:/home/zjy/document/sparsh:\${PYTHONPATH:-}; python scripts/phase2_b_multitask.py report --run-id ${RUN_ID} --encoders dinov2 mae --batch-size 100 --num-workers 2 --refresh-eval 2>&1 | tee ${LOG_DIR}/phase2_b_report.log; echo PHASE2_B_DONE ${RUN_ID}"
tmux send-keys -t "${SESSION}:b_single_gpu" "${CMD}" C-m
echo "Launched ${SESSION} with RUN_ID=${RUN_ID} on GPU=${GPU}"
echo "Logs: ${LOG_DIR}"
echo "Attach: tmux attach -t ${SESSION}"
