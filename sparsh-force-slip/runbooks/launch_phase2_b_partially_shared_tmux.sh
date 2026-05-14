#!/usr/bin/env bash
set -euo pipefail
SESSION="${SESSION:-force-slip-phase2-b-ps}"
RUN_ID="${PHASE2_B_RUN_ID:-phase2_b_ps_gsmini_$(date +%Y%m%d_%H%M%S)}"
DINO_GPUS="${DINO_GPU_LIST:-0,1}"
MAE_GPUS="${MAE_GPU_LIST:-2,3}"
LOG_DIR="/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/${RUN_ID}"
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists" >&2
  exit 1
fi
tmux new-session -d -s "${SESSION}" -n dinov2_ps
DINO_CMD="set -euo pipefail; export PHASE2_B_RUN_ID=${RUN_ID}; export CUDA_DEVICE_LIST=${DINO_GPUS}; cd /home/zjy/document/tactile-grasp/sparsh-force-slip; echo RUN_ID=${RUN_ID} DINO_GPUS=${DINO_GPUS}; bash runbooks/run_phase2_b_partially_shared_dinov2.sh ${RUN_ID} ${DINO_GPUS} 2>&1 | tee ${LOG_DIR}/dinov2_b_partially_shared.log; echo DINO_PS_DONE ${RUN_ID}"
tmux send-keys -t "${SESSION}:dinov2_ps" "${DINO_CMD}" C-m
tmux new-window -t "${SESSION}" -n mae_ps
MAE_CMD="set -euo pipefail; export PHASE2_B_RUN_ID=${RUN_ID}; export CUDA_DEVICE_LIST=${MAE_GPUS}; cd /home/zjy/document/tactile-grasp/sparsh-force-slip; echo RUN_ID=${RUN_ID} MAE_GPUS=${MAE_GPUS}; bash runbooks/run_phase2_b_partially_shared_mae.sh ${RUN_ID} ${MAE_GPUS} 2>&1 | tee ${LOG_DIR}/mae_b_partially_shared.log; echo MAE_PS_DONE ${RUN_ID}"
tmux send-keys -t "${SESSION}:mae_ps" "${MAE_CMD}" C-m
echo "Launched ${SESSION} with RUN_ID=${RUN_ID}"
echo "DINOv2 GPUs: ${DINO_GPUS}; MAE GPUs: ${MAE_GPUS}"
echo "Logs: ${LOG_DIR}"
echo "Attach: tmux attach -t ${SESSION}"
echo "After both finish, run: source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && cd /home/zjy/document/tactile-grasp/sparsh-force-slip && python scripts/phase2_b_multitask.py report --run-id ${RUN_ID} --decoder-variant partially_shared --encoders dinov2 mae --batch-size 100 --num-workers 2 --refresh-eval"
