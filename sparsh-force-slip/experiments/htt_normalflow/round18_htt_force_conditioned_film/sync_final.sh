#!/usr/bin/env bash
set -euo pipefail
LOCAL=/home/zjy/Documents/grasp/tactile_grasp/sparsh-force-slip/experiments/htt_normalflow/round18_htt_force_conditioned_film/
REMOTE_CODE=/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round18_htt_force_conditioned_film/
REMOTE_DELIVERY=/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round18_htt_force_conditioned_film/delivery/
rsync -a --exclude __pycache__ "$LOCAL" "zjy-4090:$REMOTE_CODE"
rsync -a --exclude __pycache__ "$LOCAL" "zjy-4090:$REMOTE_DELIVERY"
python "${LOCAL}verify_delivery.py" --root "$LOCAL"
ssh zjy-4090 "cd /home/zjy/document/tactile-grasp/sparsh-force-slip && /home/zjy/miniconda3/envs/sparsh/bin/python ${REMOTE_CODE}verify_delivery.py --root ${REMOTE_CODE} && /home/zjy/miniconda3/envs/sparsh/bin/python ${REMOTE_CODE}verify_delivery.py --root ${REMOTE_DELIVERY}"
