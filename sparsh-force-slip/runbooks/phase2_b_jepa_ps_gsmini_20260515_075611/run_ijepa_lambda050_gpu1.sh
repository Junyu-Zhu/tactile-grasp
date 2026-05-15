#!/usr/bin/env bash
set -uo pipefail
LOG=/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase2/phase2_b_jepa_ps_gsmini_20260515_075611/p2b_jepa_20260515_075611_ijepa_l050_g1.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
start_ts=$(date --iso-8601=seconds)
echo "[launch] session=p2b_jepa_20260515_075611_ijepa_l050_g1 encoder=ijepa lambda_slip=0.50 gpu=1 run_id=phase2_b_jepa_lam050_gsmini_20260515_075611 start=${start_ts}"
echo "[launch] host=$(hostname) user=$(whoami) pwd=$(pwd)"
nvidia-smi
cd /home/zjy/document/tactile-grasp/sparsh-force-slip
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=online
export WANDB_NAME=phase2_b_jepa_lam050_gsmini_20260515_075611_ijepa_partially_shared_lambda050
export PYTHONPATH=/home/zjy/document/tactile-grasp/sparsh-force-slip:/home/zjy/document/sparsh:${PYTHONPATH:-}
python scripts/phase2_b_multitask.py train \
  --encoder ijepa \
  --run-id phase2_b_jepa_lam050_gsmini_20260515_075611 \
  --decoder-variant partially_shared \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --lambda-slip 0.50 \
  --slip-horizon 0 \
  --log-every-steps 50 \
  --wandb-mode online \
  +trainer.devices=1
status=$?
end_ts=$(date --iso-8601=seconds)
echo "[launch] session=p2b_jepa_20260515_075611_ijepa_l050_g1 exit_status=${status} end=${end_ts}"
nvidia-smi
exit $status
