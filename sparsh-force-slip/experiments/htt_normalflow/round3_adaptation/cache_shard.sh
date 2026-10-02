#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?GPU index required}
export CUDA_VISIBLE_DEVICES="$gpu"
export XFORMERS_DISABLED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
root=/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round3_adaptation
out=/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round3_mae_slip_adaptation
exec /home/zjy/miniconda3/envs/sparsh/bin/python "$root/cache.py" build \
  --manifest /vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json \
  --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth \
  --output-dir "$out/cache" --device cuda:0 --batch-size 16 --shards 3 --shard "$gpu"
