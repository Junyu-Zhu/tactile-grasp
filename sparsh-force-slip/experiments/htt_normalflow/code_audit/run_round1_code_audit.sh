#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/zjy/miniconda3/envs/sparsh/bin/python"
SCRIPT_ROOT="/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/code_audit"
OUTPUT_ROOT="/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/code_audit"

mkdir -p "$OUTPUT_ROOT"
"$PYTHON_BIN" "$SCRIPT_ROOT/collect_server_audit.py" \
  --output-dir "$OUTPUT_ROOT" 2>&1 | tee "$OUTPUT_ROOT/collect_server_audit.log"

# GPU 0 is the only visible GPU. The verifier falls back to CPU when the
# installed torch wheel does not contain kernels for the GPU architecture.
CUDA_VISIBLE_DEVICES=0 XFORMERS_DISABLED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONPATH="/home/zjy/document/sparsh:/home/zjy/document/tactile-grasp" \
  "$PYTHON_BIN" "$SCRIPT_ROOT/verify_legacy_forward.py" \
  --device auto \
  --output "$OUTPUT_ROOT/legacy_htt_forward.json" \
  2>&1 | tee "$OUTPUT_ROOT/legacy_htt_forward.log"

echo "Round-1 code audit complete: $OUTPUT_ROOT"
