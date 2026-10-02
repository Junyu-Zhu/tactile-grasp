#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PYTHON_BIN="/home/zjy/miniconda3/envs/sparsh/bin/python"
SCRIPT_DIR="/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow/round2_events"
OUTPUT_DIR="/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round2/events"
MANIFEST="/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/splits.json"

mkdir -p "${OUTPUT_DIR}"
{
  "${PYTHON_BIN}" "${SCRIPT_DIR}/test_event_support.py"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/collect_event_support.py" \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_DIR}"
} | tee "${OUTPUT_DIR}/stdout.log"
