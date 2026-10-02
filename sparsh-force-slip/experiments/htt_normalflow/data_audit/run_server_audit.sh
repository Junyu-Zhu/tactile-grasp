#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/zjy/miniconda3/envs/sparsh/bin/python
HTT_ROOT=/vla1/zjy/tactile_dataset/HTT-dataset
NORMALFLOW_ROOT=/vla1/zjy/tactile_dataset/NormalFlow-dataset
OUTPUT_DIR=/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round1/data_audit
LOG_FILE="${OUTPUT_DIR}/audit_stdout.log"

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_datasets.py" \
  --htt-root "${HTT_ROOT}" \
  --normalflow-zip "${NORMALFLOW_ROOT}/dataset.zip" \
  --normalflow-extract-dir "${NORMALFLOW_ROOT}/extracted_round1" \
  --output-dir "${OUTPUT_DIR}" 2>&1 | tee "${LOG_FILE}.tmp"
mv "${LOG_FILE}.tmp" "${LOG_FILE}"
