#!/usr/bin/env bash
set -uo pipefail
OUT=/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round2
CODE=/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow
PY=/home/zjy/miniconda3/envs/sparsh/bin/python
exec 9>"$OUT/postprocess.lock"
flock -n 9 || exit 75
runner_pid=$(cat "$OUT/cache_runner.pid")
printf '%s\n' "$$" > "$OUT/postprocess.pid"
while kill -0 "$runner_pid" 2>/dev/null; do sleep 15; done
if [ "$(cat "$OUT/cache_runner.exit" 2>/dev/null)" != 0 ] || [ "$(cat "$OUT/cache_verification.exit" 2>/dev/null)" != 0 ]; then
  printf '%s\n' 'cache prerequisite failed' > "$OUT/postprocess.failed"
  exit 1
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 XFORMERS_DISABLED=1
"$PY" "$CODE/round2_cache/verify_future_cache.py" --index "$OUT/cache/index.json" --output "$OUT/future_reconstruction_verification.json" > "$OUT/future_reconstruction_verification.log" 2>&1
future_status=$?
if [ "$future_status" -ne 0 ]; then printf '%s\n' "$future_status" > "$OUT/postprocess.exit"; exit "$future_status"; fi
"$PY" "$CODE/round2_metrics/evaluate.py" --index "$OUT/cache/index.json" --splits "$OUT/../round1/splits.json" --output "$OUT/metrics" > "$OUT/metrics.log" 2>&1
metrics_status=$?
printf '%s\n' "$metrics_status" > "$OUT/postprocess.exit"
exit "$metrics_status"
