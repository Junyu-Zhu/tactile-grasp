#!/usr/bin/env bash
set -uo pipefail
OUT=/vla1/zjy/sparsh_runs/force_slip_htt_normalflow/round2
CODE=/home/zjy/document/tactile-grasp/sparsh-force-slip/experiments/htt_normalflow
PY=/home/zjy/miniconda3/envs/sparsh/bin/python
mkdir -p "$OUT/cache"
exec 9>"$OUT/cache_runner.lock"
flock -n 9 || exit 75
printf '%s\n' "$$" > "$OUT/cache_runner.pid"
rm -f "$OUT/cache_runner.exit" "$OUT/cache_verification.exit"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 XFORMERS_DISABLED=1
job_pids=()
for shard in 0 1 2 3; do
  "$PY" -u "$CODE/round2_cache/build_cache.py" --threads 8 --batch-size 4 --shards 4 --shard "$shard" > "$OUT/cache_shard${shard}.log" 2>&1 &
  job_pids+=("$!")
done
printf '%s\n' "${job_pids[@]}" > "$OUT/cache_worker.pids"
cache_status=0
for job_pid in "${job_pids[@]}"; do
  wait "$job_pid" || cache_status=1
done
printf '%s\n' "$cache_status" > "$OUT/cache_runner.exit"
if [ "$cache_status" -ne 0 ]; then exit "$cache_status"; fi
"$PY" "$CODE/round2_cache/merge_shards.py" --root "$OUT/cache" > "$OUT/cache_merge.log" 2>&1 || exit 1
"$PY" "$CODE/round2_cache/verify_cache.py" --manifest "$OUT/../round1/splits.json" --index "$OUT/cache/index.json" --output "$OUT/cache_verification.json" > "$OUT/cache_verification.log" 2>&1
verify_status=$?
printf '%s\n' "$verify_status" > "$OUT/cache_verification.exit"
exit "$verify_status"
