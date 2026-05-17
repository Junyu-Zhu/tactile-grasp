#!/usr/bin/env bash
set -euo pipefail
RUNBOOK_DIR="$1"
LOG_DIR="$2"
STAMP="$3"
DONE_FILE="$RUNBOOK_DIR/jobs.done"
RUNNING_FILE="$RUNBOOK_DIR/jobs.running"
touch "$DONE_FILE" "$RUNNING_FILE"
log() { echo "[$(date '+%F %T')] $*"; }

gpu_busy() {
  local gpu="$1" g pid user typ
  while read -r g pid typ rest; do
    [[ "$g" == "$gpu" ]] || continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    # Display-only graphics processes (type G, e.g. Xorg/gnome-shell) do not
    # reserve the card for training. Only compute-capable processes do.
    [[ "$typ" == *C* ]] || continue
    user=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}') || user=""
    # Treat any non-display compute process as busy, regardless of owner.
    if [[ "$user" != "root" && "$user" != "gdm" ]]; then
      return 0
    fi
  done < <(nvidia-smi pmon -c 1 2>/dev/null | awk 'NF>=3 && $1 ~ /^[0-9]+$/ {print $1, $2, $3, $0}')
  return 1
}

gpu_reserved() {
  local gpu="$1"
  # Sessions launched by this supervisor end with _g<gpu>. Treat them as
  # reservations even before the Python process appears in nvidia-smi.
  grep -Eq "_g${gpu}$" "$RUNNING_FILE" 2>/dev/null
}

first_free_gpu() {
  local g
  for g in 0 1 2 3; do
    if ! gpu_reserved "$g" && ! gpu_busy "$g"; then echo "$g"; return 0; fi
  done
  return 1
}

job_finished() {
  local session="$1"
  ! tmux has-session -t "$session" 2>/dev/null
}

while true; do
  # mark finished running sessions
  tmp=$(mktemp)
  while IFS=$'\t' read -r job session; do
    [[ -n "${job:-}" ]] || continue
    if job_finished "$session"; then
      grep -qxF "$job" "$DONE_FILE" || echo "$job" >> "$DONE_FILE"
      log "finished $job ($session)"
    else
      printf '%s\t%s\n' "$job" "$session" >> "$tmp"
    fi
  done < "$RUNNING_FILE"
  mv "$tmp" "$RUNNING_FILE"

  all_done=true
  while IFS=$'\t' read -r job script; do
    [[ -n "${job:-}" ]] || continue
    if grep -qxF "$job" "$DONE_FILE" || grep -q "^${job}"$'\t' "$RUNNING_FILE"; then
      continue
    fi
    all_done=false
    gpu="$(first_free_gpu || true)"
    if [[ -n "$gpu" ]]; then
      tmp_script="$RUNBOOK_DIR/${job}_gpu${gpu}.sh"
      sed "s/__GPU__/${gpu}/g" "$script" > "$tmp_script"
      chmod +x "$tmp_script"
      session="p41_${STAMP}_${job}_g${gpu}"
      tmux new-session -d -s "$session" "bash '$tmp_script' 2>&1 | tee '$LOG_DIR/${session}.log'"
      printf '%s\t%s\n' "$job" "$session" >> "$RUNNING_FILE"
      log "launched $job on GPU $gpu as $session"
      sleep 10
    else
      log "no free GPU; waiting"
      break
    fi
  done < "$RUNBOOK_DIR/jobs.tsv"

  # if every job is marked done and no sessions remain, exit
  pending=0
  while IFS=$'\t' read -r job script; do
    [[ -n "${job:-}" ]] || continue
    if ! grep -qxF "$job" "$DONE_FILE"; then pending=$((pending+1)); fi
  done < "$RUNBOOK_DIR/jobs.tsv"
  running_count=$(wc -l < "$RUNNING_FILE" | tr -d ' ')
  if [[ "$pending" -eq 0 && "$running_count" -eq 0 ]]; then
    log "all Phase4-1 jobs done"
    exit 0
  fi
  sleep 300
done
