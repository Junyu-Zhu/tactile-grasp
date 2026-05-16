#!/usr/bin/env bash
set -euo pipefail
STAMP=20260516_023557
REPO=/home/zjy/document/tactile-grasp
WORKSPACE="$REPO/sparsh-force-slip"
RUNBOOK_DIR="$WORKSPACE/runbooks/phase3_0_${STAMP}"
LOG_DIR="$WORKSPACE/logs/phase3/phase3_0_${STAMP}"
REPORT_DIR="$WORKSPACE/reports/phase3/phase3_0_${STAMP}"
mkdir -p "$RUNBOOK_DIR" "$LOG_DIR" "$REPORT_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

summary_exists() { [[ -f "$1" ]]; }

is_run_running() {
  local run_id="$1" encoder="$2"
  ps -u zjy -ww -o args= \
    | grep -E 'phase2_b_multitask.py' \
    | grep -F -- "$run_id" \
    | grep -F -- "--encoder $encoder" \
    | grep -q .
}

free_gpu() {
  local gpu pid user args busy
  for gpu in 0 1 2 3; do
    busy=0
    while read -r g pid _rest; do
      [[ "$g" == "$gpu" ]] || continue
      [[ "$pid" =~ ^[0-9]+$ ]] || continue
      user=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}') || user=""
      if [[ "$user" == "zjy" ]]; then
        args=$(ps -o args= -p "$pid" 2>/dev/null || true)
        if [[ "$args" =~ train_task.py|phase2_b_multitask.py ]]; then
          busy=1
          break
        fi
      fi
    done < <(nvidia-smi pmon -c 1 2>/dev/null | awk 'NF>=2 && $1 ~ /^[0-9]+$/ {print $1, $2, $0}')
    if [[ "$busy" == 0 ]]; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

write_multitask_runbook() {
  local encoder="$1" run_id="$2" lambda_slip="$3" gpu="$4" summary="$5" session_name="$6" out="$7"
  cat > "$out" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
SUMMARY="$summary"
if [[ -f "\$SUMMARY" ]]; then
  echo "SKIP existing summary: \$SUMMARY"
  exit 0
fi
cd "$REPO"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=$gpu
export WANDB_MODE=online
export WANDB_NAME="${run_id}_${encoder}_partially_shared_lambda${lambda_slip/./}_reroute"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \\
  --encoder $encoder \\
  --run-id "$run_id" \\
  --decoder-variant partially_shared \\
  --lambda-slip $lambda_slip \\
  --max-epochs 51 \\
  --batch-size 100 \\
  --num-workers 2 \\
  --validation-frequency 5 \\
  --wandb-mode online \\
  +trainer.devices=1
SCRIPT
  chmod +x "$out"
}

launch_multitask_if_possible() {
  local label="$1" encoder="$2" run_id="$3" lambda_slip="$4" summary="$5"
  if summary_exists "$summary"; then
    log "$label already complete: $summary"
    return 0
  fi
  if is_run_running "$run_id" "$encoder"; then
    log "$label already running"
    return 0
  fi
  local gpu
  if ! gpu=$(free_gpu); then
    log "$label waiting: no free zjy GPU"
    return 1
  fi
  local session="p30_${STAMP}_${label}_g${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    log "$label session already exists: $session"
    return 0
  fi
  local script="$RUNBOOK_DIR/reroute_${label}_gpu${gpu}.sh"
  write_multitask_runbook "$encoder" "$run_id" "$lambda_slip" "$gpu" "$summary" "$session" "$script"
  local log_file="$LOG_DIR/${session}.log"
  tmux new-session -d -s "$session" "bash '$script' 2>&1 | tee '$log_file'"
  log "launched $label on GPU $gpu as $session"
  # Give the new trainer time to initialize CUDA so subsequent queue checks do not
  # see the same GPU as free and launch multiple single-GPU jobs onto it.
  sleep 120
}

all_prereqs_done() {
  local missing=0
  for path in \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_forceonly_gsmini_${STAMP}/mae_partially_shared_multitask/training_summary.json \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_forceonly_gsmini_${STAMP}/dino_partially_shared_multitask/training_summary.json \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_forceonly_gsmini_${STAMP}/dinov2_partially_shared_multitask/training_summary.json \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_forceonly_gsmini_${STAMP}/ijepa_partially_shared_multitask/training_summary.json \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_forceonly_gsmini_${STAMP}/vjepa_partially_shared_multitask/training_summary.json \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_dino_b_lam025_gsmini_${STAMP}/dino_partially_shared_multitask/training_summary.json \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_dino_b_lam050_gsmini_${STAMP}/dino_partially_shared_multitask/training_summary.json; do
    [[ -f "$path" ]] || missing=1
  done
  find /vla1/zjy/sparsh_runs/experiments -maxdepth 1 -type f -name '___never___' >/dev/null 2>&1 || true
  find /vla1/zjy/sparsh_runs/experiments -maxdepth 1 -type d -name "*phase3_0_dino_a_gsmini_${STAMP}_dino_force" | grep -q . || missing=1
  find /vla1/zjy/sparsh_runs/experiments -maxdepth 1 -type d -name "*phase3_0_dino_a_gsmini_${STAMP}_dino_slip_allsource" | grep -q . || missing=1
  [[ "$missing" == 0 ]]
}

run_finalizer_if_possible() {
  if [[ -f "$REPORT_DIR/phase3_0_sanity_dino_abc_report.json" ]]; then
    log "final report already exists"
    return 0
  fi
  if ! all_prereqs_done; then
    return 1
  fi
  local gpu
  if ! gpu=$(free_gpu); then
    log "finalizer waiting: no free zjy GPU"
    return 1
  fi
  log "running finalizer on visible GPU $gpu"
  cd "$REPO"
  source /home/zjy/miniconda3/etc/profile.d/conda.sh
  conda activate sparsh
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONPATH=/home/zjy/document/sparsh:.
  set +e
  python sparsh-force-slip/scripts/phase3_0_finalize.py --stamp "$STAMP" --launch-c-if-needed --c-gpu "$gpu"
  rc=$?
  set -e
  log "finalizer exit code $rc"
  return 0
}

log "supervisor started"
while true; do
  launch_multitask_if_possible \
    dino_b_lam025 dino phase3_0_dino_b_lam025_gsmini_${STAMP} 0.25 \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_dino_b_lam025_gsmini_${STAMP}/dino_partially_shared_multitask/training_summary.json || true
  launch_multitask_if_possible \
    dino_b_lam050 dino phase3_0_dino_b_lam050_gsmini_${STAMP} 0.50 \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_dino_b_lam050_gsmini_${STAMP}/dino_partially_shared_multitask/training_summary.json || true
  launch_multitask_if_possible \
    dino_forceonly dino phase3_0_forceonly_gsmini_${STAMP} 0.0 \
    /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_0_forceonly_gsmini_${STAMP}/dino_partially_shared_multitask/training_summary.json || true
  run_finalizer_if_possible || true
  if [[ -f "$REPORT_DIR/phase3_0_sanity_dino_abc_report.json" ]]; then
    # If C was needed, finalizer may have launched it and final report may still say needed_not_launched.
    if grep -q 'needed_not_launched' "$REPORT_DIR/phase3_0_sanity_dino_abc_report.json" 2>/dev/null; then
      log "final report indicates C still pending; continuing"
    else
      log "supervisor complete"
      exit 0
    fi
  fi
  sleep 300
done
