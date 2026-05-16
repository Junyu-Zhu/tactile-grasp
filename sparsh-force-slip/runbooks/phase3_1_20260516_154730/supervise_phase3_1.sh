#!/usr/bin/env bash
set -euo pipefail
STAMP="20260516_154730"
REPO=/home/zjy/document/tactile-grasp
WORKSPACE="$REPO/sparsh-force-slip"
RUN_ID="phase3_1_decoupled_gsmini_${STAMP}"
RUNBOOK_DIR="$WORKSPACE/runbooks/phase3_1_${STAMP}"
LOG_DIR="$WORKSPACE/logs/phase3/phase3_1_${STAMP}"
REPORT_DIR="$WORKSPACE/reports/phase3/phase3_1_${STAMP}"
PHASE2_ROOT=/vla1/zjy/sparsh_runs/force_slip_phase2
log() { echo "[$(date '+%F %T')] $*"; }
summary_path() { echo "$PHASE2_ROOT/$RUN_ID/${1}_decoupled_multitask/training_summary.json"; }
is_encoder_running() {
  local encoder="$1"
  ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "$RUN_ID" | grep -F -- "--encoder $encoder" | grep -F -- "--decoder-variant decoupled" | grep -q .
}
zjy_gpu_busy() {
  local gpu="$1" pid user args
  while read -r g pid _rest; do
    [[ "$g" == "$gpu" ]] || continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    user=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}') || user=""
    if [[ "$user" == "zjy" ]]; then
      args=$(ps -o args= -p "$pid" 2>/dev/null || true)
      if [[ "$args" =~ train_task.py|phase2_b_multitask.py|phase3_1_finalize.py ]]; then
        return 0
      fi
    fi
  done < <(nvidia-smi pmon -c 1 2>/dev/null | awk 'NF>=2 && $1 ~ /^[0-9]+$/ {print $1, $2, $0}')
  return 1
}
first_free_zjy_gpu() { for gpu in 0 1 2 3; do if ! zjy_gpu_busy "$gpu"; then echo "$gpu"; return 0; fi; done; return 1; }
write_train_script() {
  local encoder="$1" gpu="$2" script="$3" summary
  summary=$(summary_path "$encoder")
  cat > "$script" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
SUMMARY="$summary"
if [[ -f "\$SUMMARY" ]]; then echo "SKIP existing summary: \$SUMMARY"; exit 0; fi
if ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "$RUN_ID" | grep -F -- "--encoder $encoder" | grep -F -- "--decoder-variant decoupled" | grep -q .; then echo "SKIP already running"; exit 0; fi
cd "$REPO"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=$gpu
export WANDB_MODE=online
export WANDB_NAME="${RUN_ID}_${encoder}_decoupled"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train --encoder $encoder --run-id "$RUN_ID" --decoder-variant decoupled --lambda-slip 1.0 --max-epochs 51 --batch-size 100 --num-workers 2 --validation-frequency 5 --wandb-mode online +trainer.devices=1
SCRIPT
  chmod +x "$script"
}
launch_train() {
  local encoder="$1"
  local gpu="$2"
  local session="p31_${STAMP}_${encoder}_g${gpu}"
  local summary script log_file
  summary=$(summary_path "$encoder")
  if [[ -f "$summary" ]]; then log "$encoder already complete"; return 0; fi
  if is_encoder_running "$encoder"; then log "$encoder already running"; return 0; fi
  if zjy_gpu_busy "$gpu"; then log "GPU $gpu busy with zjy process; wait for $encoder"; return 1; fi
  if tmux has-session -t "$session" 2>/dev/null; then log "$encoder session exists"; return 0; fi
  script="$RUNBOOK_DIR/run_${encoder}_gpu${gpu}.sh"
  log_file="$LOG_DIR/${session}.log"
  write_train_script "$encoder" "$gpu" "$script"
  tmux new-session -d -s "$session" "bash '$script' 2>&1 | tee '$log_file'"
  log "launched $encoder on GPU $gpu as $session"
  sleep 60
}
all_training_done() { for enc in mae dino dinov2 ijepa vjepa; do [[ -f "$(summary_path "$enc")" ]] || return 1; done; return 0; }
run_finalizer() {
  if [[ -f "$REPORT_DIR/phase3_1_decoupled_multitask_report.json" ]]; then log "report already exists"; return 0; fi
  all_training_done || return 1
  local gpu
  if ! gpu=$(first_free_zjy_gpu); then log "finalizer waits: no free zjy GPU"; return 1; fi
  log "running Phase3-1 finalizer on visible GPU $gpu"
  cd "$REPO"
  source /home/zjy/miniconda3/etc/profile.d/conda.sh
  conda activate sparsh
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONPATH=/home/zjy/document/sparsh:.
  python sparsh-force-slip/scripts/phase3_1_finalize.py --run-id "$RUN_ID" --stamp "$STAMP" --batch-size 100 --num-workers 2
}
log "Phase3-1 supervisor started for $RUN_ID"
while true; do
  if [[ ! -f "$(summary_path vjepa)" ]] && ! is_encoder_running vjepa; then
    if gpu=$(first_free_zjy_gpu); then launch_train vjepa "$gpu" || true; else log "vjepa waits: no free zjy GPU"; fi
  fi
  if all_training_done; then
    run_finalizer && { log "Phase3-1 supervisor complete"; exit 0; }
  fi
  sleep 300
done
