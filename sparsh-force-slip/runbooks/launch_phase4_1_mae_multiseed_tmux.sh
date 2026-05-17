#!/usr/bin/env bash
set -euo pipefail

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
REPO=/home/zjy/document/tactile-grasp
SPARSH=/home/zjy/document/sparsh
WORKSPACE="$REPO/sparsh-force-slip"
RUNBOOK_DIR="$WORKSPACE/runbooks/phase4_1_${STAMP}"
LOG_DIR="$WORKSPACE/logs/phase4/phase4_1_${STAMP}"
REPORT_DIR="$WORKSPACE/reports/phase4/phase4_1_${STAMP}"
DERIVED=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini
MAE_CKPT=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt
mkdir -p "$RUNBOOK_DIR" "$LOG_DIR" "$REPORT_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

ALL_DATA='["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]'
ALL_VAL='["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]'

cat > "$REPORT_DIR/launch_manifest.json" <<JSON
{
  "stamp": "$STAMP",
  "repo": "$REPO",
  "branch": "sparsh-force-slip",
  "phase": "phase4_1_mae_multiseed",
  "seeds": {"seed1": 43, "seed2": 44},
  "single_process_single_gpu": true,
  "auto_push_pull": false,
  "wandb_mode": "online",
  "derived_dataset": "$DERIVED",
  "created_at": "$(date -Is)"
}
JSON
nvidia-smi > "$REPORT_DIR/gpu_snapshot_at_launch.txt" 2>&1 || true
tmux ls > "$REPORT_DIR/tmux_snapshot_at_launch.txt" 2>&1 || true
ps -ww -o pid,user,etime,pcpu,pmem,args > "$REPORT_DIR/process_snapshot_at_launch.txt" 2>&1 || true

make_force_script() {
  local label="$1"
  local seed="$2"
  local gpu_placeholder="$3"
  local exp="phase4_1_mae_force_${label}_${STAMP}"
  local script="$RUNBOOK_DIR/run_force_${label}.sh"
  cat > "$script" <<SCRIPT2
#!/usr/bin/env bash
set -euo pipefail
if compgen -G "/vla1/zjy/sparsh_runs/experiments/*${exp}/checkpoints/epoch-0051.pth" >/dev/null; then
  echo "SKIP existing force experiment ${exp}"
  exit 0
fi
cd "$SPARSH"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=${gpu_placeholder}
export WANDB_MODE=online
export WANDB_NAME="${exp}"
export PYTHONPATH=.
python train_task.py --config-name=experiment/downstream_task/force/gelsight_mae \
  paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 \
  ssl_model_size=base ssl_name=mae seed=${seed} experiment_name=${exp} \
  task.checkpoint_encoder="$MAE_CKPT" \
  data.dataset.config.path_dataset="$DERIVED" \
  'data.dataset.config.list_datasets=${ALL_DATA}' \
  'data.dataset.config.list_datasets_test=${ALL_VAL}' \
  'test.data.dataset_name=${ALL_VAL}'
SCRIPT2
  chmod +x "$script"
}

make_slip_script() {
  local label="$1"
  local seed="$2"
  local gpu_placeholder="$3"
  local exp="phase4_1_mae_slip_${label}_${STAMP}"
  local script="$RUNBOOK_DIR/run_slip_${label}.sh"
  cat > "$script" <<SCRIPT2
#!/usr/bin/env bash
set -euo pipefail
if compgen -G "/vla1/zjy/sparsh_runs/experiments/*${exp}/checkpoints/epoch-0051.pth" >/dev/null; then
  echo "SKIP existing slip experiment ${exp}"
  exit 0
fi
cd "$SPARSH"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=${gpu_placeholder}
export WANDB_MODE=online
export WANDB_NAME="${exp}"
export PYTHONPATH=.
python train_task.py --config-name=experiment/downstream_task/slip/gelsight_mae \
  paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 \
  ssl_model_size=base ssl_name=mae seed=${seed} experiment_name=${exp} \
  task.checkpoint_encoder="$MAE_CKPT" \
  data.dataset.config.path_dataset="$DERIVED" \
  'data.dataset.config.list_datasets=${ALL_DATA}' \
  'data.dataset.config.list_datasets_test=${ALL_VAL}' \
  'test.data.dataset_name=${ALL_VAL}' \
  'data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]'
SCRIPT2
  chmod +x "$script"
}

make_decoupled_script() {
  local label="$1"
  local seed="$2"
  local gpu_placeholder="$3"
  local run_id="phase4_1_mae_decoupled_${label}_${STAMP}"
  local script="$RUNBOOK_DIR/run_decoupled_${label}.sh"
  cat > "$script" <<SCRIPT2
#!/usr/bin/env bash
set -euo pipefail
SUMMARY="/vla1/zjy/sparsh_runs/force_slip_phase2/${run_id}/mae_decoupled_multitask/training_summary.json"
if [[ -f "\$SUMMARY" ]]; then
  echo "SKIP existing decoupled summary ${run_id}"
  exit 0
fi
cd "$REPO"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=${gpu_placeholder}
export WANDB_MODE=online
export WANDB_NAME="${run_id}_mae_decoupled"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder mae \
  --run-id "${run_id}" \
  --decoder-variant decoupled \
  --lambda-slip 1.0 \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --seed ${seed} \
  --wandb-mode online \
  +trainer.devices=1
SCRIPT2
  chmod +x "$script"
}

# scripts use __GPU__ placeholder; launch wrapper rewrites a temp copy with the assigned GPU.
make_force_script seed1 43 __GPU__
make_slip_script seed1 43 __GPU__
make_decoupled_script seed1 43 __GPU__
make_force_script seed2 44 __GPU__
make_slip_script seed2 44 __GPU__
make_decoupled_script seed2 44 __GPU__

cat > "$RUNBOOK_DIR/jobs.tsv" <<JOBS
force_seed1	$RUNBOOK_DIR/run_force_seed1.sh
slip_seed1	$RUNBOOK_DIR/run_slip_seed1.sh
decoupled_seed1	$RUNBOOK_DIR/run_decoupled_seed1.sh
force_seed2	$RUNBOOK_DIR/run_force_seed2.sh
slip_seed2	$RUNBOOK_DIR/run_slip_seed2.sh
decoupled_seed2	$RUNBOOK_DIR/run_decoupled_seed2.sh
JOBS

cat > "$RUNBOOK_DIR/supervise_phase4_1.sh" <<'SUP'
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
SUP
chmod +x "$RUNBOOK_DIR/supervise_phase4_1.sh"

SESSION="p41_${STAMP}_supervisor"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  log "supervisor already exists: $SESSION"
else
  tmux new-session -d -s "$SESSION" "bash '$RUNBOOK_DIR/supervise_phase4_1.sh' '$RUNBOOK_DIR' '$LOG_DIR' '$STAMP' 2>&1 | tee '$LOG_DIR/${SESSION}.log'"
  log "launched Phase4-1 supervisor $SESSION"
fi
log "RUNBOOK_DIR=$RUNBOOK_DIR"
log "REPORT_DIR=$REPORT_DIR"
