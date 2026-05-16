#!/usr/bin/env bash
set -euo pipefail

REPORT_PATH="${1:-}"
STAMP="${2:-$(date +%Y%m%d_%H%M%S)}"
REPO=/home/zjy/document/tactile-grasp
WORKSPACE="$REPO/sparsh-force-slip"
RUNBOOK_DIR="$WORKSPACE/runbooks/phase3_2_${STAMP}"
LOG_DIR="$WORKSPACE/logs/phase3/phase3_2_${STAMP}"
REPORT_DIR="$WORKSPACE/reports/phase3/phase3_2_${STAMP}"
mkdir -p "$RUNBOOK_DIR" "$LOG_DIR" "$REPORT_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

if [[ -z "$REPORT_PATH" ]]; then
  if [[ -f "$WORKSPACE/reports/phase3/current_phase3_1_report.md" ]]; then
    REPORT_PATH=$(grep -o '/home/zjy/document/tactile-grasp[^`]*phase3_1_decoupled_multitask_report.md' "$WORKSPACE/reports/phase3/current_phase3_1_report.md" | head -1 || true)
    REPORT_PATH="${REPORT_PATH%.md}.json"
  fi
fi
if [[ -z "$REPORT_PATH" || ! -f "$REPORT_PATH" ]]; then
  echo "Usage: $0 /path/to/phase3_1_decoupled_multitask_report.json [stamp]" >&2
  echo "Phase3-2 is gated on a completed Phase3-1 report." >&2
  exit 2
fi

RUN_ID=$(python3 - <<PY
import json
from pathlib import Path
report=json.loads(Path('$REPORT_PATH').read_text())
enc=report['best_backbone']['encoder']
print(f'phase3_2_world_model_{enc}_${STAMP}')
PY
)

zjy_gpu_busy() {
  local gpu="$1" pid user args
  while read -r g pid _rest; do
    [[ "$g" == "$gpu" ]] || continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    user=$(ps -o user= -p "$pid" 2>/dev/null | awk '{print $1}') || user=""
    if [[ "$user" == "zjy" ]]; then
      args=$(ps -o args= -p "$pid" 2>/dev/null || true)
      if [[ "$args" =~ phase2_b_multitask.py|phase3_1_finalize.py|phase3_2_world_model.py|train_task.py ]]; then
        return 0
      fi
    fi
  done < <(nvidia-smi pmon -c 1 2>/dev/null | awk 'NF>=2 && $1 ~ /^[0-9]+$/ {print $1, $2, $0}')
  return 1
}

first_free_zjy_gpu() {
  local gpu
  for gpu in 0 1 2 3; do
    if ! zjy_gpu_busy "$gpu"; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

if ps -u zjy -ww -o args= | grep -F -- "phase3_2_world_model.py run-all" | grep -F -- "$REPORT_PATH" | grep -q .; then
  log "Phase3-2 already running for $REPORT_PATH"
  exit 0
fi

GPU="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU" ]]; then
  if ! GPU=$(first_free_zjy_gpu); then
    log "No free zjy GPU for Phase3-2 launch"
    exit 1
  fi
fi

cat > "$REPORT_DIR/launch_manifest.json" <<JSON
{
  "stamp": "$STAMP",
  "run_id": "$RUN_ID",
  "phase3_1_report": "$REPORT_PATH",
  "repo": "$REPO",
  "branch": "sparsh-force-slip",
  "single_process_single_gpu": true,
  "auto_push_pull": false,
  "wandb_mode": "online",
  "created_at": "$(date -Is)",
  "visible_gpu": "$GPU"
}
JSON
nvidia-smi > "$REPORT_DIR/gpu_snapshot_at_launch.txt" 2>&1 || true
tmux ls > "$REPORT_DIR/tmux_snapshot_at_launch.txt" 2>&1 || true
ps -u zjy -ww -o pid,etime,pcpu,pmem,args > "$REPORT_DIR/zjy_process_snapshot_at_launch.txt" 2>&1 || true

RUN_SCRIPT="$RUNBOOK_DIR/run_phase3_2_world_model_gpu${GPU}.sh"
cat > "$RUN_SCRIPT" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=$GPU
export WANDB_MODE=online
export WANDB_NAME="${RUN_ID}_world_model"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase3_2_world_model.py run-all \\
  --phase3-1-report "$REPORT_PATH" \\
  --stamp "$STAMP" \\
  --run-id "$RUN_ID" \\
  --horizons 1 3 5 \\
  --precompute-batch-size 128 \\
  --train-batch-size 1024 \\
  --num-workers 2 \\
  --max-epochs 40 \\
  --wandb-mode online
SCRIPT
chmod +x "$RUN_SCRIPT"

SESSION="p32_${STAMP}_world_g${GPU}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  log "tmux session already exists: $SESSION"
else
  tmux new-session -d -s "$SESSION" "bash '$RUN_SCRIPT' 2>&1 | tee '$LOG_DIR/${SESSION}.log'"
  log "launched Phase3-2 $RUN_ID on GPU $GPU as $SESSION"
fi
log "REPORT_DIR=$REPORT_DIR"
