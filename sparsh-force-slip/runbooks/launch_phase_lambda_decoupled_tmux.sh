#!/usr/bin/env bash
set -euo pipefail

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
REPO=/home/zjy/document/tactile-grasp
WORKSPACE="$REPO/sparsh-force-slip"
PHASE2_ROOT=/vla1/zjy/sparsh_runs/force_slip_phase2
REPORT_DIR="$WORKSPACE/reports/phase_lambda_decoupled/${STAMP}"
RUNBOOK_DIR="$WORKSPACE/runbooks/phase_lambda_decoupled_${STAMP}"
LOG_DIR="$WORKSPACE/logs/phase_lambda_decoupled/${STAMP}"
mkdir -p "$REPORT_DIR" "$RUNBOOK_DIR" "$LOG_DIR"

LAMBDAS=(0.10 0.25 0.50 0.75)
LAMBDA_TAGS=(lam010 lam025 lam050 lam075)
GPUS=(0 1 2 3)

log() { echo "[$(date '+%F %T')] $*"; }
run_id_for() { echo "phase_lambda_decoupled_mae_${2}_${STAMP}"; }
summary_for() { echo "$PHASE2_ROOT/$(run_id_for "$1" "$2")/mae_decoupled_multitask/training_summary.json"; }

write_train_script() {
  local lam="$1" tag="$2" gpu="$3" script="$4"
  local run_id summary
  run_id="$(run_id_for "$lam" "$tag")"
  summary="$(summary_for "$lam" "$tag")"
  cat > "$script" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
SUMMARY="$summary"
RUN_ID="$run_id"
LAM="$lam"
TAG="$tag"
if [[ -f "\$SUMMARY" ]]; then
  echo "SKIP existing summary: \$SUMMARY"
  exit 0
fi
if ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "\$RUN_ID" | grep -F -- "--encoder mae" | grep -F -- "--decoder-variant decoupled" | grep -q .; then
  echo "SKIP already running: \$RUN_ID"
  exit 0
fi
cd "$REPO"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=$gpu
export WANDB_MODE=online
export WANDB_NAME="\${RUN_ID}_lambda_\${TAG}"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \\
  --encoder mae \\
  --run-id "\$RUN_ID" \\
  --decoder-variant decoupled \\
  --lambda-slip "\$LAM" \\
  --max-epochs 51 \\
  --batch-size 100 \\
  --num-workers 2 \\
  --validation-frequency 5 \\
  --wandb-mode online \\
  +trainer.devices=1
SCRIPT
  chmod +x "$script"
}

python3 - <<PY
import json, pathlib, subprocess
payload = {
  "stamp": "$STAMP",
  "repo": "$REPO",
  "branch": subprocess.check_output(["git","branch","--show-current"], cwd="$REPO", text=True).strip(),
  "head": subprocess.check_output(["git","rev-parse","--short","HEAD"], cwd="$REPO", text=True).strip(),
  "phase": "phase_lambda_decoupled",
  "encoder": "mae",
  "decoder_variant": "decoupled",
  "lambda_slip": [0.10,0.25,0.50,0.75],
  "max_epochs": 51,
  "batch_size": 100,
  "num_workers": 2,
  "validation_frequency": 5,
  "single_process_single_gpu": True,
  "wandb_mode": "online",
  "auto_push_pull": False,
  "report_dir": "$REPORT_DIR",
  "runbook_dir": "$RUNBOOK_DIR",
  "log_dir": "$LOG_DIR",
}
pathlib.Path("$REPORT_DIR/launch_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

for i in "${!LAMBDAS[@]}"; do
  lam="${LAMBDAS[$i]}"; tag="${LAMBDA_TAGS[$i]}"; gpu="${GPUS[$i]}"
  run_id="$(run_id_for "$lam" "$tag")"
  summary="$(summary_for "$lam" "$tag")"
  session="pld_${STAMP}_${tag}_g${gpu}"
  script="$RUNBOOK_DIR/run_mae_decoupled_${tag}_gpu${gpu}.sh"
  log_file="$LOG_DIR/${session}.log"
  if [[ -f "$summary" ]]; then
    log "skip complete $tag summary=$summary"
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    log "session already exists $session"
    continue
  fi
  write_train_script "$lam" "$tag" "$gpu" "$script"
  tmux new-session -d -s "$session" -n train
  tmux send-keys -t "$session:train" "bash $script 2>&1 | tee $log_file" C-m
  log "launched lambda=$lam tag=$tag gpu=$gpu session=$session run_id=$run_id"
  sleep 5
done

tmux ls | grep "pld_${STAMP}" || true
