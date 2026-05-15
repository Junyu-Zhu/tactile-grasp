#!/usr/bin/env bash
set -euo pipefail

REPO=/home/zjy/document/tactile-grasp
SPARSH=/home/zjy/document/sparsh
WORKSPACE="$REPO/sparsh-force-slip"
PHASE1_RUN_ID=$(cat "$WORKSPACE/phase1_run_id.txt")
DERIVED=/vla1/zjy/sparsh_runs/force_slip_phase1/${PHASE1_RUN_ID}/derived_gsmini
STAMP=${1:-$(date +%Y%m%d_%H%M%S)}
RUN_ID="phase3_0_forceonly_gsmini_${STAMP}"
DINO_A_PREFIX="phase3_0_dino_a_gsmini_${STAMP}"
DINO_B_L025="phase3_0_dino_b_lam025_gsmini_${STAMP}"
DINO_B_L050="phase3_0_dino_b_lam050_gsmini_${STAMP}"
RUNBOOK_DIR="$WORKSPACE/runbooks/phase3_0_${STAMP}"
LOG_DIR="$WORKSPACE/logs/phase3/phase3_0_${STAMP}"
REPORT_DIR="$WORKSPACE/reports/phase3/phase3_0_${STAMP}"
mkdir -p "$RUNBOOK_DIR" "$LOG_DIR" "$REPORT_DIR"

cd "$REPO"
if [[ "$(git branch --show-current)" != "sparsh-force-slip" ]]; then
  echo "ERROR: expected sparsh-force-slip branch" >&2
  exit 1
fi

cat > "$REPORT_DIR/preflight.txt" <<PREFLIGHT
stamp=${STAMP}
repo=${REPO}
branch=$(git branch --show-current)
status=$(git status --short --branch)
derived=${DERIVED}
run_id=${RUN_ID}
dino_a_prefix=${DINO_A_PREFIX}
dino_b_l025=${DINO_B_L025}
dino_b_l050=${DINO_B_L050}
PREFLIGHT
{
  echo '--- nvidia-smi ---'
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo '--- tmux ---'
  tmux ls 2>/dev/null || true
  echo '--- zjy training procs ---'
  ps -u zjy -f | grep -E 'train_task.py|phase2_b_multitask.py|phase3' | grep -v grep || true
} >> "$REPORT_DIR/preflight.txt"

gpu_has_zjy_train() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v g="$gpu" '$1==g && $2 ~ /^[0-9]+$/ {print $2}' | while read -r pid; do
    [[ -z "$pid" ]] && continue
    ps -o user= -p "$pid" 2>/dev/null | grep -qx zjy && ps -o args= -p "$pid" 2>/dev/null | grep -Eq 'train_task.py|phase2_b_multitask.py' && return 0
  done
  return 1
}
for gpu in 0 1 2 3; do
  if gpu_has_zjy_train "$gpu"; then
    echo "ERROR: zjy training process already detected on GPU $gpu" >&2
    exit 1
  fi
done

COMMON_ENV='source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh'
TRAIN_LIST='["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]'
VAL_LIST='["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]'

write_forceonly() {
  local encoder="$1" gpu="$2" out="$3"
  cat > "$out" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
$COMMON_ENV
export CUDA_VISIBLE_DEVICES=$gpu
export WANDB_MODE=online
export WANDB_NAME="${RUN_ID}_${encoder}_forceonly_lambda0"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \\
  --encoder $encoder \\
  --run-id "$RUN_ID" \\
  --decoder-variant partially_shared \\
  --lambda-slip 0.0 \\
  --max-epochs 51 \\
  --batch-size 100 \\
  --num-workers 2 \\
  --validation-frequency 5 \\
  --wandb-mode online \\
  +trainer.devices=1
SCRIPT
  chmod +x "$out"
}

write_dino_a() {
  local task="$1" gpu="$2" out="$3"
  local exp="${DINO_A_PREFIX}_dino_${task}"
  local config="experiment/downstream_task/${task}/gelsight_dino"
  if [[ "$task" == "slip" ]]; then exp="${DINO_A_PREFIX}_dino_slip_allsource"; fi
  cat > "$out" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "$SPARSH"
$COMMON_ENV
export CUDA_VISIBLE_DEVICES=$gpu
export WANDB_MODE=online
export WANDB_NAME="$exp"
export PYTHONPATH=.
python train_task.py --config-name=$config \\
  paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 \\
  ssl_model_size=base ssl_name=dino \\
  experiment_name="$exp" \\
  task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dino-base/dino_vitbase.ckpt \\
  data.dataset.config.path_dataset="$DERIVED" \\
  'data.dataset.config.list_datasets=$TRAIN_LIST' \\
  'data.dataset.config.list_datasets_test=$VAL_LIST' \\
  'test.data.dataset_name=$VAL_LIST' \\
  'data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]'
SCRIPT
  chmod +x "$out"
}

write_dino_b() {
  local lam="$1" gpu="$2" runid="$3" out="$4"
  local suffix="${lam/./}"
  cat > "$out" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
$COMMON_ENV
export CUDA_VISIBLE_DEVICES=$gpu
export WANDB_MODE=online
export WANDB_NAME="${runid}_dino_partially_shared_lambda${suffix}"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \\
  --encoder dino \\
  --run-id "$runid" \\
  --decoder-variant partially_shared \\
  --lambda-slip $lam \\
  --max-epochs 51 \\
  --batch-size 100 \\
  --num-workers 2 \\
  --validation-frequency 5 \\
  --wandb-mode online \\
  +trainer.devices=1
SCRIPT
  chmod +x "$out"
}

# Individual run scripts.
write_forceonly mae 0 "$RUNBOOK_DIR/run_forceonly_mae_gpu0.sh"
write_forceonly dinov2 1 "$RUNBOOK_DIR/run_forceonly_dinov2_gpu1.sh"
write_forceonly ijepa 2 "$RUNBOOK_DIR/run_forceonly_ijepa_gpu2.sh"
write_forceonly vjepa 3 "$RUNBOOK_DIR/run_forceonly_vjepa_gpu3.sh"
write_forceonly dino 3 "$RUNBOOK_DIR/run_forceonly_dino_gpu3.sh"
write_dino_a force 0 "$RUNBOOK_DIR/run_dino_a_force_gpu0.sh"
write_dino_a slip 1 "$RUNBOOK_DIR/run_dino_a_slip_gpu1.sh"
write_dino_b 0.25 2 "$DINO_B_L025" "$RUNBOOK_DIR/run_dino_b_lam025_gpu2.sh"
write_dino_b 0.50 3 "$DINO_B_L050" "$RUNBOOK_DIR/run_dino_b_lam050_gpu3.sh"

# Per-GPU queues: keep exactly one zjy training command per GPU.
cat > "$RUNBOOK_DIR/queue_gpu0.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
"$RUNBOOK_DIR/run_forceonly_mae_gpu0.sh"
"$RUNBOOK_DIR/run_dino_a_force_gpu0.sh"
SCRIPT
cat > "$RUNBOOK_DIR/queue_gpu1.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
"$RUNBOOK_DIR/run_forceonly_dinov2_gpu1.sh"
"$RUNBOOK_DIR/run_dino_a_slip_gpu1.sh"
SCRIPT
cat > "$RUNBOOK_DIR/queue_gpu2.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
"$RUNBOOK_DIR/run_forceonly_ijepa_gpu2.sh"
"$RUNBOOK_DIR/run_dino_b_lam025_gpu2.sh"
SCRIPT
cat > "$RUNBOOK_DIR/queue_gpu3.sh" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
"$RUNBOOK_DIR/run_forceonly_vjepa_gpu3.sh"
"$RUNBOOK_DIR/run_dino_b_lam050_gpu3.sh"
"$RUNBOOK_DIR/run_forceonly_dino_gpu3.sh"
SCRIPT
chmod +x "$RUNBOOK_DIR"/queue_gpu*.sh

python3 - <<PY
import json, pathlib
payload = {
  "stamp": "$STAMP",
  "phase": "P3-0 sanity + DINO ABC launch",
  "run_id_forceonly": "$RUN_ID",
  "dino_a_prefix": "$DINO_A_PREFIX",
  "dino_b_run_ids": {"0.25": "$DINO_B_L025", "0.50": "$DINO_B_L050"},
  "derived_dataset": "$DERIVED",
  "runbook_dir": "$RUNBOOK_DIR",
  "log_dir": "$LOG_DIR",
  "queues": {
    "gpu0": ["forceonly_mae", "dino_a_force"],
    "gpu1": ["forceonly_dinov2", "dino_a_slip"],
    "gpu2": ["forceonly_ijepa", "dino_b_lambda025"],
    "gpu3": ["forceonly_vjepa", "dino_b_lambda050", "forceonly_dino"]
  },
  "single_process_single_gpu": True,
  "wandb_mode": "online",
  "auto_push_pull": False
}
path = pathlib.Path("$REPORT_DIR/launch_manifest.json")
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
md = pathlib.Path("$REPORT_DIR/launch_manifest.md")
md.write_text("# Phase3-0 launch manifest\n\n" + json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

for gpu in 0 1 2 3; do
  session="p30_${STAMP}_g${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "ERROR: tmux session $session already exists" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" "bash '$RUNBOOK_DIR/queue_gpu${gpu}.sh' 2>&1 | tee '$LOG_DIR/${session}.log'"
  echo "launched $session"
done

cat > "$WORKSPACE/reports/phase3/current_phase3_0.md" <<CUR
# Current Phase3-0

- stamp: \`$STAMP\`
- force-only run_id: \`$RUN_ID\`
- DINO A prefix: \`$DINO_A_PREFIX\`
- DINO B λ=0.25: \`$DINO_B_L025\`
- DINO B λ=0.50: \`$DINO_B_L050\`
- runbook_dir: \`$RUNBOOK_DIR\`
- log_dir: \`$LOG_DIR\`
- report_dir: \`$REPORT_DIR\`
- tmux sessions: \`p30_${STAMP}_g0\`, \`p30_${STAMP}_g1\`, \`p30_${STAMP}_g2\`, \`p30_${STAMP}_g3\`
CUR

echo "$STAMP" > "$WORKSPACE/reports/phase3/current_phase3_0_stamp.txt"
echo "Phase3-0 launch prepared and tmux sessions started."
