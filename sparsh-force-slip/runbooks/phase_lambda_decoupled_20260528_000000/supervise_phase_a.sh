#!/usr/bin/env bash
set -euo pipefail
STAMP="20260528_000000"
REPO=/home/zjy/document/tactile-grasp
WORKSPACE="$REPO/sparsh-force-slip"
PHASE2_ROOT=/vla1/zjy/sparsh_runs/force_slip_phase2
RUN_IDS=(
  phase_lambda_decoupled_mae_lam010_${STAMP}
  phase_lambda_decoupled_mae_lam025_${STAMP}
  phase_lambda_decoupled_mae_lam050_${STAMP}
  phase_lambda_decoupled_mae_lam075_${STAMP}
)
log(){ echo "[$(date '+%F %T')] $*"; }
summary(){ echo "$PHASE2_ROOT/$1/mae_decoupled_multitask/training_summary.json"; }
any_running(){
  for rid in "${RUN_IDS[@]}"; do
    ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "$rid" | grep -q . && return 0
  done
  return 1
}
all_done(){
  for rid in "${RUN_IDS[@]}"; do [[ -f "$(summary "$rid")" ]] || return 1; done
  return 0
}
cd "$REPO"
log "supervisor started for Phase A stamp=$STAMP"
while ! all_done; do
  missing=()
  for rid in "${RUN_IDS[@]}"; do [[ -f "$(summary "$rid")" ]] || missing+=("$rid"); done
  log "waiting for ${#missing[@]} runs: ${missing[*]}"
  if ! any_running; then
    log "ERROR: no training process is running but summaries are missing"
    exit 2
  fi
  sleep 300
done
log "all Phase A runs completed; finalizing report"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase_lambda_decoupled.py phase-a --stamp "$STAMP" --batch-size 100 --num-workers 2
log "Phase A report generated; committing"
git add sparsh-force-slip/scripts/phase_lambda_decoupled.py \
        sparsh-force-slip/runbooks/launch_phase_lambda_decoupled_tmux.sh \
        sparsh-force-slip/runbooks/phase_lambda_decoupled_${STAMP} \
        sparsh-force-slip/reports/phase_lambda_decoupled
if git diff --cached --quiet; then
  log "no staged changes; skipping commit"
else
  git commit -m "Select a decoupled loss weight for future tactile prediction" \
    -m "The MAE decoupled multitask head previously used only lambda_slip=1.0, leaving uncertainty about whether the low slip metrics were a weighting artifact. This phase runs four smaller lambda settings in parallel, evaluates them against the separate probing baseline and the existing lambda=1.0 reference, and records the best Stage-I checkpoint for the next future-ablation phase." \
    -m "Constraint: Training remains server-local on sparsh-force-slip with manual push/pull only" \
    -m "Constraint: Each sweep run uses one visible GPU and W&B online logging" \
    -m "Confidence: medium" \
    -m "Scope-risk: moderate" \
    -m "Directive: Use the selected checkpoint in reports/phase_lambda_decoupled/current_phase_lambda_decoupled.md for the following future causal-input ablation" \
    -m "Tested: Four MAE decoupled lambda trainings completed; phase-a finalizer evaluated validation metrics and wrote JSON/Markdown reports" \
    -m "Not-tested: Phase B future ablation is not included in this commit"
fi
log "Phase A supervisor complete"
