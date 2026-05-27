#!/usr/bin/env bash
set -euo pipefail
STAMP="20260528_000000"
REPO="/home/zjy/document/tactile-grasp"
PHASE_A_JSON="$REPO/sparsh-force-slip/reports/phase_lambda_decoupled/$STAMP/phase_lambda_decoupled_report.json"
LOGDIR="$REPO/sparsh-force-slip/logs/phase_lambda_decoupled_future_ablation/$STAMP"
mkdir -p "$LOGDIR"
cd "$REPO"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/supervise_phase_b.log"; }
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
log "Phase B supervisor started; waiting for Phase A report and commit boundary"
while [ ! -f "$PHASE_A_JSON" ]; do
  log "waiting for $PHASE_A_JSON"
  sleep 300
done
while ps -u zjy -ww -o args | grep -E "phase_lambda_decoupled_mae_lam(010|025|050|075)_$STAMP|phase_lambda_decoupled.py phase-a --stamp $STAMP" | grep -v grep >/dev/null; do
  log "Phase A training/finalizer still active"
  sleep 300
done
# Wait for the Phase-A supervisor to exit if it is still committing.
while tmux has-session -t "pld_${STAMP}_supervisor" 2>/dev/null; do
  log "waiting for Phase A supervisor tmux to finish"
  sleep 120
done
# Fallback Phase-A commit if the earlier supervisor produced reports but did not commit them.
git add \
  sparsh-force-slip/scripts/phase_lambda_decoupled.py \
  sparsh-force-slip/runbooks/launch_phase_lambda_decoupled_tmux.sh \
  sparsh-force-slip/runbooks/phase_lambda_decoupled_${STAMP} \
  sparsh-force-slip/reports/phase_lambda_decoupled || true
if ! git diff --cached --quiet; then
  log "committing Phase A fallback artifacts"
  git commit -m "Select a decoupled loss weight for future tactile prediction" -m "The MAE decoupled multitask head previously used only lambda_slip=1.0, leaving uncertainty about whether the low slip metrics were a weighting artifact. This phase runs four smaller lambda settings in parallel, evaluates them against the separate probing baseline and the existing lambda=1.0 reference, and records the best Stage-I checkpoint for the next future-ablation phase.

Constraint: Training remains server-local on sparsh-force-slip with manual push/pull only
Constraint: Each sweep run uses one visible GPU and W&B online logging
Confidence: medium
Scope-risk: moderate
Directive: Use the selected checkpoint in reports/phase_lambda_decoupled/current_phase_lambda_decoupled.md for the following future causal-input ablation
Tested: Four MAE decoupled lambda trainings completed; phase-a finalizer evaluated validation metrics and wrote JSON/Markdown reports
Not-tested: Phase B future ablation is not included in this commit"
else
  git reset -q
  log "Phase A artifacts already committed or unchanged"
fi
log "starting Phase B on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" WANDB_MODE=online \
  python sparsh-force-slip/scripts/phase_lambda_future_ablation.py \
    --stamp "$STAMP" \
    --horizons 1 3 5 \
    --precompute-batch-size 128 \
    --train-batch-size 1024 \
    --num-workers 2 \
    --max-epochs 40 \
    --wandb-mode online \
  2>&1 | tee "$LOGDIR/phase_b_run.log"
log "Phase B finished; committing artifacts"
git add \
  sparsh-force-slip/scripts/phase_lambda_future_ablation.py \
  sparsh-force-slip/runbooks/phase_lambda_future_ablation_${STAMP} \
  sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation \
  sparsh-force-slip/reports/phase_lambda_decoupled_summary.md
if ! git diff --cached --quiet; then
  git commit -m "Compare the best decoupled lambda on future instability prediction" -m "The lambda sweep selects a Stage-I force-slip interface, but the paper-facing claim also depends on whether that interface improves future contact-instability prediction. This phase reuses the selected checkpoint, runs the same causal input ablations as the prior Phase4/5 flow, includes the friction-risk q condition when supported, and writes a consolidated replacement-vs-ablation recommendation.

Constraint: Work remains server-local on sparsh-force-slip with manual push/pull only
Constraint: Stage-II heads use one visible GPU and W&B online logging
Confidence: medium
Scope-risk: moderate
Directive: Compare this report against the old lambda=1.0 Phase4/5 reports before replacing paper tables
Tested: Best-lambda future ablation completed; consolidated Markdown/JSON summary generated
Not-tested: Real-robot grasp success is outside this phase"
else
  git reset -q
  log "No Phase B git changes to commit"
fi
log "done"
