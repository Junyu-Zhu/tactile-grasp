#!/usr/bin/env bash
set -euo pipefail
STAMP="20260528_000000"
REPO="/home/zjy/document/tactile-grasp"
WORK="$REPO/sparsh-force-slip"
RUNBOOK="$WORK/runbooks/phase_lambda_architecture_future_ablation_${STAMP}"
LOGDIR="$WORK/logs/phase_lambda_architecture_future_ablation/${STAMP}"
REPORTDIR="$WORK/reports/phase_lambda_architecture_future_ablation/${STAMP}"
MANIFEST="$REPORTDIR/architecture_future_command_manifest.json"
PHASEB_JSON="$WORK/reports/phase_lambda_decoupled_future_ablation/${STAMP}/phase_lambda_decoupled_future_ablation_report.json"
SUPLOG="$LOGDIR/supervisor.log"
mkdir -p "$LOGDIR" "$REPORTDIR"
cd "$REPO"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUPLOG"; }
phaseb_running(){ pgrep -u zjy -af "phase_lambda_future_ablation.py --stamp ${STAMP}" >/dev/null 2>&1; }
ensure_phaseb(){
  if [[ -f "$PHASEB_JSON" ]]; then return 0; fi
  if phaseb_running; then return 0; fi
  if tmux has-session -t "pld_${STAMP}_phase_b_retry" 2>/dev/null; then return 0; fi
  log "Phase B report missing and no active process; restarting fixed Phase B on GPU0 without touching completed artifacts"
  tmux new-session -d -s "pld_${STAMP}_phase_b_retry" "cd '$REPO' && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONPATH=. && python sparsh-force-slip/scripts/phase_lambda_future_ablation.py --stamp ${STAMP} --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --wandb-mode offline 2>&1 | tee -a '$LOGDIR/phase_b_retry.log'"
}
gpu_free(){
  local gpu="$1" used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | head -1 | tr -d ' ')
  [[ "$used" -lt 1000 ]]
}
launch_task(){
  local label="$1"
  local gpu="$2"
  local session="arch_${STAMP}_${label}"
  local report script
  report=$(python3 - "$MANIFEST" "$label" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); label=sys.argv[2]
print(next(t['report_json'] for t in m['tasks'] if t['label']==label))
PY
)
  script="$RUNBOOK/run_${label}.sh"
  if [[ -f "$report" ]]; then log "DONE existing $label"; return 0; fi
  if tmux has-session -t "$session" 2>/dev/null; then log "RUNNING $label in $session"; return 0; fi
  if ! gpu_free "$gpu"; then log "WAIT $label: GPU$gpu not free"; return 0; fi
  log "LAUNCH $label on GPU$gpu -> $session"
  tmux new-session -d -s "$session" "bash '$script'"
}
all_done(){
  python3 - "$MANIFEST" "$PHASEB_JSON" <<'PY'
import json,sys,os
m=json.load(open(sys.argv[1])); phaseb=sys.argv[2]
missing=[]
if not os.path.exists(phaseb): missing.append(phaseb)
for t in m['tasks']:
    if not os.path.exists(t['report_json']): missing.append(t['report_json'])
if missing:
    print('missing:')
    print('\n'.join(missing))
    sys.exit(1)
PY
}
finalize_and_commit(){
  log "Finalizing architecture future ablation"
  source /home/zjy/miniconda3/etc/profile.d/conda.sh
  conda activate sparsh
  python sparsh-force-slip/scripts/phase_lambda_architecture_future_ablation.py finalize --stamp "$STAMP" 2>&1 | tee -a "$SUPLOG"
  python -m py_compile sparsh-force-slip/scripts/phase_lambda_future_ablation.py sparsh-force-slip/scripts/phase_lambda_architecture_future_ablation.py
  git add \
    sparsh-force-slip/scripts/phase_lambda_future_ablation.py \
    sparsh-force-slip/scripts/phase_lambda_architecture_future_ablation.py \
    sparsh-force-slip/runbooks/phase_lambda_future_ablation_${STAMP}/ \
    sparsh-force-slip/runbooks/phase_lambda_architecture_future_ablation_${STAMP}/ \
    sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/ \
    sparsh-force-slip/reports/phase_lambda_decoupled_summary.md \
    sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/ \
    sparsh-force-slip/logs/phase_lambda_architecture_future_ablation/ || true
  if git diff --cached --quiet; then
    log "No staged changes to commit"
  else
    git commit -m "Compare force-slip interfaces for future instability prediction" -m "The lambda sweep selected a force-preserving decoupled interface, but the paper needs direct evidence that consistency and partially shared alternatives do not provide a better Stage-II future-prediction interface. This supplement fixes the future head input to full dynamics and varies only Stage-I architecture while reusing Phase B Decoupled λ=0.10 as the main method." -m "Constraint: Current Phase B was not stopped; Decoupled λ=0.10 full dynamics is reused rather than retrained.\nConstraint: Each supplemental future head used one visible GPU; W&B ran in offline fallback because api.wandb.ai was unreachable at launch, with deterministic run ids for later sync.\nConfidence: medium\nScope-risk: moderate\nDirective: Use this report as the architecture-to-future bridge, not as a replacement for the current-task Table2 ablation.\nTested: py_compile for phase scripts and report finalizer; report existence checked by supervisor.\nNot-tested: Multi-seed variance for the future heads." 2>&1 | tee -a "$SUPLOG" || true
  fi
}
log "Supervisor started; 30-minute cadence after initial launch"
while true; do
  ensure_phaseb
  launch_task decoupled_lam100_old 1
  launch_task consistency_decoder 2
  launch_task partially_shared_lam025 3
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | tee -a "$SUPLOG" >/dev/null
  tmux ls 2>/dev/null | grep -E "pld_${STAMP}|arch_${STAMP}" | tee -a "$SUPLOG" >/dev/null || true
  if all_done >>"$SUPLOG" 2>&1; then
    finalize_and_commit
    log "All tasks complete"
    break
  fi
  log "Not complete yet; sleeping 1800s"
  sleep 1800
done
