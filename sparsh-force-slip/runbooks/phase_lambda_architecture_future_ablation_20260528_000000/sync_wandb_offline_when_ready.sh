#!/usr/bin/env bash
set -euo pipefail
REPO="/home/zjy/document/tactile-grasp"
SYNC_SCRIPT="$REPO/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/wandb_offline_sync_commands.sh"
LOG="$REPO/sparsh-force-slip/logs/phase_lambda_architecture_future_ablation/20260528_000000/wandb_sync_retry.log"
DONE="$REPO/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/wandb_sync_complete.txt"
cd "$REPO"
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
can_reach_wandb(){
  timeout 8 python3 - <<'PY' >/dev/null 2>&1
import socket
s=socket.create_connection(("api.wandb.ai",443), timeout=5)
s.close()
PY
}
log "W&B offline sync supervisor started"
while true; do
  if [[ -f "$DONE" ]]; then log "already synced: $DONE"; exit 0; fi
  if ! can_reach_wandb; then
    log "api.wandb.ai:443 unreachable; sleeping 1800s"
    sleep 1800
    continue
  fi
  log "api.wandb.ai reachable; starting wandb sync"
  source /home/zjy/miniconda3/etc/profile.d/conda.sh
  conda activate sparsh
  if bash "$SYNC_SCRIPT" 2>&1 | tee -a "$LOG"; then
    date '+%F %T' > "$DONE"
    log "wandb sync complete"
    exit 0
  fi
  log "wandb sync failed; sleeping 1800s"
  sleep 1800
done
