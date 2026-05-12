#!/usr/bin/env bash
set -euo pipefail
SESSION="force-slip-diag-allsource-20260513_012000"
cd /home/zjy/document/tactile-grasp
if ! git branch --show-current | grep -qx 'sparsh-force-slip'; then
  echo "ERROR: expected tactile-grasp branch sparsh-force-slip" >&2
  exit 1
fi
if [ -n "$(git status --short)" ]; then
  echo "WARNING: repo has uncommitted changes before launch:" >&2
  git status --short >&2
fi
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
wandb login --verify
test -d /vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini || { echo "Missing derived dataset /vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini" >&2; exit 1; }
for ckpt in /vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt /vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt; do
  test -f "$ckpt" || { echo "Missing checkpoint $ckpt" >&2; exit 1; }
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session $SESSION already exists; attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -n dinov2_slip_allsrc 'bash /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_dinov2_slip_allsource.sh'
tmux new-window -t "$SESSION" -n mae_slip_allsrc 'bash /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_mae_slip_allsource.sh'
tmux list-windows -t "$SESSION"
