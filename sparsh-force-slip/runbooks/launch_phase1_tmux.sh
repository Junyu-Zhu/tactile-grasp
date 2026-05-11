#!/usr/bin/env bash
set -euo pipefail
SESSION="force-slip-phase1"
cd /home/zjy/document/tactile-grasp
git status --short --branch
if ! git branch --show-current | grep -qx 'sparsh-force-slip'; then
  echo "ERROR: expected tactile-grasp branch sparsh-force-slip" >&2
  exit 1
fi
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
wandb login --verify
for ckpt in /vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt /vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt; do
  test -f "$ckpt" || { echo "Missing checkpoint $ckpt" >&2; exit 1; }
done
test -d /vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini || { echo "Missing derived dataset /vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session $SESSION already exists; attach with: tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -n dinov2_force 'bash /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_dinov2_force.sh'
tmux new-window -t "$SESSION" -n dinov2_slip 'bash /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_dinov2_slip.sh'
tmux new-window -t "$SESSION" -n mae_force 'bash /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_mae_force.sh'
tmux new-window -t "$SESSION" -n mae_slip 'bash /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_mae_slip.sh'
tmux list-windows -t "$SESSION"
