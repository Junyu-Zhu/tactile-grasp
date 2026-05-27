#!/usr/bin/env bash
set -euo pipefail
REPORT="/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000_report.json"
LOG="/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase_lambda_architecture_future_ablation/20260528_000000/partially_shared_lam025.log"
if [[ -f "$REPORT" ]]; then
  echo "SKIP existing report: $REPORT"
  exit 0
fi
echo "[$(date '+%F %T')] START Partially shared λ=0.25 on GPU3 (WANDB offline fallback)" | tee -a "$LOG"
cd /home/zjy/document/tactile-grasp && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline WANDB_NAME=phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000 PYTHONPATH=. python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_arch_future_partially_shared_lam025_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_ps_lam025_gsmini_20260514_063052/mae_partially_shared_multitask/checkpoints/epoch-0050.pth --experiment-name phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000 --input-mode full --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --wandb-mode offline 2>&1 | tee -a "$LOG"
echo "[$(date '+%F %T')] DONE Partially shared λ=0.25" | tee -a "$LOG"
