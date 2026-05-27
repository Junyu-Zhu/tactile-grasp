# Phase Lambda Decoupled Sweep Report

- generated_at: `2026-05-28T02:55:37`
- stamp: `20260528_000000`
- encoder: `mae`
- decoder_variant: `decoupled`
- raw_data_modified: `False`
- selection policy: non-hard-fail -> minimum Force RMSE -> higher Slip F1/accuracy.

## Current force-slip validation results

| lambda | run id | F RMSE | ΔF | SF1 | ΔSF1 drop | SA | ΔSA drop | gate | epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 0.10 | `phase_lambda_decoupled_mae_lam010_20260528_000000` | 0.0303 | -5.88% | 0.9676 | +0.53% | 0.9837 | +0.27% | pass | 30 |
| 0.25 | `phase_lambda_decoupled_mae_lam025_20260528_000000` | 0.0312 | -3.03% | 0.9653 | +0.77% | 0.9825 | +0.39% | pass | 45 |
| 0.50 | `phase_lambda_decoupled_mae_lam050_20260528_000000` | 0.0307 | -4.64% | 0.9663 | +0.66% | 0.9830 | +0.34% | pass | 45 |
| 0.75 | `phase_lambda_decoupled_mae_lam075_20260528_000000` | 0.0307 | -4.45% | 0.9686 | +0.42% | 0.9842 | +0.22% | pass | 30 |
| 1.00 | `phase3_1_decoupled_gsmini_20260516_154730` | 0.0304 | -5.42% | 0.9649 | +0.80% | 0.9824 | +0.41% | pass | 30 |

## Best lambda selection

Selected λ: `0.10`
- checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- gate: `pass`
- Force RMSE: `0.0303`; Slip F1: `0.9676`; Slip accuracy: `0.9837`
- beats existing λ=1.0 reference: `True`

## W&B links

- λ=0.10: [phase_lambda_decoupled_mae_lam010_20260528_000000_lambda_lam010](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_decoupled_mae_lam010_20260528_000000_lambda_lam010)
- λ=0.25: [phase_lambda_decoupled_mae_lam025_20260528_000000_lambda_lam025](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_decoupled_mae_lam025_20260528_000000_lambda_lam025)
- λ=0.50: [phase_lambda_decoupled_mae_lam050_20260528_000000_lambda_lam050](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_decoupled_mae_lam050_20260528_000000_lambda_lam050)
- λ=0.75: [phase_lambda_decoupled_mae_lam075_20260528_000000_lambda_lam075](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_decoupled_mae_lam075_20260528_000000_lambda_lam075)

## Training commands

### lam010

```bash
#!/usr/bin/env bash
set -euo pipefail
SUMMARY="/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/training_summary.json"
RUN_ID="phase_lambda_decoupled_mae_lam010_20260528_000000"
LAM="0.10"
TAG="lam010"
if [[ -f "$SUMMARY" ]]; then
  echo "SKIP existing summary: $SUMMARY"
  exit 0
fi
if ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "$RUN_ID" | grep -F -- "--encoder mae" | grep -F -- "--decoder-variant decoupled" | grep -q .; then
  echo "SKIP already running: $RUN_ID"
  exit 0
fi
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export WANDB_NAME="${RUN_ID}_lambda_${TAG}"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder mae \
  --run-id "$RUN_ID" \
  --decoder-variant decoupled \
  --lambda-slip "$LAM" \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --wandb-mode online \
  +trainer.devices=1
```
### lam025

```bash
#!/usr/bin/env bash
set -euo pipefail
SUMMARY="/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam025_20260528_000000/mae_decoupled_multitask/training_summary.json"
RUN_ID="phase_lambda_decoupled_mae_lam025_20260528_000000"
LAM="0.25"
TAG="lam025"
if [[ -f "$SUMMARY" ]]; then
  echo "SKIP existing summary: $SUMMARY"
  exit 0
fi
if ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "$RUN_ID" | grep -F -- "--encoder mae" | grep -F -- "--decoder-variant decoupled" | grep -q .; then
  echo "SKIP already running: $RUN_ID"
  exit 0
fi
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=online
export WANDB_NAME="${RUN_ID}_lambda_${TAG}"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder mae \
  --run-id "$RUN_ID" \
  --decoder-variant decoupled \
  --lambda-slip "$LAM" \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --wandb-mode online \
  +trainer.devices=1
```
### lam050

```bash
#!/usr/bin/env bash
set -euo pipefail
SUMMARY="/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam050_20260528_000000/mae_decoupled_multitask/training_summary.json"
RUN_ID="phase_lambda_decoupled_mae_lam050_20260528_000000"
LAM="0.50"
TAG="lam050"
if [[ -f "$SUMMARY" ]]; then
  echo "SKIP existing summary: $SUMMARY"
  exit 0
fi
if ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "$RUN_ID" | grep -F -- "--encoder mae" | grep -F -- "--decoder-variant decoupled" | grep -q .; then
  echo "SKIP already running: $RUN_ID"
  exit 0
fi
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=online
export WANDB_NAME="${RUN_ID}_lambda_${TAG}"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder mae \
  --run-id "$RUN_ID" \
  --decoder-variant decoupled \
  --lambda-slip "$LAM" \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --wandb-mode online \
  +trainer.devices=1
```
### lam075

```bash
#!/usr/bin/env bash
set -euo pipefail
SUMMARY="/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam075_20260528_000000/mae_decoupled_multitask/training_summary.json"
RUN_ID="phase_lambda_decoupled_mae_lam075_20260528_000000"
LAM="0.75"
TAG="lam075"
if [[ -f "$SUMMARY" ]]; then
  echo "SKIP existing summary: $SUMMARY"
  exit 0
fi
if ps -u zjy -ww -o args= | grep -E 'phase2_b_multitask.py' | grep -F -- "$RUN_ID" | grep -F -- "--encoder mae" | grep -F -- "--decoder-variant decoupled" | grep -q .; then
  echo "SKIP already running: $RUN_ID"
  exit 0
fi
cd "/home/zjy/document/tactile-grasp"
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=online
export WANDB_NAME="${RUN_ID}_lambda_${TAG}"
export PYTHONPATH=/home/zjy/document/sparsh:.
python sparsh-force-slip/scripts/phase2_b_multitask.py train \
  --encoder mae \
  --run-id "$RUN_ID" \
  --decoder-variant decoupled \
  --lambda-slip "$LAM" \
  --max-epochs 51 \
  --batch-size 100 \
  --num-workers 2 \
  --validation-frequency 5 \
  --wandb-mode online \
  +trainer.devices=1
```

## Interpretation

Among the completed small-lambda sweep runs, λ=0.10 is selected by the predefined policy. It has Force RMSE 0.0303, Slip F1 0.9676, and gate pass. This checkpoint should be used as the Stage-I interface for the next future causal-input ablation unless a later audit finds a stronger constraint violation.
