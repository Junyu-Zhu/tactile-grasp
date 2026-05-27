# Best-λ Future Causal-input Ablation

- generated_at: `2026-05-28T04:15:33`
- stamp: `20260528_000000`
- best_lambda: `0.1`
- stage_i_checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- feature_run_id: `phase_lambda_best_future_features_20260528_000000`
- raw_data_modified: `False`

## Best-λ Stage-II results

| condition | H1 F1 | H1 AUPRC | H1 ECE | H1 lead | H1 HR recall | H3 F1 | H3 AUPRC | H3 ECE | H5 F1 | H5 AUPRC | H5 ECE | ckpt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| z_only | 0.9450 | 0.9655 | 0.0136 | 2.4865 | n/a | 0.8890 | 0.9258 | 0.0169 | 0.8401 | 0.8971 | 0.0184 | `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_z_only_20260528_000000/checkpoints/best.pth` |
| z_p_slip | 0.9612 | 0.9838 | 0.0088 | 6.1389 | n/a | 0.9173 | 0.9645 | 0.0173 | 0.8723 | 0.9404 | 0.0254 | `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_z_p_slip_20260528_000000/checkpoints/best.pth` |
| z_force | 0.9589 | 0.9763 | 0.0029 | 3.3929 | n/a | 0.9019 | 0.9455 | 0.0114 | 0.8569 | 0.9200 | 0.0176 | `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_z_force_20260528_000000/checkpoints/best.pth` |
| z_force_slip | 0.9631 | 0.9845 | 0.0101 | 4.7313 | n/a | 0.9157 | 0.9652 | 0.0202 | 0.8713 | 0.9415 | 0.0296 | `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_z_force_slip_20260528_000000/checkpoints/best.pth` |
| full | 0.9654 | 0.9951 | 0.0055 | 3.5248 | n/a | 0.9567 | 0.9936 | 0.0068 | 0.9351 | 0.9803 | 0.0093 | `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth` |
| full_plus_q | 0.9640 | 0.9954 | 0.0127 | 2.0455 | 0.9973 | 0.9416 | 0.9938 | 0.0238 | 0.9142 | 0.9813 | 0.0389 | `/vla1/zjy/sparsh_runs/force_slip_phase5/phase_lambda_best_friction_features_20260528_000000/heads/phase_lambda_best_full_plus_q_20260528_000000/checkpoints/best.pth` |

## Selection and comparison

- Best Stage-II condition by mean H1/H3/H5 AUPRC: `full_plus_q`.
- Old λ=1.0 Phase4 report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase4/phase4_3_20260517_171500/phase4_3_world_model_input_ablation_report.json`.
- Old λ=1.0 Phase5 report: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase5/phase5_2_friction_future_head/phase5_2_friction_future_head_report.json`.
- Interpret replacement conservatively: use best-λ as the main result only if it improves the paper-facing future metrics without violating current force/slip gates; otherwise keep it as an ablation.

## Direct comparison to old λ=1.0 future ablation

| condition | new mean AUPRC | old mean AUPRC | Δ mean AUPRC | old row found |
|---|---:|---:|---:|---|
| z_only | 0.9295 | 0.9295 | 0.0000 | True |
| z_p_slip | 0.9629 | 0.9609 | 0.0020 | True |
| z_force | 0.9473 | 0.9508 | -0.0035 | True |
| z_force_slip | 0.9637 | 0.9626 | 0.0011 | True |
| full | 0.9896 | 0.9896 | 0.0000 | True |
| full_plus_q | 0.9902 | 0.9906 | -0.0005 | True |

## W&B runs

- z_only: [phase_lambda_best_z_only_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_best_z_only_20260528_000000)
- z_p_slip: [phase_lambda_best_z_p_slip_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_best_z_p_slip_20260528_000000)
- z_force: [phase_lambda_best_z_force_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_best_z_force_20260528_000000)
- z_force_slip: [phase_lambda_best_z_force_slip_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_best_z_force_slip_20260528_000000)
- full: [phase_lambda_best_full_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_best_full_20260528_000000)
- full_plus_q: [phase_lambda_best_full_plus_q_20260528_000000_full_plus_q](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_best_full_plus_q_20260528_000000_full_plus_q)

## Commands

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_NAME=phase_lambda_best_z_only_20260528_000000 WANDB_MODE=offline python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_best_future_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth --experiment-name phase_lambda_best_z_only_20260528_000000 --input-mode z_only --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --lr 0.001 --hidden-dim 512 --dropout 0.1 --seed 42 --wandb-mode offline
```
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_NAME=phase_lambda_best_z_p_slip_20260528_000000 WANDB_MODE=offline python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_best_future_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth --experiment-name phase_lambda_best_z_p_slip_20260528_000000 --input-mode z_p_slip --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --lr 0.001 --hidden-dim 512 --dropout 0.1 --seed 42 --wandb-mode offline
```
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_NAME=phase_lambda_best_z_force_20260528_000000 WANDB_MODE=offline python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_best_future_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth --experiment-name phase_lambda_best_z_force_20260528_000000 --input-mode z_force --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --lr 0.001 --hidden-dim 512 --dropout 0.1 --seed 42 --wandb-mode offline
```
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_NAME=phase_lambda_best_z_force_slip_20260528_000000 WANDB_MODE=offline python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_best_future_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth --experiment-name phase_lambda_best_z_force_slip_20260528_000000 --input-mode z_force_slip --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --lr 0.001 --hidden-dim 512 --dropout 0.1 --seed 42 --wandb-mode offline
```
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_NAME=phase_lambda_best_full_20260528_000000 WANDB_MODE=offline python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_best_future_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth --experiment-name phase_lambda_best_full_20260528_000000 --input-mode full --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --lr 0.001 --hidden-dim 512 --dropout 0.1 --seed 42 --wandb-mode offline
```
```bash
CUDA_VISIBLE_DEVICES=0 WANDB_NAME=phase_lambda_best_full_plus_q_20260528_000000 WANDB_MODE=offline python sparsh-force-slip/scripts/phase_lambda_future_ablation.py --stamp 20260528_000000 --feature-run-id phase_lambda_best_future_features_20260528_000000 --q-feature-run-id phase_lambda_best_friction_features_20260528_000000 --horizons 1 3 5 --max-epochs 40 --wandb-mode offline # internal full_plus_q head
```
