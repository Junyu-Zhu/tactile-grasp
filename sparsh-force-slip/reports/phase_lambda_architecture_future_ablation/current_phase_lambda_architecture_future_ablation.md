# Architecture-to-Future Full-dynamics Ablation

- generated_at: `2026-05-28T04:23:05`
- stamp: `20260528_000000`
- branch/head: `sparsh-force-slip` / `f42ce21`
- Stage-II input: `Z + predicted force + slip probability + causal ΔF` (`input_mode=full`).
- Decoupled λ=0.10 full-dynamics result is reused from Phase B; it is not retrained.
- raw_data_modified: `False`
- W&B mode: `offline fallback` because `api.wandb.ai:443` was unreachable from zjy-4090 during launch; run `wandb sync` on the saved run directories when network recovers. The deterministic W&B run ids/names are still recorded below.

## Current and future metrics

| architecture | F RMSE | SF1 | SA | H1 F1 | H1 AUPRC | H1 ECE | H1 lead | H3 F1 | H3 AUPRC | H3 ECE | H3 lead | H5 F1 | H5 AUPRC | H5 ECE | H5 lead | mean AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decoupled λ=0.10 (main, Phase B reused) | 0.0303 | 0.9676 | 0.9837 | 0.9654 | 0.9951 | 0.0055 | 3.5248 | 0.9567 | 0.9936 | 0.0068 | n/a | 0.9351 | 0.9803 | 0.0093 | n/a | 0.9896 |
| Decoupled λ=1.00 old | 0.0284 | 0.9593 | 0.9819 | 0.9676 | 0.9953 | 0.0061 | 4.9646 | 0.9562 | 0.9936 | 0.0072 | n/a | 0.9353 | 0.9800 | 0.0097 | n/a | 0.9896 |
| Consistency decoder | 0.0319 | 0.9787 | 0.9908 | 0.9662 | 0.9954 | 0.0076 | 2.8370 | 0.9582 | 0.9946 | 0.0115 | n/a | 0.9341 | 0.9825 | 0.0188 | n/a | 0.9908 |
| Partially shared λ=0.25 | 0.0294 | 0.9735 | 0.9885 | 0.9617 | 0.9944 | 0.0100 | 4.0000 | 0.9535 | 0.9933 | 0.0156 | n/a | 0.9274 | 0.9813 | 0.0251 | n/a | 0.9897 |

## Checkpoints and W&B

### Decoupled λ=0.10 (main, Phase B reused)
- Stage-I checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase_lambda_decoupled_mae_lam010_20260528_000000/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- Future-head checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_best_future_features_20260528_000000/heads/phase_lambda_best_full_20260528_000000/checkpoints/best.pth`
- W&B: [phase_lambda_best_full_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_best_full_20260528_000000)
- report_json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_decoupled_future_ablation/20260528_000000/phase_lambda_decoupled_future_ablation_report.json`

### Decoupled λ=1.00 old
- Stage-I checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth`
- Future-head checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_arch_future_decoupled_lam100_old_features_20260528_000000/heads/phase_lambda_arch_future_decoupled_lam100_old_full_20260528_000000/checkpoints/best.pth`
- W&B: [phase_lambda_arch_future_decoupled_lam100_old_full_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_arch_future_decoupled_lam100_old_full_20260528_000000)
- report_json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/phase_lambda_arch_future_decoupled_lam100_old_full_20260528_000000_report.json`

### Consistency decoder
- Stage-I checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_c_consistency_gsmini_20260514_161845/mae_consistency_multitask/checkpoints/epoch-0015.pth`
- Future-head checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_arch_future_consistency_decoder_features_20260528_000000/heads/phase_lambda_arch_future_consistency_decoder_full_20260528_000000/checkpoints/best.pth`
- W&B: [phase_lambda_arch_future_consistency_decoder_full_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_arch_future_consistency_decoder_full_20260528_000000)
- report_json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/phase_lambda_arch_future_consistency_decoder_full_20260528_000000_report.json`

### Partially shared λ=0.25
- Stage-I checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_ps_lam025_gsmini_20260514_063052/mae_partially_shared_multitask/checkpoints/epoch-0050.pth`
- Future-head checkpoint: `/vla1/zjy/sparsh_runs/force_slip_phase4/phase_lambda_arch_future_partially_shared_lam025_features_20260528_000000/heads/phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000/checkpoints/best.pth`
- W&B: [phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000](https://wandb.ai/junyuzhuzjy-zhejiang-university/sparsh-finetune-tactile-grasp/runs/phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000)
- report_json: `/home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000_report.json`

## Interpretation

- Decoupled λ=0.10 remains the preferred Stage-I interface for the main method because it is the λ-sweep selection: it preserves the current force-slip balance while matching the old decoupled λ=1.00 future mean AUPRC under the same full-dynamics Stage-II input.
- Best mean H1/H3/H5 future AUPRC in this comparison is `Consistency decoder` (0.9908), but the margin over Decoupled λ=0.10 is small and must be weighed against current Force RMSE, calibration, and architectural simplicity.
- Consistency decoder is not used as the main interface because its small future-AUPRC gain comes with worse current Force RMSE than the selected Decoupled λ=0.10 interface and higher calibration error at longer horizons; it is better presented as a tested alternative/upper-bound ablation.
- Partially shared λ=0.25 is not used as the main method because it does not dominate the future metrics and shows worse long-horizon calibration/ECE, so shared capacity is still vulnerable to force-slip negative transfer.
- The final manuscript can use this table to justify Decoupled λ=0.10 as a conservative, force-stable Stage-I interface and cite consistency/partially-shared rows as alternatives tested under identical full-dynamics Stage-II inputs.

## Training commands

### Decoupled λ=1.00 old
```bash
cd /home/zjy/document/tactile-grasp && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline WANDB_NAME=phase_lambda_arch_future_decoupled_lam100_old_full_20260528_000000 PYTHONPATH=. python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_arch_future_decoupled_lam100_old_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase3_1_decoupled_gsmini_20260516_154730/mae_decoupled_multitask/checkpoints/epoch-0030.pth --experiment-name phase_lambda_arch_future_decoupled_lam100_old_full_20260528_000000 --input-mode full --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --wandb-mode offline
```

### Consistency decoder
```bash
cd /home/zjy/document/tactile-grasp && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline WANDB_NAME=phase_lambda_arch_future_consistency_decoder_full_20260528_000000 PYTHONPATH=. python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_arch_future_consistency_decoder_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase2_c_consistency_gsmini_20260514_161845/mae_consistency_multitask/checkpoints/epoch-0015.pth --experiment-name phase_lambda_arch_future_consistency_decoder_full_20260528_000000 --input-mode full --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --wandb-mode offline
```

### Partially shared λ=0.25
```bash
cd /home/zjy/document/tactile-grasp && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline WANDB_NAME=phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000 PYTHONPATH=. python sparsh-force-slip/scripts/phase4_paper_experiments.py train-head --feature-kind decoupled --feature-run-id phase_lambda_arch_future_partially_shared_lam025_features_20260528_000000 --checkpoint /vla1/zjy/sparsh_runs/force_slip_phase2/phase2_b_ps_lam025_gsmini_20260514_063052/mae_partially_shared_multitask/checkpoints/epoch-0050.pth --experiment-name phase_lambda_arch_future_partially_shared_lam025_full_20260528_000000 --input-mode full --report-dir /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000 --horizons 1 3 5 --precompute-batch-size 128 --train-batch-size 1024 --num-workers 2 --max-epochs 40 --wandb-mode offline
```


## W&B offline sync commands

Because `api.wandb.ai:443` was unreachable during this run, local offline runs were created with deterministic run ids. Sync them later with:

```bash
bash /home/zjy/document/tactile-grasp/sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/wandb_offline_sync_commands.sh
```

A background retry supervisor has also been installed:

```bash
tmux attach -t arch_future_20260528_wandb_sync
# or run manually:
bash /home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/phase_lambda_architecture_future_ablation_20260528_000000/sync_wandb_offline_when_ready.sh
```

## W&B cloud sync status

The server could not reach `api.wandb.ai:443`, so the offline W&B runs were copied to the local client and synced from there. Sync completed for all 9 offline runs; see:

- `sparsh-force-slip/reports/phase_lambda_architecture_future_ablation/20260528_000000/wandb_sync_complete.txt`
- `sparsh-force-slip/logs/phase_lambda_architecture_future_ablation/20260528_000000/local_wandb_sync_from_client.log`
