# Phase 1 Training Launch Status

- generated_at: `2026-05-12T04:46:44`
- run_id: `phase1_gsmini_20260512_043331`
- server_repo: `/home/zjy/document/tactile-grasp`
- branch: `sparsh-force-slip`
- tmux_session: `force-slip-phase1`
- logs_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331`
- sync_policy: no automatic `git push` or `git pull`; user performs sync manually

## tmux windows
```text
0: dinov2_force (1 panes) [80x24] [layout b261,80x24,0,0,4] @4
1: dinov2_slip (1 panes) [80x24] [layout b262,80x24,0,0,5] @5
2: mae_force- (1 panes) [80x24] [layout b263,80x24,0,0,6] @6
3: mae_slip* (1 panes) [80x24] [layout b264,80x24,0,0,7] @7 (active)
```

## GPU snapshot
```text
0, NVIDIA GeForce RTX 4090, 2976 MiB, 32 %
1, NVIDIA GeForce RTX 4090, 2938 MiB, 44 %
2, NVIDIA GeForce RTX 4090, 2936 MiB, 33 %
3, NVIDIA GeForce RTX 4090, 2837 MiB, 32 %
```

## main train_task processes
The list below filters out child worker processes whose parent is another `train_task.py` process.
```text
pid=4169285 ppid=4169247 pgid=4169241 etime=07:20 stat=Sl+ cmd=python train_task.py --config-name=experiment/downstream_task/force/gelsight_dinov2 paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=dinov2 experiment_name=phase1_gsmini_20260512_043331_dinov2_force_gsmini_20260512_043652 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt data.dataset.conf
pid=4169286 ppid=4169256 pgid=4169245 etime=07:20 stat=Sl+ cmd=python train_task.py --config-name=experiment/downstream_task/slip/gelsight_dinov2 paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=dinov2 experiment_name=phase1_gsmini_20260512_043331_dinov2_slip_gsmini_20260512_043652 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt data.dataset.config
pid=4169287 ppid=4169267 pgid=4169257 etime=07:20 stat=Rl+ cmd=python train_task.py --config-name=experiment/downstream_task/force/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset
pid=4169288 ppid=4169278 pgid=4169270 etime=07:20 stat=Rl+ cmd=python train_task.py --config-name=experiment/downstream_task/slip/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_slip_gsmini_20260512_043652 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset=/
```

## W&B run names and logs
- `dinov2_force`: GPU `0`, W&B `phase1_gsmini_20260512_043331_dinov2_force_gsmini_20260512_043652`, log `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/dinov2_force.log` (286834 bytes)
  ```text
self.gen = func(*args, **kwds)
Validation:   0%|          | 0/149 [00:02<?, ?it/s,  val_loss: 0.179]
Validation:   1%|          | 1/149 [00:02<05:11,  2.11s/it,  val_loss: 0.179]
Validation:   1%|          | 1/149 [00:02<05:11,  2.11s/it,  val_loss: 0.192]
Validation:   1%|▏         | 2/149 [00:02<02:18,  1.06it/s,  val_loss: 0.192]
Validation:   1%|▏         | 2/149 [00:02<02:18,  1.06it/s,  val_loss: 0.189]
Validation:   2%|▏         | 3/149 [00:02<01:22,  1.76it/s,  val_loss: 0.189]
Validation:   2%|▏         | 3/149 [00:02<01:22,  1.76it/s,  val_loss: 0.186]
Validation:   3%|▎         | 4/149 [00:02<00:56,  2.55it/s,  val_loss: 0.186]
Validation:   3%|▎         | 4/149 [00:02<00:56,  2.55it/s,  val_loss: 0.184]
Validation:   3%|▎         | 5/149 [00:02<00:42,  3.39it/s,  val_loss: 0.184]
Validation:   3%|▎         | 5/149 [00:02<00:42,  3.39it/s,  val_loss: 0.194]
Validation:   4%|▍         | 6/149 [00:02<00:38,  3.72it/s,  val_loss: 0.194]
Validation:   4%|▍         | 6/149 [00:02<00:38,  3.72it/s,  val_loss: 0.176]
Validation:   5%|▍         | 7/149 [00:02<00:32,  4.36it/s,  val_loss: 0.176]
Validation:   5%|▍         | 7/149 [00:03<00:32,  4.36it/s,  val_loss: 0.188]
Validat
  ```
- `dinov2_slip`: GPU `1`, W&B `phase1_gsmini_20260512_043331_dinov2_slip_gsmini_20260512_043652`, log `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/dinov2_slip.log` (807665 bytes)
  ```text
self.gen = func(*args, **kwds)
Epoch 3:  14%|█▍        | 64/446 [00:17<01:30,  4.22it/s,  train_loss: 0.039]
Epoch 3:  15%|█▍        | 65/446 [00:17<01:51,  3.43it/s,  train_loss: 0.039]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manag
self.gen = func(*args, **kwds)
Epoch 3:  15%|█▍        | 65/446 [00:17<01:51,  3.43it/s,  train_loss: 0.020]
Epoch 3:  15%|█▍        | 66/446 [00:17<01:32,  4.12it/s,  train_loss: 0.020]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manag
self.gen = func(*args, **kwds)
  ```
- `mae_force`: GPU `2`, W&B `phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652`, log `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/mae_force.log` (305463 bytes)
  ```text
self.gen = func(*args, **kwds)
Validation:   0%|          | 0/149 [00:01<?, ?it/s,  val_loss: 0.157]
Validation:   1%|          | 1/149 [00:01<04:49,  1.96s/it,  val_loss: 0.157]
Validation:   1%|          | 1/149 [00:02<04:49,  1.96s/it,  val_loss: 0.167]
Validation:   1%|▏         | 2/149 [00:02<02:08,  1.14it/s,  val_loss: 0.167]
Validation:   1%|▏         | 2/149 [00:02<02:08,  1.14it/s,  val_loss: 0.166]
Validation:   2%|▏         | 3/149 [00:02<01:17,  1.88it/s,  val_loss: 0.166]
Validation:   2%|▏         | 3/149 [00:02<01:17,  1.88it/s,  val_loss: 0.163]
Validation:   3%|▎         | 4/149 [00:02<00:53,  2.71it/s,  val_loss: 0.163]
Validation:   3%|▎         | 4/149 [00:02<00:53,  2.71it/s,  val_loss: 0.161]
Validation:   3%|▎         | 5/149 [00:02<00:40,  3.55it/s,  val_loss: 0.161]
Validation:   3%|▎         | 5/149 [00:03<00:40,  3.55it/s,  val_loss: 0.170]
Validation:   4%|▍         | 6/149 [00:03<00:55,  2.55it/s,  val_loss: 0.170]
Validation:   4%|▍         | 6/149 [00:03<00:55,  2.55it/s,  val_loss: 0.156]
Validation:   5%|▍         | 7/149 [00:03<00:43,  3.30it/s,  val_loss: 0.156]
Validation:   5%|▍         | 7/149 [00:03<00:43,  3.30it/s,  val_loss: 0.165]
Validat
  ```
- `mae_slip`: GPU `3`, W&B `phase1_gsmini_20260512_043331_mae_slip_gsmini_20260512_043652`, log `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/mae_slip.log` (858484 bytes)
  ```text
self.gen = func(*args, **kwds)
Epoch 3:  38%|███▊      | 169/446 [00:39<00:57,  4.79it/s,  train_loss: 0.018]
Epoch 3:  38%|███▊      | 170/446 [00:39<01:06,  4.14it/s,  train_loss: 0.018]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context mana
self.gen = func(*args, **kwds)
Epoch 3:  38%|███▊      | 170/446 [00:39<01:06,  4.14it/s,  train_loss: 0.246]
Epoch 3:  38%|███▊      | 171/446 [00:39<00:58,  4.74it/s,  train_loss: 0.246]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context mana
self.gen = func(*args, **kwds)
  ```

## Verification summary
- Derived dataset was generated under `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`.
- `dataloader_smoke_test.json` passed.
- `slip_alignment_audit.json` passed with zero issues.
- Hydra compose verification passed for all four downstream training commands.
- Four tmux windows were launched for DINOv2/MAE force/slip downstream tasks.
- Live training logs are intentionally ignored by git; this status file records their paths.
