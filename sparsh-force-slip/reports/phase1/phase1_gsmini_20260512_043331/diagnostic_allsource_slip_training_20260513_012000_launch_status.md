# Diagnostic All-Source Slip Launch Status

- generated_at: `2026-05-13T01:22:49`
- run_id: `phase1_gsmini_20260512_043331`
- tmux_session: `force-slip-diag-allsource-20260513_012000`
- derived_root: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- logs_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/diagnostic_allsource_slip_20260513_012000`
- data: `flat + sharp + sphere`
- sync_policy: no automatic `git push` or `git pull`

## tmux windows
```text
0: dinov2_slip_allsrc- (1 panes) [80x24] [layout b25d,80x24,0,0,0] @0
1: mae_slip_allsrc* (1 panes) [80x24] [layout b25e,80x24,0,0,1] @1 (active)
```

## GPU snapshot
```text
0, NVIDIA GeForce RTX 4090, 2980 MiB, 96 %
1, NVIDIA GeForce RTX 4090, 2836 MiB, 0 %
2, NVIDIA GeForce RTX 4090, 12 MiB, 0 %
3, NVIDIA GeForce RTX 4090, 12 MiB, 0 %
```

## train_task processes
```text
3654 python train_task.py --config-name=experiment/downstream_task/slip/gelsight_dinov2 paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=dinov2 experiment_name=phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"] data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]
3655 python train_task.py --config-name=experiment/downstream_task/slip/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"] data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]
4677 python train_task.py --config-name=experiment/downstream_task/slip/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"] data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]
4740 python train_task.py --config-name=experiment/downstream_task/slip/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"] data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]
4875 python train_task.py --config-name=experiment/downstream_task/slip/gelsight_dinov2 paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=dinov2 experiment_name=phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"] data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]
4938 python train_task.py --config-name=experiment/downstream_task/slip/gelsight_dinov2 paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=dinov2 experiment_name=phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"] data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"] data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]
5076 python train_task.py --config-name=experiment/downstream_task/slip/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmin
```

## Runs
- `dinov2_slip_allsource`: GPU `0`, W&B `phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000`, log `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/diagnostic_allsource_slip_20260513_012000/dinov2_slip_allsource.log` (171739 bytes)
  ```text
ated. In the futur
self.gen = func(*args, **kwds)
Epoch 0:  32%|███▏      | 160/500 [00:45<01:20,  4.21it/s,  train_loss: 0.159]
Epoch 0:  32%|███▏      | 161/500 [00:45<01:36,  3.52it/s,  train_loss: 0.159]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the futur
self.gen = func(*args, **kwds)
Epoch 0:  32%|███▏      | 161/500 [00:45<01:36,  3.52it/s,  train_loss: 0.525]
Epoch 0:  32%|███▏      | 162/500 [00:45<01:20,  4.19it/s,  train_loss: 0.525]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the futur
self.gen = func(*args, **kwds)
Epoch 0:  32%|███▏      | 162/500 [00:46<01:20,  4.19it/s,  train_loss: 0.178]
Epoch 0:  33%|███▎      | 163/500 [00:46<01:35,  3.52it/s,  train_loss: 0.178]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the futur
self.gen = func(*args, **kwds)
Epoch 0:  33%|███▎      | 163/500 [00:46<01:35,  3.52it/s,  train_loss: 0.257]
Epoch 0:  33%|███▎      | 164/500 [00:46<01:20,  4.18it/s,  train_loss: 0.257]
  ```
- `mae_slip_allsource`: GPU `1`, W&B `phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000`, log `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/diagnostic_allsource_slip_20260513_012000/mae_slip_allsource.log` (189118 bytes)
  ```text
12,  4.20it/s,  train_loss: 0.080]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the futur
self.gen = func(*args, **kwds)
Epoch 0:  39%|███▉      | 196/500 [00:50<01:12,  4.20it/s,  train_loss: 0.379]
Epoch 0:  39%|███▉      | 197/500 [00:50<01:25,  3.54it/s,  train_loss: 0.379]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the futur
self.gen = func(*args, **kwds)
Epoch 0:  39%|███▉      | 197/500 [00:50<01:25,  3.54it/s,  train_loss: 0.206]
Epoch 0:  40%|███▉      | 198/500 [00:50<01:12,  4.18it/s,  train_loss: 0.206]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the futur
self.gen = func(*args, **kwds)
Epoch 0:  40%|███▉      | 198/500 [00:50<01:12,  4.18it/s,  train_loss: 0.127]
Epoch 0:  40%|███▉      | 199/500 [00:50<01:26,  3.49it/s,  train_loss: 0.127]/home/zjy/miniconda3/envs/sparsh/lib/python3.9/contextlib.py:87: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the futur
self.gen = func(*args, **kwds)
  ```

## Verification
- Hydra compose verification passed before launch; see paired `_hydra_compose.txt`.
- W&B login verified by launcher.
- Both tmux windows are active and GPUs 0/1 show training memory/utilization.
- Live logs are ignored by git; this status records their paths.
