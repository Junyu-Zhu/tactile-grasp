# Diagnostic All-Source Slip Training Launch

- generated_at: `2026-05-13T01:20:00`
- run_id: `phase1_gsmini_20260512_043331`
- tmux_session: `force-slip-diag-allsource-20260513_012000`
- derived_root: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- logs_dir: `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/diagnostic_allsource_slip_20260513_012000`
- data: `flat + sharp + sphere` train/val derived splits
- sync_policy: no automatic `git push` or `git pull`

## Train datasets

```text
flat_batch_1_train
flat_batch_2_train
sharp_batch_1_train
sharp_batch_2_train
sphere_batch_1_train
sphere_batch_2_train
sphere_batch_3_train
sphere_batch_4_train
sphere_batch_5_train
sphere_batch_6_train
```

## Validation datasets

```text
flat_batch_1_val
flat_batch_2_val
sharp_batch_1_val
sharp_batch_2_val
sphere_batch_1_val
sphere_batch_2_val
sphere_batch_3_val
sphere_batch_4_val
sphere_batch_5_val
sphere_batch_6_val
```

## Runs

### dinov2_slip_allsource

- GPU: `0`
- W&B: `phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000`
- script: `/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_dinov2_slip_allsource.sh`
- log: `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/diagnostic_allsource_slip_20260513_012000/dinov2_slip_allsource.log`

```bash
cd /home/zjy/document/sparsh && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export CUDA_VISIBLE_DEVICES=0 && export WANDB_MODE=online && export WANDB_NAME=phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000 && export PYTHONPATH=. && python train_task.py --config-name=experiment/downstream_task/slip/gelsight_dinov2 paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=dinov2 experiment_name=phase1_gsmini_20260512_043331_dinov2_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini 'data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]' 'data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]'
```

### mae_slip_allsource

- GPU: `1`
- W&B: `phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000`
- script: `/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_mae_slip_allsource.sh`
- log: `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331/diagnostic_allsource_slip_20260513_012000/mae_slip_allsource.log`

```bash
cd /home/zjy/document/sparsh && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export CUDA_VISIBLE_DEVICES=1 && export WANDB_MODE=online && export WANDB_NAME=phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000 && export PYTHONPATH=. && python train_task.py --config-name=experiment/downstream_task/slip/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_slip_allsource_diag_gsmini_20260513_012000 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini 'data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]' 'data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]'
```

## Launcher

```bash
/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/launch_diagnostic_allsource_slip_tmux.sh
```
