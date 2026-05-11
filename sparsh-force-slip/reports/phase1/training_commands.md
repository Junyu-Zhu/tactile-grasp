# Phase 1 Downstream Training Commands

- run_id: `phase1_gsmini_20260512_043331`
- derived_root: `/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini`
- tmux session: `force-slip-phase1`
- logs: `/home/zjy/document/tactile-grasp/sparsh-force-slip/logs/phase1_gsmini_20260512_043331`

## Launch

```bash
/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/launch_phase1_tmux.sh
```

## dinov2_force

- gpu: `0`
- encoder/task: `dinov2` / `force`
- wandb: `phase1_gsmini_20260512_043331_dinov2_force_gsmini_20260512_043652`
- script: `/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_dinov2_force.sh`

```bash
cd /home/zjy/document/sparsh && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export CUDA_VISIBLE_DEVICES=0 && export WANDB_MODE=online && export WANDB_NAME=phase1_gsmini_20260512_043331_dinov2_force_gsmini_20260512_043652 && export PYTHONPATH=. && python train_task.py --config-name=experiment/downstream_task/force/gelsight_dinov2 paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=dinov2 experiment_name=phase1_gsmini_20260512_043331_dinov2_force_gsmini_20260512_043652 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini 'data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]' 'data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]'
```

## dinov2_slip

- gpu: `1`
- encoder/task: `dinov2` / `slip`
- wandb: `phase1_gsmini_20260512_043331_dinov2_slip_gsmini_20260512_043652`
- script: `/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_dinov2_slip.sh`

```bash
cd /home/zjy/document/sparsh && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export CUDA_VISIBLE_DEVICES=1 && export WANDB_MODE=online && export WANDB_NAME=phase1_gsmini_20260512_043331_dinov2_slip_gsmini_20260512_043652 && export PYTHONPATH=. && python train_task.py --config-name=experiment/downstream_task/slip/gelsight_dinov2 paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=dinov2 experiment_name=phase1_gsmini_20260512_043331_dinov2_slip_gsmini_20260512_043652 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dinov2-base/dinov2_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini 'data.dataset.config.list_datasets=["sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]' 'data.dataset.config.list_datasets_test=["sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'test.data.dataset_name=["sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]'
```

## mae_force

- gpu: `2`
- encoder/task: `mae` / `force`
- wandb: `phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652`
- script: `/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_mae_force.sh`

```bash
cd /home/zjy/document/sparsh && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export CUDA_VISIBLE_DEVICES=2 && export WANDB_MODE=online && export WANDB_NAME=phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652 && export PYTHONPATH=. && python train_task.py --config-name=experiment/downstream_task/force/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_force_gsmini_20260512_043652 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini 'data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]' 'data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]'
```

## mae_slip

- gpu: `3`
- encoder/task: `mae` / `slip`
- wandb: `phase1_gsmini_20260512_043331_mae_slip_gsmini_20260512_043652`
- script: `/home/zjy/document/tactile-grasp/sparsh-force-slip/runbooks/run_mae_slip.sh`

```bash
cd /home/zjy/document/sparsh && source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh && export CUDA_VISIBLE_DEVICES=3 && export WANDB_MODE=online && export WANDB_NAME=phase1_gsmini_20260512_043331_mae_slip_gsmini_20260512_043652 && export PYTHONPATH=. && python train_task.py --config-name=experiment/downstream_task/slip/gelsight_mae paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=mae experiment_name=phase1_gsmini_20260512_043331_mae_slip_gsmini_20260512_043652 task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-mae-base/mae_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini 'data.dataset.config.list_datasets=["sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]' 'data.dataset.config.list_datasets_test=["sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'test.data.dataset_name=["sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' 'data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]'
```
