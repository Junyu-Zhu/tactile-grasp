#!/usr/bin/env bash
set -euo pipefail
cd "/home/zjy/document/sparsh"
source /home/zjy/miniconda3/etc/profile.d/conda.sh && conda activate sparsh
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=online
export WANDB_NAME="phase3_0_dino_a_gsmini_20260516_023557_dino_slip_allsource"
export PYTHONPATH=.
python train_task.py --config-name=experiment/downstream_task/slip/gelsight_dino \
  paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 \
  ssl_model_size=base ssl_name=dino \
  experiment_name="phase3_0_dino_a_gsmini_20260516_023557_dino_slip_allsource" \
  task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-dino-base/dino_vitbase.ckpt \
  data.dataset.config.path_dataset="/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini" \
  'data.dataset.config.list_datasets=["flat_batch_1_train","flat_batch_2_train","sharp_batch_1_train","sharp_batch_2_train","sphere_batch_1_train","sphere_batch_2_train","sphere_batch_3_train","sphere_batch_4_train","sphere_batch_5_train","sphere_batch_6_train"]' \
  'data.dataset.config.list_datasets_test=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' \
  'test.data.dataset_name=["flat_batch_1_val","flat_batch_2_val","sharp_batch_1_val","sharp_batch_2_val","sphere_batch_1_val","sphere_batch_2_val","sphere_batch_3_val","sphere_batch_4_val","sphere_batch_5_val","sphere_batch_6_val"]' \
  'data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]'
