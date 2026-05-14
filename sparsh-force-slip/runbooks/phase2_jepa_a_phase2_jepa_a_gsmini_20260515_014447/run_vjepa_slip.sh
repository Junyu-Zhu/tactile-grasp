#!/usr/bin/env bash
set -euo pipefail
echo "START $(date -Is) phase2_jepa_a_gsmini_20260515_014447_vjepa_slip gpu=3"
echo "TMUX_SESSION=phase2_jepa_a_gsmini_20260515_014447_vjepa_slip_gpu3"
echo "WANDB_NAME=phase2_jepa_a_gsmini_20260515_014447_vjepa_slip"
echo "EXPERIMENT_NAME=phase2_jepa_a_gsmini_20260515_014447_vjepa_slip"
echo "DERIVED_DATASET=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini"
echo "CHECKPOINT_ENCODER=/vla1/zjy/sparsh_models/sparsh-vjepa-base/vjepa_vitbase.ckpt"
cd /home/zjy/document/sparsh
source /home/zjy/miniconda3/etc/profile.d/conda.sh
conda activate sparsh
export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=online
export WANDB_NAME=phase2_jepa_a_gsmini_20260515_014447_vjepa_slip
export PYTHONPATH=.
python train_task.py --config-name=experiment/downstream_task/slip/gelsight_vjepa paths=zjy_4090 wandb=tactile_grasp +trainer.devices=1 ssl_model_size=base ssl_name=vjepa experiment_name=phase2_jepa_a_gsmini_20260515_014447_vjepa_slip task.checkpoint_encoder=/vla1/zjy/sparsh_models/sparsh-vjepa-base/vjepa_vitbase.ckpt data.dataset.config.path_dataset=/vla1/zjy/sparsh_runs/force_slip_phase1/phase1_gsmini_20260512_043331/derived_gsmini 'data.dataset.config.list_datasets=["flat_batch_1_train", "flat_batch_2_train", "sharp_batch_1_train", "sharp_batch_2_train", "sphere_batch_1_train", "sphere_batch_2_train", "sphere_batch_3_train", "sphere_batch_4_train", "sphere_batch_5_train", "sphere_batch_6_train"]' 'data.dataset.config.list_datasets_test=["flat_batch_1_val", "flat_batch_2_val", "sharp_batch_1_val", "sharp_batch_2_val", "sphere_batch_1_val", "sphere_batch_2_val", "sphere_batch_3_val", "sphere_batch_4_val", "sphere_batch_5_val", "sphere_batch_6_val"]' 'test.data.dataset_name=["flat_batch_1_val", "flat_batch_2_val", "sharp_batch_1_val", "sharp_batch_2_val", "sphere_batch_1_val", "sphere_batch_2_val", "sphere_batch_3_val", "sphere_batch_4_val", "sphere_batch_5_val", "sphere_batch_6_val"]' 'data.dataset.config.max_delta_forceXYZ=[0.80,0.80,0.40]'
echo "END $(date -Is) phase2_jepa_a_gsmini_20260515_014447_vjepa_slip gpu=3"
