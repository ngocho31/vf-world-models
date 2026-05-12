#!/bin/bash

ENCODER_PT="${WORKSPACE_ROOT}/.cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt"

# Setup PYTHONPATH: custom src FIRST, then upstream, then others
export PYTHONPATH="${VF_DRIVE_JEPA_ROOT}:${NAVSIM_ROOT}:${VJEPA2_ROOT}:${PYTHONPATH:-}"

cd "${VF_DRIVE_JEPA_ROOT}"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
  --config-path $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/training \
  --config-name drive_jepa_perception_free_training.yaml \
  agent.pretrain_pt_path="${ENCODER_PT}" \
  experiment_name=cache_agent \
  train_test_split=$TRAIN_TEST_SPLIT \
  worker.max_workers=4 \
  worker=single_machine_thread_pool
