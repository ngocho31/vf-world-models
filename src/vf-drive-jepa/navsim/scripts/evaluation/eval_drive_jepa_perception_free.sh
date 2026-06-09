#!/bin/bash

PLANNER_CKPT="${WORKSPACE_ROOT}/.cache/checkpoints/phase2/vf_drive_jepa_perception_free_agent_vitl.ckpt"
ENCODER_PT="${WORKSPACE_ROOT}/.cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt"

# Setup PYTHONPATH: custom src FIRST, then upstream, then others
export PYTHONPATH="${VF_DRIVE_JEPA_ROOT}:${NAVSIM_ROOT}:${VJEPA2_ROOT}:${PYTHONPATH:-}"

cd "${VF_DRIVE_JEPA_ROOT}"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
  --config-path $NAVSIM_DEVKIT_ROOT/navsim/planning/script/config/pdm_scoring \
  --config-name vf_run_pdm_score.yaml \
  train_test_split=$TRAIN_TEST_SPLIT_NAVTEST \
  metric_cache_path=$METRIC_CACHE_PATH \
  agent=drive_jepa_perception_free_agent \
  agent.checkpoint_path="${PLANNER_CKPT}" \
  agent.pretrain_pt_path="${ENCODER_PT}" \
  worker=single_machine_thread_pool \
  worker.max_workers=1 \
  worker.use_process_pool=false \
  experiment_name=eval_drive_jepa_perception_free_agent
