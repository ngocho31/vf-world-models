#!/usr/bin/env bash
# Local environment for VF-Drive-JEPA.

export WORKSPACE_ROOT="/workspace"
export VF_DRIVE_JEPA_ROOT="${WORKSPACE_ROOT}/vf-drive-jepa"

export NAVSIM_ROOT="${VF_DRIVE_JEPA_ROOT}/navsim"
export VJEPA2_ROOT="${VF_DRIVE_JEPA_ROOT}/vjepa2"

export NUPLAN_MAP_VERSION="vf-maps-v1.0"
export NUPLAN_MAPS_ROOT="${WORKSPACE_ROOT}/dataset_vf/navsim_ready/maps"
export OPENSCENE_DATA_ROOT="${WORKSPACE_ROOT}/dataset_vf/navsim_ready"
export NAVSIM_EXP_ROOT="${WORKSPACE_ROOT}/outputs/navsim/exp_vf"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_ROOT}"

export METRIC_CACHE_PATH="${NAVSIM_EXP_ROOT}/eval_drive_jepa_perception_free_cache"

export TRAIN_TEST_SPLIT=vftrain
export TRAIN_TEST_SPLIT_NAVTEST=vftest
