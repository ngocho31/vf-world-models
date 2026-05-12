#!/bin/bash

# Setup PYTHONPATH: custom src FIRST, then upstream, then others
export PYTHONPATH="${VF_DRIVE_JEPA_ROOT}:${NAVSIM_ROOT}:${VJEPA2_ROOT}:${PYTHONPATH:-}"

cd "${VF_DRIVE_JEPA_ROOT}"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=$TRAIN_TEST_SPLIT_NAVTEST \
  metric_cache_path=$METRIC_CACHE_PATH
