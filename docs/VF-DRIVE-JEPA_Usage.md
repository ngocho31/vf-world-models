# VF-DRIVE-JEPA: Train and Evaluate

This document provides a practical end-to-end workflow to run training and evaluation for `vf-drive-jepa` in this repository.

## 1. Prerequisites

Assumptions:
- Repository root is `vf-world-models`.
- You have a CUDA-capable GPU and working PyTorch/CUDA stack.

Install docker and run the container with:

```bash
docker build -t ai-base -f tools/docker/Dockerfile .
docker run --gpus all --shm-size=8g -u $(id -u):$(id -g) -v $(pwd):/workspace -p 6006:6006 -it --rm ai-base
```

## 2. Required Data

At minimum, make sure these datasets exist under your data root:
- `maps`
- `navsim_logs`
- `sensor_blobs`

Expected layout (example):

```text
<OPENSCENE_DATA_ROOT>/
	maps/
	navsim_logs/
		trainval/
		test/
	sensor_blobs/
		trainval/
		test/
```

Useful split names:
- Training: `trainval`
- Evaluation: `test`

## 3. Download Checkpoints

Create checkpoint folders and download pretrained weights:

```bash
mkdir -p .cache/checkpoints/vjepa2
wget -4 -c -N -O .cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt \
	https://huggingface.co/datasets/LinhanWang/Drive-JEPA/resolve/main/vitl_merge_3dataset_e50.pt

mkdir -p .cache/checkpoints/phase2
wget -4 -c -N -O .cache/checkpoints/phase2/drive_jepa_perception_free_agent_vitl.ckpt \
	https://huggingface.co/datasets/LinhanWang/Drive-JEPA/resolve/main/drive_jepa_perception_free_agent_vitl.ckpt
```

## 4. Environment Variables

Set these environment variables to point to the correct paths.

```bash
source ./src/vf-drive-jepa/env.local.sh
```

## 5. Training Workflow

Run from repo root after exporting env vars.

### 5.1 Build training cache (recommended first)

```bash
bash src/vf-drive-jepa/navsim/scripts/training/run_drive_jepa_perception_free_cache.sh
```

This uses:
- config: `drive_jepa_perception_free_training.yaml`
- split: `$TRAIN_TEST_SPLIT`
- cache target: `${NAVSIM_EXP_ROOT}/train_drive_jepa_perception_free_cache`

### 5.2 Train the planner

```bash
bash src/vf-drive-jepa/navsim/scripts/training/train_drive_jepa_perception_free.sh
```

Training outputs are written under:

```text
${NAVSIM_EXP_ROOT}/train_drive_jepa_perception_free_agent/<timestamp>/
```

## 6. Evaluation Workflow

You can run two types of evaluation.

### 6.1 Quantitative evaluation (PDM score)

Step 1: build metric cache

```bash
bash src/vf-drive-jepa/navsim/scripts/evaluation/run_drive_jepa_perception_free_metric_cache.sh
```

Step 2: run PDM score evaluation

```bash
bash src/vf-drive-jepa/navsim/scripts/evaluation/eval_drive_jepa_perception_free.sh
```

Important variables for this flow:
- `TRAIN_TEST_SPLIT_NAVTEST`
- `METRIC_CACHE_PATH`

Outputs are written under:

```text
${NAVSIM_EXP_ROOT}/eval_drive_jepa_perception_free_agent/<timestamp>/
```

### 6.2 Qualitative evaluation (video + trajectory JSON)

```bash
bash src/vf-drive-jepa/navsim/scripts/evaluation/run_vf_qualitative_eval.sh
```

Default output folder:

```text
outputs/vf_qualitative_eval/
	vf_qualitative_eval.mp4
	frames/
	predicted_trajectories.json
```
