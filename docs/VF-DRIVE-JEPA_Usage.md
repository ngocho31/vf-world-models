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

## 7. VF to NAVSIM Converter Usage

This section explains how to run the converter package at `src/vf_navsim_converter` to produce NAVSIM-style outputs from one VF scene root.

### 7.1 Prerequisites

From repository root:

```bash
cd /workspace
source ./src/vf-drive-jepa/env.local.sh
```

### 7.2 Required input layout

`--input-root` must point to a VF scene root containing at least:

```text
<VF_INPUT_ROOT>/
	CAMERA/
		CAM_P_F/
		CAM_P_FL/
		CAM_P_L/
		CAM_P_LB/
		CAM_P_FR/
		CAM_P_R/
		CAM_P_RB/
		CAM_P_B/
	OTHERS/
		NAV/*.csv
		IMU/*.csv                 # optional but strongly recommended
		VEHICLE_INFO/*.csv        # optional but strongly recommended
		VEHICLE_STEER/*.csv       # optional but strongly recommended
		Calibration/
			Camera_Intrinsics.json
			Extrinsics.json
	LIDAR/
		LIDAR_E_F/
		LIDAR_E_L/
		LIDAR_E_R/
		LIDAR_E_B/
		LIDAR_TOP/
	vn-hdmap-demo/
		lanelet2_map.osm          # or any *.osm in this folder
```

Notes:
- The map folder name must match `--map-location` (default: `vn-hdmap-demo`).

### 7.3 Run command

Minimal run:

```bash
python -m vf_navsim_converter.cli \
	--input-root /workspace/data/<VF_INPUT_ROOT> \
	--output-root /workspace/data/<VF_OUTPUT_ROOT>
```

Run with explicit metadata:

```bash
python -m vf_navsim_converter.cli \
	--input-root /workspace/data/<VF_INPUT_ROOT> \
	--output-root /workspace/data/<VF_OUTPUT_ROOT> \
	--vehicle-name veh-01 \
	--map-location vn-hdmap-demo \
	--map-version 2026-06-12
```

CLI arguments:
- `--input-root` (required): VF scene root
- `--output-root` (required): output root for NAVSIM-style artifacts
- `--vehicle-name` (optional, default `veh-01`)
- `--map-location` (optional, default `vn-hdmap-demo`)
- `--map-version` (optional, default current date `YYYY-MM-DD`)

### 7.4 Output layout

After success, `--output-root` contains:

```text
<VF_OUTPUT_ROOT>/
	maps/
		vf-maps-v1.0.json
		<map_location>/<map_version>/map.gpkg
	navsim_logs/
		trainval/
			<log_name>.pkl
	sensor_blobs/
		trainval/
			<log_name>/
				CAM_F0/ ... CAM_B0/
				MergedPointCloud/
	.conversion_metadata/
		stage0_map_conversion_report.json
		stage1_discovery_manifest.json
		stage2_raw_sensor_bundle.json
		stage3_alignment_manifest.json
		stage4_canonical_manifest.json
		stage45_scene_map_sync_context.json
		stage5_navsim_manifest.json
		stage7_artifact_manifest.json
		conversion_report.json
```
