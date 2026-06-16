#!/usr/bin/env bash
set -euo pipefail

PLANNER_CKPT="${WORKSPACE_ROOT}/.cache/checkpoints/phase2/drive_jepa_perception_free_agent_vitl.ckpt"
ENCODER_PT="${WORKSPACE_ROOT}/.cache/checkpoints/vjepa2/vitl_merge_3dataset_e50.pt"
AGENT_CONFIG="${NAVSIM_ROOT}/navsim/planning/script/config/common/agent/drive_jepa_perception_free_agent.yaml"

# Folder paths based on VF-DRIVE-JEPA_Usage.md
CAMERA_DIR="${WORKSPACE_ROOT}/dataset_vf/data/CAMERA/CAM_P_F"
NAV_DIR="${WORKSPACE_ROOT}/dataset_vf/data/OTHERS/NAV"
STEER_DIR="${WORKSPACE_ROOT}/dataset_vf/data/OTHERS/VEHICLE_STEER"

OUTPUT_DIR="${WORKSPACE_ROOT}/outputs/vf_qualitative_eval_enhanced"
FRAME_STRIDE=0
SAMPLE_GAP_SECONDS=1
MAX_FRAMES=100
FPS=10

# Setup PYTHONPATH: custom src FIRST, then upstream, then others
export PYTHONPATH="${VF_DRIVE_JEPA_ROOT}:${NAVSIM_ROOT}:${VJEPA2_ROOT}:${PYTHONPATH:-}"

echo "[INFO] Running enhanced qualitative evaluation with telemetry and units..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${VF_DRIVE_JEPA_ROOT}"
python "${SCRIPT_DIR}/vf_qualitative_eval_enhanced.py" \
  --checkpoint-path "${PLANNER_CKPT}" \
  --encoder-path "${ENCODER_PT}" \
  --agent-config "${AGENT_CONFIG}" \
  --camera-dir "${CAMERA_DIR}" \
  --nav-dir "${NAV_DIR}" \
  --steer-dir "${STEER_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --frame-stride "${FRAME_STRIDE}" \
  --sample-gap-seconds "${SAMPLE_GAP_SECONDS}" \
  --max-frames "${MAX_FRAMES}" \
  --fps "${FPS}"