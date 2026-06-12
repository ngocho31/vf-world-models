"""Stage 5: assemble canonical frames into NAVSIM-compatible frame records."""

import json
import hashlib
import math
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts.models_canonical import CanonicalEgoState, CanonicalFrame
from ..contracts.models_navsim import (
    NavsimCameraRecord,
    NavsimFrameRecord,
    NavsimSceneArtifact,
    SceneMapSyncContext,
)


class DefaultNavsimAssembler:
    """Map canonical contracts to NAVSIM frame dictionaries.

    Driving command inference strategy:
    - First frame: use lateral acceleration (no prior frame available)
    - Subsequent frames: use heading change (more reliable), fallback to lateral acceleration
    """

    def __init__(
        self,
        *,
        vehicle_name: str = "veh-01",
        map_location: str = "vn-hdmap-demo",
        heading_threshold: float = 0.05,  # radians, ~2.9°
        lateral_accel_threshold: float = 0.7,  # m/s²
        lateral_velocity_threshold: float = 0.8,  # m/s
        min_forward_velocity: float = 0.5,  # m/s (threshold for "stopped" → unknown)
    ) -> None:
        self._vehicle_name = vehicle_name
        self._map_location = map_location
        self._heading_threshold = heading_threshold
        self._lateral_accel_threshold = lateral_accel_threshold
        self._lateral_velocity_threshold = lateral_velocity_threshold
        self._min_forward_velocity = min_forward_velocity

    def assemble(
        self,
        frames: Sequence[CanonicalFrame],
        scene_map_context: SceneMapSyncContext | None = None,
    ) -> NavsimSceneArtifact:
        log_name, scene_token, log_token, vehicle_name = _resolve_scene_metadata(
            frames=frames,
            configured_vehicle_name=self._vehicle_name,
        )

        resolved_map_location = self._map_location
        resolved_roadblock_ids: List[str] = []
        if scene_map_context is not None:
            if scene_map_context.map_location:
                resolved_map_location = scene_map_context.map_location
            resolved_roadblock_ids = [str(roadblock_id) for roadblock_id in scene_map_context.roadblock_ids]

        navsim_frames: List[NavsimFrameRecord] = []
        prev_ego_state: Optional[CanonicalEgoState] = None

        for frame_index, frame in enumerate(frames):
            # Infer driving command from ego dynamics
            if prev_ego_state is not None:
                # Use heading change (more reliable with frame history)
                driving_cmd = _infer_driving_command_from_heading(
                    prev_ego_state=prev_ego_state,
                    curr_ego_state=frame.ego_state,
                    heading_threshold=self._heading_threshold,
                    min_forward_velocity=self._min_forward_velocity,
                )
            else:
                # First frame: use lateral acceleration
                driving_cmd = _infer_driving_command_from_lateral_accel(
                    ego_state=frame.ego_state,
                    lateral_accel_threshold=self._lateral_accel_threshold,
                    lateral_velocity_threshold=self._lateral_velocity_threshold,
                    min_forward_velocity=self._min_forward_velocity,
                )

            cams = {
                cam_name: NavsimCameraRecord(
                    data_path=sample.relative_path,
                    sensor2lidar_rotation=sample.sensor2lidar_rotation,
                    sensor2lidar_translation=sample.sensor2lidar_translation,
                    cam_intrinsic=sample.cam_intrinsic,
                    distortion=sample.distortion,
                )
                for cam_name, sample in frame.cameras.items()
            }

            navsim_frames.append(
                NavsimFrameRecord(
                    token=frame.token,
                    frame_idx=frame_index,
                    timestamp=frame.timestamp_ns // 1_000,
                    scene_token=scene_token,
                    log_name=log_name,
                    vehicle_name=vehicle_name,
                    log_token=log_token,
                    ego2global_translation=list(frame.ego_state.translation_xyz_m),
                    ego2global_rotation=list(frame.ego_state.rotation_wxyz),
                    ego_dynamic_state=list(frame.ego_state.dynamic_state),
                    driving_command=driving_cmd,
                    roadblock_ids=list(resolved_roadblock_ids),
                    map_location=resolved_map_location,
                    traffic_lights=[],
                    anns=_empty_annotations(),
                    cams=cams,
                    lidar_path=frame.lidar_path,
                )
            )
            prev_ego_state = frame.ego_state

        return NavsimSceneArtifact(log_name=log_name, frames=navsim_frames)


def _resolve_scene_metadata(
    *,
    frames: Sequence[CanonicalFrame],
    configured_vehicle_name: str,
) -> Tuple[str, str, str, str]:
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    start_idx = 0
    end_idx = max(0, len(frames) - 1)
    log_name = f"{timestamp}_{configured_vehicle_name}_{start_idx:05d}_{end_idx:05d}"

    vehicle_name = _extract_vehicle_name(log_name) or configured_vehicle_name
    scene_token = _stable_short_token(log_name + ":scene")
    log_token = _stable_short_token(log_name + ":log")
    return log_name, scene_token, log_token, vehicle_name


def _stable_short_token(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _extract_vehicle_name(log_name: str) -> str:
    match = re.search(r"(veh-\d+)", log_name)
    return match.group(1) if match else ""


def _empty_annotations() -> Dict[str, Any]:
    return {
        "gt_boxes": [],
        "gt_names": [],
        "gt_velocity_3d": [],
        "instance_tokens": [],
        "track_tokens": [],
    }


def _quat_to_yaw(wxyz: Tuple[float, float, float, float]) -> float:
    """Convert quaternion [w, x, y, z] to yaw angle (radians).

    Assumes small pitch and roll (near-planar motion).
    """
    w, x, y, z = wxyz
    # Yaw from quaternion: atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
    sinr_cosp = 2 * (w * z + x * y)
    cosr_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(sinr_cosp, cosr_cosp)
    return yaw


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _infer_driving_command_from_heading(
    prev_ego_state: CanonicalEgoState,
    curr_ego_state: CanonicalEgoState,
    heading_threshold: float,
    min_forward_velocity: float,
) -> List[int]:
    """Infer driving command from heading change between frames.

    More reliable than lateral acceleration when we have frame sequence.

    Returns: One-hot 4D vector [left, straight, right, unknown]
    """
    curr_vx = curr_ego_state.dynamic_state[0]  # Forward velocity

    # If stopped, mark as unknown
    if abs(curr_vx) < min_forward_velocity:
        return [0, 0, 0, 1]

    # Calculate heading change
    curr_yaw = _quat_to_yaw(curr_ego_state.rotation_wxyz)
    prev_yaw = _quat_to_yaw(prev_ego_state.rotation_wxyz)
    yaw_delta = _normalize_angle(curr_yaw - prev_yaw)

    # Classify based on heading change
    if yaw_delta > heading_threshold:
        return [1, 0, 0, 0]  # Left turn (positive yaw change)
    elif yaw_delta < -heading_threshold:
        return [0, 0, 1, 0]  # Right turn (negative yaw change)
    else:
        return [0, 1, 0, 0]  # Straight (minimal heading change)


def _infer_driving_command_from_lateral_accel(
    ego_state: CanonicalEgoState,
    lateral_accel_threshold: float,
    lateral_velocity_threshold: float,
    min_forward_velocity: float,
) -> List[int]:
    """Infer driving command from lateral acceleration and velocity.

    Used for first frame when no prior frame available.

    Returns: One-hot 4D vector [left, straight, right, unknown]
    """
    vx, vy, ax, ay = ego_state.dynamic_state

    # If stopped, mark as unknown
    if abs(vx) < min_forward_velocity:
        return [0, 0, 0, 1]

    # Detect turning intent if lateral acceleration and velocity exceed thresholds and are in the same direction.
    is_turning_intent = (ay * vy > 0) and (abs(ay) > lateral_accel_threshold) and (abs(vy) > lateral_velocity_threshold)

    if is_turning_intent:
        if vy > 0:
            return [1, 0, 0, 0]  # Left turn
        else:
            return [0, 0, 1, 0]  # Right turn
    else:
        return [0, 1, 0, 0]  # Straight


def save_navsim_assembly_manifest(scene: NavsimSceneArtifact, output_path: Path) -> None:
    """Save Stage 5 assembly summary for debugging and visibility."""
    sample_1 = None
    sample_2 = None
    if scene.frames:
        first = scene.frames[0]
        sample_1 = {
            "token": first.token,
            "timestamp": first.timestamp,
            "scene_token": first.scene_token,
            "log_name": first.log_name,
            "camera_count": len(first.cams),
            "camera_keys": sorted(first.cams.keys()),
            "lidar_path": first.lidar_path,
            "map_location": first.map_location,
            "driving_command": first.driving_command,
        }
        second = scene.frames[1] if len(scene.frames) > 1 else None
        if second:
            sample_2 = {
                "token": second.token,
                "timestamp": second.timestamp,
                "scene_token": second.scene_token,
                "log_name": second.log_name,
                "camera_count": len(second.cams),
                "camera_keys": sorted(second.cams.keys()),
                "lidar_path": second.lidar_path,
                "map_location": second.map_location,
                "driving_command": second.driving_command,
            }

    manifest_data = {
        "stage": "assemble_navsim_frames",
        "status": "ok",
        "log_name": scene.log_name,
        "frame_count": len(scene.frames),
        "sample_1": sample_1,
        "sample_2": sample_2,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)