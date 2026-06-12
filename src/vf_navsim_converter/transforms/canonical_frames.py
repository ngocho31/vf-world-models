"""Canonical frame assembly for Stage 4."""

import json
import logging
import math
import time
from bisect import bisect_left
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from pyproj import Transformer as _ProjTransformer

from ..constants import VF_REQUIRED_CAMERAS, VF_TO_NAVSIM_CAMERA_MAP
from ..contracts.models_canonical import (
    CanonicalBatch,
    CanonicalCameraSample,
    CanonicalEgoState,
    CanonicalFrame,
)
from ..contracts.models_raw import RawLidarFrame, RawSensorBundle, RawTimeSeriesRow

import hashlib
import pickle

CACHE_DIR = Path(".cache/stage4")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def _token_to_ns(token: str) -> int:
    try:
        sec_str, subsec_str = token.split("-", 1)
    except ValueError as exc:
        raise ValueError("Invalid timestamp token: %s" % token) from exc

    if not sec_str.isdigit():
        raise ValueError("Invalid second component in token: %s" % token)

    subsec_digits = "".join(ch for ch in subsec_str if ch.isdigit())
    if not subsec_digits:
        subsec_digits = "0"

    ns_component = int((subsec_digits + "000000000")[:9])
    return int(sec_str) * 1_000_000_000 + ns_component


def _nearest_row(rows: Sequence[RawTimeSeriesRow], target_ns: int) -> RawTimeSeriesRow:
    if not rows:
        raise ValueError("Rows cannot be empty")

    timestamps = [row.timestamp_ns for row in rows]
    idx = bisect_left(timestamps, target_ns)

    if idx <= 0:
        return rows[0]
    if idx >= len(rows):
        return rows[-1]

    before = rows[idx - 1]
    after = rows[idx]
    if abs(before.timestamp_ns - target_ns) <= abs(after.timestamp_ns - target_ns):
        return before
    return after


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_value(mapping: Mapping[str, object], keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        if key in mapping:
            return _to_float(mapping.get(key), default=default)
    return default


def _yaw_to_quaternion_wxyz(yaw_rad: float) -> Tuple[float, float, float, float]:
    half = yaw_rad / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _transpose_3x3(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]


def _matmul_3x3(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = (
                a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
            )
    return out


def _matvec_3x3(matrix: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    return [
        matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2],
        matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2],
        matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2],
    ]


def _normalize_angle_unit(raw_angle: float) -> float:
    # If the magnitude is clearly degree-like, convert to radians.
    if abs(raw_angle) > 2.0 * math.pi:
        return math.radians(raw_angle)
    return raw_angle


def _extract_pose6(values: object) -> Tuple[float, float, float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) < 6:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    x = _to_float(values[0])
    y = _to_float(values[1])
    z = _to_float(values[2])
    roll = _normalize_angle_unit(_to_float(values[3]))
    pitch = _normalize_angle_unit(_to_float(values[4]))
    yaw = _normalize_angle_unit(_to_float(values[5]))
    return (x, y, z, roll, pitch, yaw)


def _extract_intrinsic_bundle(
    calibration_intrinsics: Mapping[str, Mapping[str, object]],
    sensor_name: str,
) -> Tuple[List[List[float]], List[float]]:
    entry = calibration_intrinsics.get(sensor_name, {})

    raw_matrix = entry.get("camera_matrix") if isinstance(entry, Mapping) else None
    raw_distortion = entry.get("distortion_coefficients") if isinstance(entry, Mapping) else None

    matrix: List[List[float]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    if isinstance(raw_matrix, (list, tuple)) and len(raw_matrix) >= 3:
        parsed_rows: List[List[float]] = []
        for row in raw_matrix[:3]:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                parsed_rows.append([_to_float(row[0]), _to_float(row[1]), _to_float(row[2])])
        if len(parsed_rows) == 3:
            matrix = parsed_rows

    distortion: List[float] = []
    if isinstance(raw_distortion, (list, tuple)):
        distortion = [_to_float(v) for v in raw_distortion]

    return matrix, distortion


def _build_sensor_to_lidar_transform(
    extrinsics: Mapping[str, Sequence[float]],
    sensor_name: str,
    lidar_sensor: str,
) -> Tuple[List[List[float]], List[float]]:
    sensor_pose = _extract_pose6(extrinsics.get(sensor_name))
    lidar_pose = _extract_pose6(extrinsics.get(lidar_sensor))

    sensor_t = [sensor_pose[0], sensor_pose[1], sensor_pose[2]]
    lidar_t = [lidar_pose[0], lidar_pose[1], lidar_pose[2]]

    sensor_r = _rpy_to_matrix(sensor_pose[3], sensor_pose[4], sensor_pose[5])
    lidar_r = _rpy_to_matrix(lidar_pose[3], lidar_pose[4], lidar_pose[5])
    lidar_r_inv = _transpose_3x3(lidar_r)

    sensor2lidar_rotation = _matmul_3x3(lidar_r_inv, sensor_r)
    delta = [
        sensor_t[0] - lidar_t[0],
        sensor_t[1] - lidar_t[1],
        sensor_t[2] - lidar_t[2],
    ]
    sensor2lidar_translation = _matvec_3x3(lidar_r_inv, delta)
    return sensor2lidar_rotation, sensor2lidar_translation


def _compute_data_checksum(data: RawSensorBundle, tokens: Sequence[str]) -> str:
    """Compute a checksum for the input data to detect changes."""
    hasher = hashlib.sha256()
    for token in tokens:
        hasher.update(token.encode())
    for lidar_frame in data.lidar_frames:
        hasher.update(lidar_frame.sensor_name.encode())
        hasher.update(str(lidar_frame.timestamp_ns).encode())
    for camera_frames in data.camera_frames_by_sensor.values():
        for frame in camera_frames:
            hasher.update(frame.token.encode())
            hasher.update(frame.source_path.encode())
    for key, value in data.calibration.extrinsics.items():
        hasher.update(key.encode())
        hasher.update(str(value).encode())
    return hasher.hexdigest()


class DefaultCanonicalFrameBuilder:
    """Build canonical frames from aligned tokens and raw sensor bundle.

    Lidar selection strategy:
    - Primary (default): LIDAR_TOP (roof lidar) — 360° coverage, most stable, nuPlan-compatible
    - Fallback: LIDAR_E_F → LIDAR_E_B → LIDAR_E_L/R (ego-frame lidar in priority order)
    - Last resort: Any available lidar

    Performance optimization:
    - Pre-compute and cache all sensor2lidar transforms at init time
    - Pre-build lidar index by sensor for fast lookup
    - Stream-based frame building with progress logging
    """

    def __init__(self, reference_lidar_sensor: str = "LIDAR_TOP") -> None:
        self._reference_lidar_sensor = reference_lidar_sensor
        self._calibration_cache: Dict[str, Tuple[List[List[float]], List[float]]] = {}
        self._lidar_index_by_sensor: Dict[str, List[object]] = {}
        self._lidar_timestamps_by_sensor: Dict[str, List[int]] = {}

    def build(self, data: RawSensorBundle, tokens: Sequence[str]) -> CanonicalBatch:
        if not tokens:
            logger.info("No tokens to process, returning empty batch")
            return CanonicalBatch(frames=[])

        logger.info(f"Stage 4 starting: processing {len(tokens)} frames")
        start_time = time.time()

        # Pre-build all caches for fast per-frame access
        logger.debug("Pre-computing calibration cache (8 sensors × 2 reference frames = 16 transforms)")
        self._build_calibration_cache(data.calibration)

        logger.debug("Building lidar index by sensor")
        self._build_lidar_index(data.lidar_frames)

        camera_index = self._index_camera_frames(data.camera_frames_by_sensor)
        nav_anchor = self._build_nav_anchor(data.nav_rows)

        frames: List[CanonicalFrame] = []
        for idx, token in enumerate(tokens):
            # Progress logging every 100 frames or at end
            if (idx + 1) % 100 == 0 or idx == len(tokens) - 1:
                elapsed = time.time() - start_time
                avg_per_frame = elapsed / (idx + 1)
                remaining_frames = len(tokens) - idx - 1
                eta_seconds = remaining_frames * avg_per_frame
                logger.info(
                    f"Progress: {idx + 1}/{len(tokens)} frames "
                    f"({100 * (idx + 1) / len(tokens):.1f}%) | "
                    f"elapsed: {elapsed:.1f}s | "
                    f"ETA: {eta_seconds:.1f}s"
                )

            timestamp_ns = _token_to_ns(token)
            ego_state = self._build_ego_state(
                timestamp_ns=timestamp_ns,
                nav_rows=data.nav_rows,
                imu_rows=data.imu_rows,
                nav_anchor=nav_anchor,
            )

            cameras: Dict[str, CanonicalCameraSample] = {}
            for vf_sensor in VF_REQUIRED_CAMERAS:
                frame_entry = camera_index.get(vf_sensor, {}).get(token)
                if frame_entry is None:
                    raise ValueError("Missing camera frame for sensor=%s token=%s" % (vf_sensor, token))

                navsim_camera_name = VF_TO_NAVSIM_CAMERA_MAP[vf_sensor]
                # Use cached transforms instead of recomputing
                sensor2lidar_rotation, sensor2lidar_translation = self._calibration_cache[vf_sensor]

                # Get cached intrinsics
                cam_intrinsic, distortion = _extract_intrinsic_bundle(data.calibration.intrinsics, vf_sensor)

                cameras[navsim_camera_name] = CanonicalCameraSample(
                    navsim_camera_name=navsim_camera_name,
                    token=token,
                    relative_path=frame_entry.source_path,
                    sensor2lidar_rotation=sensor2lidar_rotation,
                    sensor2lidar_translation=sensor2lidar_translation,
                    cam_intrinsic=cam_intrinsic,
                    distortion=distortion,
                )

            lidar_path = self._select_lidar_path_fast(token_ns=timestamp_ns)

            frames.append(
                CanonicalFrame(
                    token=token,
                    timestamp_ns=timestamp_ns,
                    ego_state=ego_state,
                    cameras=cameras,
                    lidar_path=lidar_path,
                )
            )

        total_time = time.time() - start_time
        logger.info(
            f"Stage 4 complete: {len(frames)} frames in {total_time:.2f}s "
            f"({1000 * total_time / len(frames):.1f}ms per frame)"
        )
        return CanonicalBatch(frames=frames)

    def build_with_cache(self, data: RawSensorBundle, tokens: Sequence[str], cache_key: str = None) -> CanonicalBatch:
        """Build canonical frames with caching support."""
        if cache_key is None:
            cache_key = _compute_data_checksum(data, tokens)

        cache_path = CACHE_DIR / f"{cache_key}.pkl"
        if cache_path.exists():
            logger.info(f"Cache hit: Loading Stage 4 results from {cache_path}")
            with open(cache_path, "rb") as cache_file:
                return pickle.load(cache_file)

        logger.info("Cache miss: Running Stage 4 pipeline")
        result = self.build(data, tokens)

        logger.info(f"Saving Stage 4 results to cache: {cache_path}")
        with open(cache_path, "wb") as cache_file:
            pickle.dump(result, cache_file)

        return result

    def _build_calibration_cache(self, calibration: object) -> None:
        """Pre-compute sensor2lidar transforms for all required cameras.

        This avoids redundant RPY→matrix conversions and matrix multiplications
        per frame. Each sensor's transform to reference lidar is computed once.
        """
        extrinsics = getattr(calibration, "extrinsics", {})
        for vf_sensor in VF_REQUIRED_CAMERAS:
            rotation, translation = _build_sensor_to_lidar_transform(
                extrinsics,
                vf_sensor,
                self._reference_lidar_sensor,
            )
            self._calibration_cache[vf_sensor] = (rotation, translation)
        logger.debug(f"Cached {len(self._calibration_cache)} sensor calibrations")

    def _build_lidar_index(self, lidar_frames: Sequence[object]) -> None:
        """Build lidar index by sensor for fast per-frame lookup.

        Organizes lidar frames by sensor_name so we can quickly find frames
        for specific sensors without iterating all lidar frames each time.
        """
        for frame in lidar_frames:
            sensor_name = getattr(frame, "sensor_name", "unknown")
            if sensor_name not in self._lidar_index_by_sensor:
                self._lidar_index_by_sensor[sensor_name] = []
                self._lidar_timestamps_by_sensor[sensor_name] = []
            self._lidar_index_by_sensor[sensor_name].append(frame)
            self._lidar_timestamps_by_sensor[sensor_name].append(getattr(frame, "timestamp_ns", 0))
        logger.debug(f"Indexed lidar frames: {list(self._lidar_index_by_sensor.keys())}")

    def _select_lidar_path_fast(self, token_ns: int) -> str:
        """Fast lidar path selection using pre-built indices.

        Uses fallback chain to find nearest frame for preferred lidar sensor,
        without rebuilding indices each time.
        """
        if not self._lidar_index_by_sensor:
            return ""

        # Build fallback chain using available indexed sensors
        available_sensors = set(self._lidar_index_by_sensor.keys())
        fallback_chain = self._get_lidar_fallback_chain(available_sensors)

        # Try each sensor in fallback chain
        for sensor_name in fallback_chain:
            frames = self._lidar_index_by_sensor.get(sensor_name)
            timestamps = self._lidar_timestamps_by_sensor.get(sensor_name)

            if not frames or not timestamps:
                continue

            idx = bisect_left(timestamps, token_ns)

            if idx <= 0:
                return getattr(frames[0], "source_path", "")
            if idx >= len(frames):
                return getattr(frames[-1], "source_path", "")

            before = frames[idx - 1]
            after = frames[idx]
            before_ts = getattr(before, "timestamp_ns", 0)
            after_ts = getattr(after, "timestamp_ns", 0)

            if abs(before_ts - token_ns) <= abs(after_ts - token_ns):
                return getattr(before, "source_path", "")
            return getattr(after, "source_path", "")

        return ""

    @staticmethod
    def _index_camera_frames(
        camera_frames_by_sensor: Mapping[str, Sequence[object]],
    ) -> Dict[str, Dict[str, object]]:
        indexed: Dict[str, Dict[str, object]] = {}
        for sensor_name, frames in camera_frames_by_sensor.items():
            token_map: Dict[str, object] = {}
            for frame in frames:
                token = getattr(frame, "token", "")
                if token:
                    token_map[token] = frame
            indexed[sensor_name] = token_map
        return indexed

    @staticmethod
    def _build_nav_anchor(nav_rows: Sequence[RawTimeSeriesRow]) -> MutableMapping[str, float]:
        if not nav_rows:
            return {"lat": 0.0, "lon": 0.0, "alt": 0.0}

        first = nav_rows[0].values
        return {
            "lat": _first_value(first, ("Latitude", "latitude", "lat"), default=0.0),
            "lon": _first_value(first, ("Longitude", "longitude", "lon"), default=0.0),
            "alt": _first_value(first, ("Altitude", "altitude", "alt", "Height"), default=0.0),
        }

    @staticmethod
    def _build_ego_state(
        timestamp_ns: int,
        nav_rows: Sequence[RawTimeSeriesRow],
        imu_rows: Sequence[RawTimeSeriesRow],
        nav_anchor: Mapping[str, float],
    ) -> CanonicalEgoState:
        nav_row = _nearest_row(nav_rows, timestamp_ns) if nav_rows else RawTimeSeriesRow(timestamp_ns, {})
        imu_row = _nearest_row(imu_rows, timestamp_ns) if imu_rows else RawTimeSeriesRow(timestamp_ns, {})

        nav_values = nav_row.values
        imu_values = imu_row.values

        lat = _first_value(nav_values, ("Latitude", "latitude", "lat"), default=nav_anchor["lat"])
        lon = _first_value(nav_values, ("Longitude", "longitude", "lon"), default=nav_anchor["lon"])
        alt = _first_value(nav_values, ("Altitude", "altitude", "alt", "Height"), default=nav_anchor["alt"])

        _tf = _ProjTransformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
        x, y = _tf.transform(lon, lat)
        z = alt - nav_anchor["alt"]

        heading = _first_value(nav_values, ("Heading2", "Heading", "heading", "Yaw", "yaw"), default=0.0)
        heading = _normalize_angle_unit(heading)
        rotation = _yaw_to_quaternion_wxyz(heading)

        vx = _first_value(nav_values, ("Ve", "ve", "vx", "Vx"), default=0.0)
        vy = _first_value(nav_values, ("Vn", "vn", "vy", "Vy"), default=0.0)
        ax = _first_value(imu_values, ("AccX", "acc_x", "ax", "Ax"), default=0.0)
        ay = _first_value(imu_values, ("AccY", "acc_y", "ay", "Ay"), default=0.0)

        return CanonicalEgoState(
            timestamp_ns=timestamp_ns,
            translation_xyz_m=(x, y, z),
            rotation_wxyz=rotation,
            dynamic_state=(vx, vy, ax, ay),
        )

    def _get_lidar_fallback_chain(self, available_sensors: set) -> List[str]:
        """Build fallback chain for lidar selection.

        Priority order:
        1. LIDAR_TOP (roof, 360° coverage, nuPlan-standard)
        2. LIDAR_E_F (front, most important for autonomous driving)
        3. LIDAR_E_B (back, secondary direction)
        4. LIDAR_E_L/R (side, least preferred but useful if all else fails)
        5. Any remaining sensor
        """
        chain = []

        # Primary: requested reference sensor (usually LIDAR_TOP)
        if self._reference_lidar_sensor in available_sensors:
            chain.append(self._reference_lidar_sensor)

        # Secondary preferences (standard fallback chain)
        for sensor in ["LIDAR_TOP", "LIDAR_E_F", "LIDAR_E_B", "LIDAR_E_L", "LIDAR_E_R"]:
            if sensor not in chain and sensor in available_sensors:
                chain.append(sensor)

        # Last resort: any remaining sensor
        for sensor in sorted(available_sensors):
            if sensor not in chain:
                chain.append(sensor)

        return chain


def save_canonical_manifest(batch: CanonicalBatch, output_path: Path) -> None:
    """Save Stage 4 canonical frame summary for debugging and visibility."""
    sample = None
    if batch.frames:
        first = batch.frames[0]
        sample = {
            "token": first.token,
            "timestamp_ns": first.timestamp_ns,
            "ego_state": {
                "translation_xyz_m": first.ego_state.translation_xyz_m,
                "rotation_wxyz": first.ego_state.rotation_wxyz,
                "dynamic_state": first.ego_state.dynamic_state,
            },
            "camera_count": len(first.cameras),
            "camera_keys": sorted(first.cameras.keys()),
            "camera": {
                k: {
                    "navsim_camera_name": v.navsim_camera_name,
                    "token": v.token,
                    "relative_path": v.relative_path,
                    "sensor2lidar_rotation": v.sensor2lidar_rotation,
                    "sensor2lidar_translation": v.sensor2lidar_translation,
                    "cam_intrinsic": v.cam_intrinsic,
                    "distortion": v.distortion,
                }
                for k, v in first.cameras.items()
            },
            "lidar_path": first.lidar_path,
        }

    manifest_data = {
        "stage": "build_canonical_frames",
        "status": "ok",
        "frame_count": len(batch.frames),
        "sample": sample,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)
