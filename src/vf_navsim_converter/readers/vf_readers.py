"""Concrete Stage 2 readers for VF camera, CSV, and calibration sources."""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..constants import (
    VF_HDMAP_EXTENSIONS,
    VF_IMAGE_EXTENSIONS,
    VF_LIDAR_EXTENSIONS,
    VF_REQUIRED_LIDARS,
    VF_OPTIONAL_LIDARS,
    VF_REQUIRED_HDMAPS,
    VF_OPTIONAL_HDMAPS,
)
from ..contracts.models_raw import (
    RawCalibrationBundle,
    RawCameraFrame,
    RawHdMapAsset,
    RawLidarFrame,
    RawSensorBundle,
    RawTimeSeriesRow,
)

logger = logging.getLogger(__name__)


class VFSourceReader:
    """Read raw VF inputs into Stage 2 raw contracts."""

    def __init__(self, input_root: Path):
        self.input_root = Path(input_root)
        self.camera_root = self.input_root / "CAMERA"
        self.others_root = self.input_root / "OTHERS"
        self.lidar_root = self.input_root / "LIDAR"

    def list_camera_frames(self, sensor_name: str) -> Sequence[RawCameraFrame]:
        """Return all frames for one camera sensor sorted by token."""
        sensor_dir = self.camera_root / sensor_name
        if not sensor_dir.exists():
            return []

        frames: List[RawCameraFrame] = []
        for ext in VF_IMAGE_EXTENSIONS:
            for image_path in sorted(sensor_dir.glob(f"*{ext}")):
                frames.append(
                    RawCameraFrame(
                        sensor_name=sensor_name,
                        token=image_path.stem,
                        source_path=str(image_path.relative_to(self.input_root)),
                    )
                )
        return sorted(frames, key=lambda item: item.token)

    def read_csv_stream(self, stream_name: str) -> Sequence[RawTimeSeriesRow]:
        """Read one OTHERS stream and convert all rows to RawTimeSeriesRow."""
        stream_dir = self.others_root / stream_name
        if not stream_dir.exists():
            return []

        rows: List[RawTimeSeriesRow] = []
        for csv_file in sorted(stream_dir.glob("*.csv")):
            with open(csv_file, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    timestamp_token = (row.get("Timestamp") or "").strip()
                    if not timestamp_token:
                        continue
                    rows.append(
                        RawTimeSeriesRow(
                            timestamp_ns=self._parse_timestamp_token_to_ns(timestamp_token),
                            values=row,
                        )
                    )
        return rows

    def read_calibration(self) -> RawCalibrationBundle:
        """Read camera intrinsics and sensor extrinsics from JSON files."""
        calib_dir = self._resolve_calibration_dir()
        if calib_dir is None:
            raise FileNotFoundError(
                "Calibration files not found in OTHERS/Calibration or input root"
            )

        intrinsics_path = calib_dir / "Camera_Intrinsics.json"
        extrinsics_path = calib_dir / "Extrinsics.json"

        if not intrinsics_path.exists():
            raise FileNotFoundError(f"Missing calibration file: {intrinsics_path}")
        if not extrinsics_path.exists():
            raise FileNotFoundError(f"Missing calibration file: {extrinsics_path}")

        with open(intrinsics_path, encoding="utf-8") as handle:
            intrinsics_data = json.load(handle)
        with open(extrinsics_path, encoding="utf-8") as handle:
            extrinsics_data = json.load(handle)

        return RawCalibrationBundle(
            intrinsics=intrinsics_data,
            extrinsics=extrinsics_data,
        )

    def list_lidar_frames(self) -> Sequence[RawLidarFrame]:
        """Return all LIDAR frames with parsed timestamp for alignment."""
        frames: List[RawLidarFrame] = []
        for dir_name in VF_REQUIRED_LIDARS + VF_OPTIONAL_LIDARS:
            lidar_dir = self.lidar_root / dir_name
            if not lidar_dir.exists():
                continue

            for ext in VF_LIDAR_EXTENSIONS:
                for file_path in sorted(lidar_dir.glob(f"*{ext}")):
                    token = file_path.stem
                    frames.append(
                        RawLidarFrame(
                            token=token,
                            timestamp_ns=self._parse_timestamp_token_to_ns(token),
                            source_path=str(file_path.relative_to(self.input_root)),
                            sensor_name=dir_name,
                        )
                    )
        return sorted(frames, key=lambda item: item.timestamp_ns)

    def list_hdmap_assets(self) -> Sequence[RawHdMapAsset]:
        """Return all HD map assets from configured map dirs."""
        assets: List[RawHdMapAsset] = []
        for dir_name in VF_REQUIRED_HDMAPS + VF_OPTIONAL_HDMAPS:
            hdmap_dir = self.input_root / dir_name
            if not hdmap_dir.exists():
                continue

            for ext in VF_HDMAP_EXTENSIONS:
                for file_path in sorted(hdmap_dir.glob(f"*{ext}")):
                    assets.append(
                        RawHdMapAsset(
                            map_id=dir_name,
                            source_path=str(file_path.relative_to(self.input_root)),
                        )
                    )
        return assets

    def _resolve_calibration_dir(self) -> Optional[Path]:
        for candidate in (self.others_root / "Calibration", self.input_root):
            if (candidate / "Camera_Intrinsics.json").exists() or (candidate / "Extrinsics.json").exists():
                return candidate
        return None

    def _parse_timestamp_token_to_ns(self, token: str) -> int:
        """Parse VF token format '<seconds>-<subsecond>' to nanoseconds."""
        try:
            sec_str, subsec_str = token.split("-", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp token: {token}") from exc

        if not sec_str.isdigit():
            raise ValueError(f"Invalid second component in token: {token}")

        # VF subsecond precision varies by source file; normalize to nanoseconds.
        subsec_digits = "".join(ch for ch in subsec_str if ch.isdigit())
        if not subsec_digits:
            subsec_digits = "0"

        ns_component = int((subsec_digits + "000000000")[:9])
        return int(sec_str) * 1_000_000_000 + ns_component


def save_raw_sensor_bundle_manifest(bundle: RawSensorBundle, output_path: Path) -> None:
    """Save a JSON manifest of the RawSensorBundle for debugging and visibility."""
    camera_samples = {}
    for sensor, frames in bundle.camera_frames_by_sensor.items():
        if frames:
            camera_samples[sensor] = {
                "token": frames[0].token,
                "source_path": frames[0].source_path,
            }
        else:
            camera_samples[sensor] = None

    nav_sample = None
    if bundle.nav_rows:
        nav_sample = {
            "timestamp_ns": bundle.nav_rows[0].timestamp_ns,
            "values": bundle.nav_rows[0].values,
        }

    imu_sample = None
    if bundle.imu_rows:
        imu_sample = {
            "timestamp_ns": bundle.imu_rows[0].timestamp_ns,
            "values": bundle.imu_rows[0].values,
        }

    vehicle_info_sample = None
    if bundle.vehicle_info_rows:
        vehicle_info_sample = {
            "timestamp_ns": bundle.vehicle_info_rows[0].timestamp_ns,
            "values": bundle.vehicle_info_rows[0].values,
        }

    steer_sample = None
    if bundle.steer_rows:
        steer_sample = {
            "timestamp_ns": bundle.steer_rows[0].timestamp_ns,
            "values": bundle.steer_rows[0].values,
        }

    lidar_sample = None
    if hasattr(bundle, "lidar_frames") and bundle.lidar_frames:
        lidar_sample = {
            "sensor_name": bundle.lidar_frames[0].sensor_name,
            "token": bundle.lidar_frames[0].token,
            "timestamp_ns": bundle.lidar_frames[0].timestamp_ns,
            "source_path": bundle.lidar_frames[0].source_path,
        }
    hdmap_sample = None
    if hasattr(bundle, "hdmap_assets") and bundle.hdmap_assets:
        hdmap_sample = {
            "map_id": bundle.hdmap_assets[0].map_id,
            "source_path": bundle.hdmap_assets[0].source_path,
        }

    manifest_data = {
        "camera_rows_count": sum(len(frames) for frames in bundle.camera_frames_by_sensor.values()),
        "camera_samples": camera_samples,
        "nav_rows_count": len(bundle.nav_rows),
        "nav_sample": nav_sample,
        "imu_rows_count": len(bundle.imu_rows),
        "imu_sample": imu_sample,
        "vehicle_info_rows_count": len(bundle.vehicle_info_rows),
        "vehicle_info_sample": vehicle_info_sample,
        "steer_rows_count": len(bundle.steer_rows),
        "steer_sample": steer_sample,
        "calibration_intrinsics_keys": list(bundle.calibration.intrinsics.keys()),
        "calibration_extrinsics_keys": list(bundle.calibration.extrinsics.keys()),
        "lidar_rows_count": len(getattr(bundle, "lidar_frames", [])),
        "lidar_sample": lidar_sample,
        "hdmap_assets_count": len(getattr(bundle, "hdmap_assets", [])),
        "hdmap_sample": hdmap_sample,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)

    logger.info(f"Raw sensor bundle manifest saved to {output_path}")
