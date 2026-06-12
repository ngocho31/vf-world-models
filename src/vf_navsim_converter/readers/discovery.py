"""Input discovery and manifest building for VF datasets."""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..constants import (
    VF_REQUIRED_CAMERAS,
    VF_OPTIONAL_CAMERAS,
    VF_IMAGE_EXTENSIONS,
    VF_REQUIRED_LIDARS,
    VF_OPTIONAL_LIDARS,
    VF_LIDAR_EXTENSIONS,
    VF_REQUIRED_HDMAPS,
    VF_OPTIONAL_HDMAPS,
    VF_HDMAP_EXTENSIONS,
    VF_REQUIRED_TIMESERIES_STREAMS,
    VF_OPTIONAL_TIMESERIES_STREAMS,
)

logger = logging.getLogger(__name__)


@dataclass
class CameraStreamManifest:
    """Manifest for a single camera stream."""
    sensor_name: str
    directory: Path
    frame_count: int
    frames: List[Tuple[str, str]]  # (token, file_path) pairs
    sample_frame: Optional[str] = None


@dataclass
class TimeSeriesManifest:
    """Manifest for a time series stream (NAV, IMU, etc)."""
    stream_name: str
    directory: Path
    file_count: int
    total_rows: int
    files: List[str]  # CSV file names
    sample_file: Optional[str] = None


@dataclass
class CalibrationManifest:
    """Manifest for calibration files."""
    calibration_dir: Path
    has_camera_intrinsics: bool
    has_extrinsics: bool
    intrinsics_file: Optional[str] = None
    extrinsics_file: Optional[str] = None


@dataclass
class LidarManifest:
    directory: Path
    file_count: int
    files: List[str]
    sample_file: Optional[str] = None


@dataclass
class HdMapManifest:
    directory: Path
    file_count: int
    files: List[str]
    sample_file: Optional[str] = None


@dataclass
class DiscoveryResult:
    """Result of input discovery."""
    input_root: Path
    is_valid: bool
    cameras: Dict[str, CameraStreamManifest]
    timeseries_streams: Dict[str, TimeSeriesManifest]
    calibration: Optional[CalibrationManifest]
    lidar: Optional[LidarManifest]
    hdmap: Optional[HdMapManifest]
    errors: List[str]
    warnings: List[str]


class InputDiscovery:
    """Discover and index VF dataset structure."""

    def __init__(self, input_root: Path):
        self.input_root = Path(input_root)
        self.camera_root = self.input_root / "CAMERA"
        self.others_root = self.input_root / "OTHERS"
        self.lidar_root = self.input_root / "LIDAR"

    def discover(self) -> DiscoveryResult:
        """Run full input discovery."""
        errors = []
        warnings = []

        logger.info(f"Starting input discovery at {self.input_root}")

        # Verify root exists
        if not self.input_root.exists():
            errors.append(f"Input root does not exist: {self.input_root}")
            return DiscoveryResult(
                input_root=self.input_root,
                is_valid=False,
                cameras={},
                timeseries_streams={},
                calibration=None,
                lidar=None,
                hdmap=None,
                errors=errors,
                warnings=warnings,
            )

        # Discover cameras
        cameras = self._discover_cameras(errors, warnings)

        # Discover time series streams from constants.
        timeseries_streams = self._discover_all_timeseries_streams(errors, warnings)

        # Discover calibration
        calibration = self._discover_calibration(errors, warnings)

        # Discover lidar data
        lidar = self._discover_lidar(errors, warnings)

        # Discover HD map data
        hdmap = self._discover_hdmap(errors, warnings)

        # Determine validity
        is_valid = (
            len(errors) == 0 and
            len(cameras) >= len(VF_REQUIRED_CAMERAS) and
            all(stream in timeseries_streams for stream in VF_REQUIRED_TIMESERIES_STREAMS) and
            calibration is not None and
            lidar is not None and
            hdmap is not None
        )

        result = DiscoveryResult(
            input_root=self.input_root,
            is_valid=is_valid,
            cameras=cameras,
            timeseries_streams=timeseries_streams,
            calibration=calibration,
            lidar=lidar,
            hdmap=hdmap,
            errors=errors,
            warnings=warnings,
        )

        logger.info(
            f"Discovery complete: {len(cameras)} cameras, "
            f"timeseries={len(timeseries_streams)} streams, "
            f"lidar={'found' if lidar else 'missing'}, "
            f"hdmap={'found' if hdmap else 'missing'}"
        )

        return result

    def _discover_cameras(self, errors: List[str], warnings: List[str]) -> Dict[str, CameraStreamManifest]:
        """Discover all camera streams."""
        cameras = {}

        if not self.camera_root.exists():
            errors.append(f"Camera root does not exist: {self.camera_root}")
            return cameras

        # Check required cameras
        for cam_name in VF_REQUIRED_CAMERAS:
            cam_dir = self.camera_root / cam_name
            manifest = self._build_camera_manifest(cam_dir, cam_name)

            if manifest is None:
                errors.append(f"Required camera not found: {cam_name}")
            else:
                cameras[cam_name] = manifest
                if manifest.frame_count == 0:
                    errors.append(f"Camera {cam_name} has no frames")

        # Check optional cameras
        for cam_name in VF_OPTIONAL_CAMERAS:
            cam_dir = self.camera_root / cam_name
            manifest = self._build_camera_manifest(cam_dir, cam_name)

            if manifest is not None:
                cameras[cam_name] = manifest
                if manifest.frame_count > 0:
                    logger.debug(f"Optional camera found: {cam_name} ({manifest.frame_count} frames)")

        return cameras

    def _build_camera_manifest(self, cam_dir: Path, sensor_name: str) -> Optional[CameraStreamManifest]:
        """Build manifest for a single camera directory."""
        if not cam_dir.exists():
            return None

        frames = []
        for ext in VF_IMAGE_EXTENSIONS:
            for img_path in sorted(cam_dir.glob(f"*{ext}")):
                token = img_path.stem
                frames.append((token, str(img_path.relative_to(self.input_root))))

        if not frames:
            return None

        return CameraStreamManifest(
            sensor_name=sensor_name,
            directory=cam_dir,
            frame_count=len(frames),
            frames=frames,
            sample_frame=frames[0][0] if frames else None,
        )

    def _discover_all_timeseries_streams(
        self,
        errors: List[str],
        warnings: List[str],
    ) -> Dict[str, TimeSeriesManifest]:
        """Discover all required and optional time series streams."""
        manifests: Dict[str, TimeSeriesManifest] = {}

        for stream_name in VF_REQUIRED_TIMESERIES_STREAMS:
            manifest = self._discover_timeseries(stream_name, required=True, errors=errors, warnings=warnings)
            if manifest is not None:
                manifests[stream_name] = manifest

        for stream_name in VF_OPTIONAL_TIMESERIES_STREAMS:
            manifest = self._discover_timeseries(stream_name, required=False, errors=errors, warnings=warnings)
            if manifest is not None:
                manifests[stream_name] = manifest

        return manifests

    def _discover_timeseries(
        self,
        stream_name: str,
        required: bool,
        errors: List[str],
        warnings: List[str],
    ) -> Optional[TimeSeriesManifest]:
        """Discover a time series stream (NAV, IMU, etc)."""
        stream_dir = self.others_root / stream_name

        if not stream_dir.exists():
            if required:
                errors.append(f"Stream directory not found: {stream_name}")
            else:
                warnings.append(f"Optional stream directory not found: {stream_name}")
            return None

        csv_files = list(stream_dir.glob("*.csv"))
        if not csv_files:
            if required:
                errors.append(f"No CSV files found in {stream_name}")
            else:
                warnings.append(f"Optional stream has no CSV files: {stream_name}")
            return None

        # Count rows
        total_rows = 0
        for csv_file in csv_files:
            try:
                with open(csv_file) as f:
                    # Count lines, subtract 1 for header
                    total_rows += sum(1 for _ in f) - 1
            except Exception as e:
                logger.warning(f"Could not count rows in {csv_file}: {e}")

        return TimeSeriesManifest(
            stream_name=stream_name,
            directory=stream_dir,
            file_count=len(csv_files),
            total_rows=total_rows,
            files=[f.name for f in csv_files],
            sample_file=csv_files[0].name if csv_files else None,
        )

    def _discover_calibration(self, errors: List[str], warnings: List[str]) -> Optional[CalibrationManifest]:
        """Discover calibration files."""
        # Calibration files can be in OTHERS/Calibration or at input_root level
        possible_calib_dirs = [
            self.others_root / "Calibration",
            self.input_root,  # Try root level
        ]

        calib_dir = None
        for possible_dir in possible_calib_dirs:
            if (possible_dir / "Camera_Intrinsics.json").exists() or (possible_dir / "Extrinsics.json").exists():
                calib_dir = possible_dir
                break

        if calib_dir is None:
            errors.append(f"Calibration files not found in {possible_calib_dirs[0]} or {possible_calib_dirs[1]}")
            return None

        intrinsics_file = calib_dir / "Camera_Intrinsics.json"
        extrinsics_file = calib_dir / "Extrinsics.json"

        has_intrinsics = intrinsics_file.exists()
        has_extrinsics = extrinsics_file.exists()

        if not has_intrinsics:
            errors.append("Camera_Intrinsics.json not found")
        if not has_extrinsics:
            errors.append("Extrinsics.json not found")

        return CalibrationManifest(
            calibration_dir=calib_dir,
            has_camera_intrinsics=has_intrinsics,
            has_extrinsics=has_extrinsics,
            intrinsics_file="Camera_Intrinsics.json" if has_intrinsics else None,
            extrinsics_file="Extrinsics.json" if has_extrinsics else None,
        )

    def _discover_lidar(self, errors: List[str], warnings: List[str]) -> Optional[LidarManifest]:
        """Discover lidar files from configured lidar directories."""
        found_any = False
        all_files = []
        for dir_name in VF_REQUIRED_LIDARS:
            lidar_dir = self.lidar_root / dir_name
            if not lidar_dir.exists():
                errors.append(f"Required lidar directory not found: {lidar_dir}")
                continue
            found_any = True
            for ext in VF_LIDAR_EXTENSIONS:
                all_files.extend(sorted(lidar_dir.glob(f"*{ext}")))
        for dir_name in VF_OPTIONAL_LIDARS:
            lidar_dir = self.lidar_root / dir_name
            if not lidar_dir.exists():
                continue
            found_any = True
            for ext in VF_LIDAR_EXTENSIONS:
                all_files.extend(sorted(lidar_dir.glob(f"*{ext}")))
        if all_files:
            # Use the first required dir as directory for manifest, or the first found
            manifest_dir = self.lidar_root / (VF_REQUIRED_LIDARS[0] if VF_REQUIRED_LIDARS else all_files[0].parent.name)
            return LidarManifest(
                directory=manifest_dir,
                file_count=len(all_files),
                files=[item.name for item in all_files],
                sample_file=all_files[0].name,
            )
        if not found_any:
            warnings.append("No LIDAR directories found (all optional in current bootstrap pipeline)")
        return None

    def _discover_hdmap(self, errors: List[str], warnings: List[str]) -> Optional[HdMapManifest]:
        """Discover HD map files from configured map directories."""
        found_any = False
        all_files = []
        for dir_name in VF_REQUIRED_HDMAPS:
            map_dir = self.input_root / dir_name
            if not map_dir.exists():
                errors.append(f"Required HD map directory not found: {map_dir}")
                continue
            found_any = True
            for ext in VF_HDMAP_EXTENSIONS:
                all_files.extend(sorted(map_dir.glob(f"*{ext}")))
        for dir_name in VF_OPTIONAL_HDMAPS:
            map_dir = self.input_root / dir_name
            if not map_dir.exists():
                continue
            found_any = True
            for ext in VF_HDMAP_EXTENSIONS:
                all_files.extend(sorted(map_dir.glob(f"*{ext}")))
        if all_files:
            manifest_dir = all_files[0].parent
            return HdMapManifest(
                directory=manifest_dir,
                file_count=len(all_files),
                files=[item.name for item in all_files],
                sample_file=all_files[0].name,
            )
        if not found_any:
            warnings.append("No HD map directories found (all optional in current bootstrap pipeline)")
        return None


def print_discovery_report(result: DiscoveryResult) -> None:
    """Print formatted discovery report."""
    print("\n" + "=" * 70)
    print("INPUT DISCOVERY REPORT")
    print("=" * 70)

    print(f"\nInput Root: {result.input_root}")
    print(f"Status: {'✓ VALID' if result.is_valid else '✗ INVALID'}")

    # Cameras
    print(f"\nCameras Found: {len(result.cameras)}")
    required_found = sum(1 for cam in result.cameras if cam in VF_REQUIRED_CAMERAS)
    print(f"  Required (8): {required_found}/8")
    if result.cameras:
        for cam_name in sorted(VF_REQUIRED_CAMERAS):
            if cam_name in result.cameras:
                manifest = result.cameras[cam_name]
                status = "✓"
            else:
                status = "✗"
            print(f"    {status} {cam_name}: {manifest.frame_count if cam_name in result.cameras else 'MISSING'} frames")

    optional_cameras = [c for c in result.cameras if c in VF_OPTIONAL_CAMERAS]
    if optional_cameras:
        print(f"  Optional ({len(optional_cameras)}): {', '.join(optional_cameras)}")

    # Time series
    print(f"\nTime Series:")
    for stream_name in VF_REQUIRED_TIMESERIES_STREAMS:
        manifest = result.timeseries_streams.get(stream_name)
        if manifest:
            print(f"  ✓ {stream_name}: {manifest.file_count} files, {manifest.total_rows} rows")
        else:
            print(f"  ✗ {stream_name}: MISSING")

    for stream_name in VF_OPTIONAL_TIMESERIES_STREAMS:
        manifest = result.timeseries_streams.get(stream_name)
        if manifest:
            print(f"  ✓ {stream_name}: {manifest.file_count} files, {manifest.total_rows} rows")
        else:
            print(f"  ⚠ {stream_name}: MISSING (optional)")

    # Calibration
    print(f"\nCalibration:")
    if result.calibration:
        print(f"  Directory: {result.calibration.calibration_dir}")
        if result.calibration.has_camera_intrinsics:
            print(f"    ✓ Camera_Intrinsics.json")
        else:
            print(f"    ✗ Camera_Intrinsics.json MISSING")
        if result.calibration.has_extrinsics:
            print(f"    ✓ Extrinsics.json")
        else:
            print(f"    ✗ Extrinsics.json MISSING")
    else:
        print(f"  ✗ MISSING")

    # Lidar
    print(f"\nLidar:")
    if result.lidar:
        print(f"  ✓ {result.lidar.directory.name}: {result.lidar.file_count} files")
    else:
        print(f"  ⚠ MISSING (optional in current bootstrap pipeline)")

    # HD map
    print(f"\nHD Map:")
    if result.hdmap:
        print(f"  ✓ {result.hdmap.directory.name}: {result.hdmap.file_count} files")
    else:
        print(f"  ⚠ MISSING (optional in current bootstrap pipeline)")

    # Errors and warnings
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  ✗ {error}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")

    if not result.errors and not result.warnings:
        print(f"\n✓ No errors or warnings")

    print("\n" + "=" * 70 + "\n")


def save_discovery_manifest(result: DiscoveryResult, output_path: Path) -> None:
    """Save discovery result as JSON manifest."""
    manifest = {
        "input_root": str(result.input_root),
        "timestamp": Path(output_path).stem,
        "is_valid": result.is_valid,
        "cameras": {
            name: {
                "sensor_name": manifest.sensor_name,
                "frame_count": manifest.frame_count,
                "sample_frame": manifest.sample_frame,
            }
            for name, manifest in result.cameras.items()
        },
        "timeseries_streams": {
            name: {
                "file_count": manifest.file_count,
                "total_rows": manifest.total_rows,
                "sample_file": manifest.sample_file,
            }
            for name, manifest in result.timeseries_streams.items()
        },
        "calibration": {
            "has_camera_intrinsics": result.calibration.has_camera_intrinsics,
            "has_extrinsics": result.calibration.has_extrinsics,
        } if result.calibration else None,
        "lidar": {
            "directory": str(result.lidar.directory),
            "file_count": result.lidar.file_count,
            "sample_file": result.lidar.sample_file,
        } if result.lidar else None,
        "hdmap": {
            "directory": str(result.hdmap.directory),
            "file_count": result.hdmap.file_count,
            "sample_file": result.hdmap.sample_file,
        } if result.hdmap else None,
        "errors": result.errors,
        "warnings": result.warnings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Discovery manifest saved to {output_path}")
