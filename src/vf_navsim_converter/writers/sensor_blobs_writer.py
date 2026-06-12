"""Default implementation for materializing camera and lidar assets (Stage 6)."""

import os
import shutil
from pathlib import Path
from typing import Sequence

from ..contracts.models_navsim import NavsimFrameRecord
from .interfaces import SensorBlobsWriter


class DefaultSensorBlobsWriter:
    def __init__(
        self,
        *,
        input_root: Path,
        link_mode: str = "symlink",
        split_name: str = "trainval",
        lidar_policy: str = "permissive",
    ) -> None:
        self._input_root = Path(input_root)
        self._link_mode = link_mode
        self._split_name = split_name
        self._lidar_policy = lidar_policy

    def materialize_camera_assets(self, frames: Sequence[NavsimFrameRecord], output_root: Path) -> None:
        """Materialize camera and lidar assets into NAVSIM sensor_blobs layout.

        Target layout:
        - sensor_blobs/trainval/<log_name>/CAM_<X>/<frame_token>.<ext>
        - sensor_blobs/trainval/<log_name>/MergedPointCloud/<frame_token>.<ext>

        The method also rewrites `cams[*].data_path` and `lidar_path` in-place so
        later stages persist NAVSIM-compatible relative paths into the pickle.
        """
        for frame in frames:
            scene_root = output_root / "sensor_blobs" / self._split_name / frame.log_name
            scene_root.mkdir(parents=True, exist_ok=True)

            for cam_key, cam in frame.cams.items():
                source_path = self._resolve_source_path(cam.data_path)
                target_name = Path(cam.data_path).name or f"{frame.token}.jpg"
                target_relative = Path(frame.log_name) / cam_key / target_name
                target_path = output_root / "sensor_blobs" / self._split_name / target_relative

                self._materialize_one(source_path, target_path, required=True)
                cam.data_path = target_relative.as_posix()

            frame.lidar_path = self._materialize_lidar_asset(frame=frame, output_root=output_root).as_posix()

    def _materialize_lidar_asset(self, *, frame: NavsimFrameRecord, output_root: Path) -> Path:
        source_token_path = Path(frame.lidar_path) if frame.lidar_path else Path()
        source_path = self._resolve_source_path(frame.lidar_path) if frame.lidar_path else None
        lidar_suffix = source_token_path.suffix or ".pcd"

        target_relative = Path(frame.log_name) / "MergedPointCloud" / f"{frame.token}{lidar_suffix}"
        target_path = output_root / "sensor_blobs" / self._split_name / target_relative

        if source_path is None or not source_path.exists():
            if self._lidar_policy == "strict":
                raise FileNotFoundError(f"Missing lidar source for frame {frame.token}: {frame.lidar_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            return target_relative

        self._materialize_one(source_path, target_path, required=False)
        return target_relative

    def _resolve_source_path(self, source_path: str) -> Path:
        candidate = Path(source_path)
        if candidate.is_absolute():
            return candidate
        return self._input_root / candidate

    def _materialize_one(self, source_path: Path, target_path: Path, *, required: bool) -> None:
        if not source_path.exists():
            if required:
                raise FileNotFoundError(f"Missing source asset: {source_path}")
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.exists() or target_path.is_symlink():
            if self._is_up_to_date(source_path, target_path):
                return
            target_path.unlink()

        if self._link_mode == "copy":
            shutil.copy2(source_path, target_path)
            return

        os.symlink(source_path.resolve(), target_path)

    @staticmethod
    def _is_up_to_date(source_path: Path, target_path: Path) -> bool:
        if target_path.is_symlink():
            try:
                return target_path.resolve() == source_path.resolve()
            except FileNotFoundError:
                return False
        return target_path.exists()
