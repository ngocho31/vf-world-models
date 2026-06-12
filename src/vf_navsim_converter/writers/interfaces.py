"""Writer interface contracts."""

from pathlib import Path
from typing import Protocol, Sequence

from ..contracts.models_navsim import NavsimFrameRecord


class SensorBlobsWriter(Protocol):
    def materialize_camera_assets(self, frames: Sequence[NavsimFrameRecord], output_root: Path) -> None:
        """Copy or symlink camera files into NAVSIM sensor_blobs layout."""
        ...


class NavsimLogsWriter(Protocol):
    def write_frames(self, frames: Sequence[NavsimFrameRecord], output_pkl_path: Path) -> None:
        """Write NAVSIM frame dictionaries to a pickle file."""
        ...
