"""Reader interface contracts."""

from typing import Mapping, Protocol, Sequence

from ..contracts.models_raw import RawCameraFrame, RawCalibrationBundle, RawTimeSeriesRow


class CameraReader(Protocol):
    def list_camera_frames(self, sensor_name: str) -> Sequence[RawCameraFrame]:
        """Return frames for one VF camera stream."""
        ...


class TimeSeriesReader(Protocol):
    def read_csv_stream(self, stream_name: str) -> Sequence[RawTimeSeriesRow]:
        """Read one VF OTHERS stream such as NAV or IMU."""
        ...


class CalibrationReader(Protocol):
    def read_calibration(self) -> RawCalibrationBundle:
        """Read camera intrinsics and cross-sensor extrinsics."""
        ...
