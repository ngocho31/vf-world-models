"""Raw VF input data contracts."""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence


@dataclass
class RawCameraFrame:
    sensor_name: str
    token: str
    source_path: str


@dataclass
class RawLidarFrame:
    token: str
    timestamp_ns: int
    source_path: str
    sensor_name: str = ""


@dataclass
class RawHdMapAsset:
    map_id: str
    source_path: str


@dataclass
class RawTimeSeriesRow:
    timestamp_ns: int
    values: Dict[str, str]


@dataclass
class RawCalibrationBundle:
    intrinsics: Mapping[str, Mapping[str, object]]
    extrinsics: Mapping[str, Sequence[float]]


@dataclass
class RawSensorBundle:
    camera_frames_by_sensor: Dict[str, List[RawCameraFrame]] = field(default_factory=dict)
    lidar_frames: List[RawLidarFrame] = field(default_factory=list)
    hdmap_assets: List[RawHdMapAsset] = field(default_factory=list)
    nav_rows: List[RawTimeSeriesRow] = field(default_factory=list)
    imu_rows: List[RawTimeSeriesRow] = field(default_factory=list)
    vehicle_info_rows: List[RawTimeSeriesRow] = field(default_factory=list)
    steer_rows: List[RawTimeSeriesRow] = field(default_factory=list)
    calibration: RawCalibrationBundle = field(default_factory=lambda: RawCalibrationBundle({}, {}))
