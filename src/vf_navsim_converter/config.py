"""Configuration dataclasses for VF to NAVSIM conversion."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    input_root: Path
    output_root: Path
    vehicle_name: str = "veh-01"
    map_location: str = "vn-hdmap-demo"
    map_version: str | None = None
    map_sync_min_overlap: float = 0.8


@dataclass
class AlignmentConfig:
    max_sensor_nav_drift_ms: int = 80
    frame_stride: int = 1
    max_frames: int = 0


@dataclass
class LidarConfig:
    policy: str = "permissive"  # permissive | strict
    reference_sensor: str = "LIDAR_TOP"


@dataclass
class IOConfig:
    link_mode: str = "symlink"  # symlink | copy


@dataclass
class ValidationConfig:
    fail_on_schema_error: bool = True
    fail_on_missing_assets: bool = True


@dataclass
class ConversionConfig:
    pipeline: PipelineConfig
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    io: IOConfig = field(default_factory=IOConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass
class HdMapConversionConfig:
    input_root: Path
    output_root: Path
    map_location: str = "vn-hdmap-demo"
    map_version: str | None = None
    output_filename: str = "map.gpkg"
    source_map_filename: str = "lanelet2_map.osm"
