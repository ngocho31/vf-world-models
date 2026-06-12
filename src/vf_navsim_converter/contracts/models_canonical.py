"""Canonical intermediate contracts for conversion."""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class CanonicalEgoState:
    timestamp_ns: int
    translation_xyz_m: Tuple[float, float, float]
    rotation_wxyz: Tuple[float, float, float, float]
    dynamic_state: Tuple[float, float, float, float]  # vx, vy, ax, ay


@dataclass
class CanonicalCameraSample:
    navsim_camera_name: str
    token: str
    relative_path: str
    sensor2lidar_rotation: Sequence[Sequence[float]]
    sensor2lidar_translation: Sequence[float]
    cam_intrinsic: Sequence[Sequence[float]]
    distortion: Sequence[float]


@dataclass
class CanonicalFrame:
    token: str
    timestamp_ns: int
    ego_state: CanonicalEgoState
    cameras: Dict[str, CanonicalCameraSample] = field(default_factory=dict)
    lidar_path: str = ""


@dataclass
class CanonicalBatch:
    frames: List[CanonicalFrame] = field(default_factory=list)
