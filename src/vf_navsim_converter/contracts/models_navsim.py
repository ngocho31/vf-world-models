"""NAVSIM-compatible output contracts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class NavsimCameraRecord:
    data_path: str
    sensor2lidar_rotation: Sequence[Sequence[float]]
    sensor2lidar_translation: Sequence[float]
    cam_intrinsic: Sequence[Sequence[float]]
    distortion: Sequence[float]


@dataclass
class NavsimFrameRecord:
    token: str
    frame_idx: int
    timestamp: int
    scene_token: str
    log_name: str
    vehicle_name: str
    ego2global_translation: Sequence[float]
    ego2global_rotation: Sequence[float]
    ego_dynamic_state: Sequence[float]
    driving_command: Sequence[int]
    roadblock_ids: List[str]
    map_location: str
    traffic_lights: List[Tuple[Union[str, int], bool]]
    anns: Dict[str, Any]
    log_token: str = ""
    scene_name: str = ""
    cams: Dict[str, NavsimCameraRecord] = field(default_factory=dict)
    lidar_path: str = ""


@dataclass
class NavsimSceneArtifact:
    log_name: str
    frames: List[NavsimFrameRecord] = field(default_factory=list)


@dataclass
class SceneMapSyncContext:
    map_location: str
    roadblock_ids: List[str] = field(default_factory=list)
    overlap_ratio: float = 0.0
