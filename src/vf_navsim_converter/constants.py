"""Constants for VF to NAVSIM conversion."""

from typing import Dict, Tuple

# Required panoramic VF cameras for NAVSIM-compatible 8-camera export.
VF_REQUIRED_CAMERAS: Tuple[str, ...] = (
    "CAM_P_F",
    "CAM_P_FL",
    "CAM_P_L",
    "CAM_P_LB",
    "CAM_P_FR",
    "CAM_P_R",
    "CAM_P_RB",
    "CAM_P_B",
)
# Optional fish-eye cameras that are currently excluded from NAVSIM conversion.
VF_OPTIONAL_CAMERAS: Tuple[str, ...] = (
    "CAM_F_F",
    "CAM_F_L",
    "CAM_F_R",
    "CAM_F_B",
)
VF_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Required Lidar directories that must be present in VF data for NAVSIM conversion.
VF_REQUIRED_LIDARS: Tuple[str, ...] = (
    "LIDAR_E_F",
    "LIDAR_E_L",
    "LIDAR_E_R",
    "LIDAR_E_B",
    "LIDAR_TOP",
)
# Optional Lidar directories that may be present in VF data but not required.
VF_OPTIONAL_LIDARS: Tuple[str, ...] = (
    "MergedPointCloud",
)
VF_LIDAR_EXTENSIONS: Tuple[str, ...] = (".pcd", ".bin", ".ply", ".las", ".laz")

# Required HD map assets that must be present in VF data for NAVSIM conversion.
VF_REQUIRED_HDMAPS: Tuple[str, ...] = (
    "vn-hdmap-demo",
)
VF_OPTIONAL_HDMAPS: Tuple[str, ...] = ()
VF_HDMAP_EXTENSIONS: Tuple[str, ...] = (".json", ".geojson", ".pkl", ".bin", ".osm", ".xodr")

# Required NAV timeseries stream for timestamp alignment.
VF_REQUIRED_TIMESERIES_STREAMS: Tuple[str, ...] = ("NAV",)
# Optional timeseries streams are included when present but not required.
VF_OPTIONAL_TIMESERIES_STREAMS: Tuple[str, ...] = (
    "IMU",
    "VEHICLE_INFO",
    "VEHICLE_STEER",
)

# Mapping from VF timeseries stream names to RawSensorBundle field names.
VF_STREAM_TO_BUNDLE_FIELD: Dict[str, str] = {
    "NAV": "nav_rows",
    "IMU": "imu_rows",
    "VEHICLE_INFO": "vehicle_info_rows",
    "VEHICLE_STEER": "steer_rows",
}

# Required fields must exist and be non-empty; optional fields are used only when present.
VF_REQUIRED_ALIGNMENT_BUNDLE_FIELDS: Tuple[str, ...] = (
    "nav_rows",
    "imu_rows",
    "vehicle_info_rows",
    "steer_rows",
    "lidar_frames",
)

VF_TO_NAVSIM_CAMERA_MAP: Dict[str, str] = {
    "CAM_P_F": "CAM_F0",    # Front panoramic -> Front camera
    "CAM_P_FL": "CAM_L0",   # Front-left panoramic -> Left0 camera
    "CAM_P_L": "CAM_L1",    # Left panoramic -> Left1 camera
    "CAM_P_LB": "CAM_L2",   # Left-back panoramic -> Left2 camera
    "CAM_P_FR": "CAM_R0",   # Front-right panoramic -> Right0 camera
    "CAM_P_R": "CAM_R1",    # Right panoramic -> Right1 camera
    "CAM_P_RB": "CAM_R2",   # Right-back panoramic -> Right2 camera
    "CAM_P_B": "CAM_B0",    # Back panoramic -> Back camera
}

REQUIRED_NAVSIM_CAMERAS: Tuple[str, ...] = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)
