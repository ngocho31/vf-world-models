"""GPKG output data models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class MapFeature:
    feature_id: str
    geometry: BaseGeometry
    properties: Dict[str, object] = field(default_factory=dict)


@dataclass
class MapLayerData:
    name: str
    geometry_type: str
    features: List[MapFeature] = field(default_factory=list)


@dataclass
class MapConversionSummary:
    source_path: Path
    output_path: Path
    map_location: str
    map_version: str
    node_count: int = 0
    way_count: int = 0
    relation_count: int = 0
    boundary_count: int = 0
    baseline_path_count: int = 0
    lane_polygon_count: int = 0
    crosswalk_count: int = 0
    carpark_count: int = 0
    lane_connector_count: int = 0
    road_segment_count: int = 0
    dubins_node_count: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
