"""OSM / Lanelet2 input data models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class OSMNode:
    node_id: int
    lat: float
    lon: float
    ele: float = 0.0


@dataclass(frozen=True)
class OSMWay:
    way_id: int
    node_refs: List[int]
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OSMRelationMember:
    member_type: str
    ref: int
    role: str


@dataclass(frozen=True)
class OSMRelation:
    relation_id: int
    members: List[OSMRelationMember] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedLaneletMap:
    source_path: Path
    nodes: Dict[int, OSMNode] = field(default_factory=dict)
    ways: Dict[int, OSMWay] = field(default_factory=dict)
    relations: Dict[int, OSMRelation] = field(default_factory=dict)
