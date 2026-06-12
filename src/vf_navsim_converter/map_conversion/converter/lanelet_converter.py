"""Convert a parsed Lanelet2 map into NuPlan-compatible GPKG layer row-dicts.

Produces typed column schemas matching the reference NuPlan gpkg format
(derived from sg-one-north/9.17.1964/map.gpkg), including derived layers:
  - lane_connectors  (from shared endpoint node topology)
  - road_segments    (connected chains of lanelets)
  - dubins_nodes     (junction endpoints of baseline_paths)
  - lane_groups_polygons / lane_group_connectors / intersections
    (routing layers required by NuPlan ROADBLOCK API)
"""

from __future__ import annotations

import json
import math
import struct
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.wkb import dumps as wkb_dumps, loads as wkb_loads

from ..contracts.osm_models import OSMRelation, OSMWay, ParsedLaneletMap

# ---------------------------------------------------------------------------
# Lanelet2 tag → NuPlan fid mappings
# ---------------------------------------------------------------------------

# turn_direction tag -> NuPlan turn_type_fid (1=straight, 2=left, 3=right, 4=uturn)
_TURN_FID: Dict[str, int] = {"straight": 1, "left": 2, "right": 3, "u_turn": 4, "merge": 1}

# subtype -> NuPlan lane_type_fid
_LANE_TYPE_FID: Dict[str, int] = {
    "road": 1, "highway": 1, "emergency": 4,
    "bus": 6, "bicycle": 3, "parking": 2, "crosswalk": 7,
}


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _member_way_id(rel: OSMRelation, role: str) -> Optional[int]:
    for m in rel.members:
        if m.role == role and m.member_type == "way":
            return m.ref
    return None


def _build_topology(
    lanelets: List[OSMRelation],
    parsed: ParsedLaneletMap,
) -> Dict[str, Any]:
    """Derive successor/predecessor maps from shared right-boundary endpoint nodes."""
    lanelet_start_node: Dict[int, int] = {}
    lanelet_end_node: Dict[int, int] = {}

    for rel in lanelets:
        right_id = _member_way_id(rel, "right")
        if right_id is not None and right_id in parsed.ways:
            node_refs = parsed.ways[right_id].node_refs
            if len(node_refs) >= 2:
                lanelet_start_node[rel.relation_id] = node_refs[0]
                lanelet_end_node[rel.relation_id] = node_refs[-1]

    start_index: Dict[int, List[int]] = defaultdict(list)
    for rel_id, node_id in lanelet_start_node.items():
        start_index[node_id].append(rel_id)

    successors: Dict[int, List[int]] = {}
    predecessors: Dict[int, List[int]] = defaultdict(list)
    for rel_id, end_node in lanelet_end_node.items():
        succs = [s for s in start_index.get(end_node, []) if s != rel_id]
        if succs:
            successors[rel_id] = succs
            for s in succs:
                predecessors[s].append(rel_id)

    return {
        "lanelet_start_node": lanelet_start_node,
        "lanelet_end_node": lanelet_end_node,
        "successors": successors,
        "predecessors": dict(predecessors),
    }


def _assign_road_segments(
    lanelets: List[OSMRelation],
    topology: Dict[str, Any],
    lanelet_fid: Dict[int, int],
) -> Dict[int, List[int]]:
    """Group lanelets into road segments (connected chains without branching).

    Returns {road_segment_fid -> [lanelet_fid, ...]}
    """
    successors = topology["successors"]
    predecessors = topology["predecessors"]
    rel_ids = {rel.relation_id for rel in lanelets}

    chain_starts: set = set()
    for rel in lanelets:
        rid = rel.relation_id
        preds = predecessors.get(rid, [])
        if not preds:
            chain_starts.add(rid)
        elif len(preds) > 1:
            chain_starts.add(rid)
        elif len(successors.get(preds[0], [])) > 1:
            chain_starts.add(rid)

    visited: set = set()
    road_segs: Dict[int, List[int]] = {}
    rs_fid = 1

    for start_id in sorted(chain_starts):
        if start_id in visited:
            continue
        chain: List[int] = []
        cur: Optional[int] = start_id
        while cur is not None and cur not in visited and cur in rel_ids:
            visited.add(cur)
            chain.append(lanelet_fid[cur])
            succs = successors.get(cur, [])
            cur = succs[0] if len(succs) == 1 and succs[0] not in chain_starts else None
        if chain:
            road_segs[rs_fid] = chain
            rs_fid += 1

    for rel in lanelets:
        if rel.relation_id not in visited:
            road_segs[rs_fid] = [lanelet_fid[rel.relation_id]]
            rs_fid += 1

    return road_segs


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _way_coords(way: OSMWay, parsed: ParsedLaneletMap) -> List[Tuple[float, float]]:
    return [
        (parsed.nodes[nid].lon, parsed.nodes[nid].lat)
        for nid in way.node_refs if nid in parsed.nodes
    ]


def _way_linestring(way: OSMWay, parsed: ParsedLaneletMap) -> Optional[LineString]:
    coords = _way_coords(way, parsed)
    return LineString(coords) if len(coords) >= 2 else None


def _resample_coords(coords: List[Tuple], n: int) -> List[Tuple]:
    if len(coords) <= 1 or n <= 1:
        return coords
    line = LineString(coords)
    if line.length == 0:
        return coords
    return [line.interpolate(i / (n - 1) * line.length).coords[0] for i in range(n)]


def _centerline(rel: OSMRelation, parsed: ParsedLaneletMap) -> Optional[LineString]:
    center_id = _member_way_id(rel, "centerline")
    if center_id is not None and center_id in parsed.ways:
        return _way_linestring(parsed.ways[center_id], parsed)

    left_id = _member_way_id(rel, "left")
    right_id = _member_way_id(rel, "right")
    if left_id is None or right_id is None:
        return None
    if left_id not in parsed.ways or right_id not in parsed.ways:
        return None

    lc = _way_coords(parsed.ways[left_id], parsed)
    rc = _way_coords(parsed.ways[right_id], parsed)
    if len(lc) < 2 or len(rc) < 2:
        return None

    n = max(len(lc), len(rc), 4)
    lc = _resample_coords(lc, n)
    rc = _resample_coords(rc, n)
    mid = [((lx + rx) / 2.0, (ly + ry) / 2.0) for (lx, ly), (rx, ry) in zip(lc, rc)]
    return LineString(mid) if len(mid) >= 2 else None


def _lane_polygon(rel: OSMRelation, parsed: ParsedLaneletMap) -> Optional[Polygon]:
    left_id = _member_way_id(rel, "left")
    right_id = _member_way_id(rel, "right")
    if left_id is None or right_id is None:
        return None
    if left_id not in parsed.ways or right_id not in parsed.ways:
        return None

    lc = _way_coords(parsed.ways[left_id], parsed)
    rc = _way_coords(parsed.ways[right_id], parsed)
    if len(lc) < 2 or len(rc) < 2:
        return None

    ring = lc + list(reversed(rc))
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def _speed_limit_mps(tags: Dict[str, str]) -> Optional[float]:
    sl = tags.get("speed_limit")
    if sl is None:
        return None
    try:
        return float(sl) / 3.6
    except ValueError:
        return None


def _heading_deg(p1: Tuple, p2: Tuple) -> float:
    """Bearing in degrees [0, 360) from p1=(lon1,lat1) to p2=(lon2,lat2)."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 360.0


def _geom_blob(geom: BaseGeometry, srs_id: int = 4326) -> bytes:
    wkb = wkb_dumps(geom, hex=False)
    header = b"GP" + bytes([0, 0]) + struct.pack("<i", srs_id)
    return header + wkb


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def _build_boundaries(
    parsed: ParsedLaneletMap,
    way_fid: Dict[int, int],
) -> List[Dict[str, Any]]:
    rows = []
    for way_id in sorted(parsed.ways):
        way = parsed.ways[way_id]
        geom = _way_linestring(way, parsed)
        if geom is None or geom.is_empty:
            continue
        rows.append({
            "fid": way_fid[way_id],
            "geom": _geom_blob(geom),
            "boundary_segment_fids": None,
            "has_reflectors": False,
            "boundary_type_fid": 1,
            "creator_id": "vf_converter",
        })
    return rows


def _build_lanes_and_baselines(
    lanelets: List[OSMRelation],
    parsed: ParsedLaneletMap,
    lanelet_fid: Dict[int, int],
    way_fid: Dict[int, int],
    rs_fid_map: Dict[int, int],
) -> Tuple[List[Dict], List[Dict]]:
    """Build lanes_polygons and lane-level baseline_paths rows."""
    lane_rows: List[Dict] = []
    bp_rows: List[Dict] = []
    bp_fid = 1

    for rel in sorted(lanelets, key=lambda r: r.relation_id):
        lfid = lanelet_fid[rel.relation_id]
        left_id = _member_way_id(rel, "left")
        right_id = _member_way_id(rel, "right")

        poly = _lane_polygon(rel, parsed)
        if poly is None:
            continue

        lane_rows.append({
            "fid": lfid,
            "geom": _geom_blob(poly),
            "lane_group_fid": rs_fid_map.get(lfid),
            "lane_fid": lfid,
            "lane_index": 0,
            "creator_id": "vf_converter",
            "name": f"lane_{rel.relation_id}",
            "road_type_fid": 1,
            "lane_type_fid": _LANE_TYPE_FID.get(rel.tags.get("subtype", "road"), 1),
            "width": None,
            "left_offset": None,
            "right_offset": None,
            "min_speed": None,
            "max_speed": _speed_limit_mps(rel.tags),
            "stops": 0,
            "left_boundary_fid": way_fid.get(left_id) if left_id else None,
            "right_boundary_fid": way_fid.get(right_id) if right_id else None,
            "speed_limit_mps": _speed_limit_mps(rel.tags),
            "left_has_reflectors": False,
            "right_has_reflectors": False,
            "from_edge_fid": None,
            "to_edge_fid": None,
        })

        cl = _centerline(rel, parsed)
        if cl is not None and not cl.is_empty and cl.length > 0:
            bp_rows.append({
                "fid": bp_fid,
                "geom": _geom_blob(cl),
                "path_type_fid": 1,
                "creator_id": "vf_converter",
                "lane_fid": lfid,
                "lane_connector_fid": None,
                "dubins_node_fids": None,
            })
            bp_fid += 1

    return lane_rows, bp_rows


def _build_connectors_dubins_and_connector_paths(
    lanelets: List[OSMRelation],
    parsed: ParsedLaneletMap,
    lanelet_fid: Dict[int, int],
    rs_fid_map: Dict[int, int],
    topology: Dict[str, Any],
    first_bp_fid: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Build lane_connectors, dubins_nodes, and connector baseline_paths."""
    successors = topology["successors"]
    rel_by_id: Dict[int, OSMRelation] = {rel.relation_id: rel for rel in lanelets}

    centerlines: Dict[int, LineString] = {}
    for rel in lanelets:
        cl = _centerline(rel, parsed)
        if cl is not None and not cl.is_empty:
            centerlines[rel.relation_id] = cl

    conn_rows: List[Dict] = []
    dubins_rows: List[Dict] = []
    bp_rows: List[Dict] = []

    conn_fid = 1
    dubins_fid = 1
    bp_fid = first_bp_fid
    dubins_point_fid: Dict[Tuple[float, float], int] = {}

    def _get_or_create_dubins(pt: Tuple, heading: float) -> int:
        nonlocal dubins_fid
        key = (round(pt[0], 8), round(pt[1], 8))
        if key not in dubins_point_fid:
            dubins_point_fid[key] = dubins_fid
            dubins_rows.append({
                "fid": dubins_fid,
                "geom": _geom_blob(Point(pt)),
                "radius": 0.0,
                "heading": heading,
                "creator_id": "vf_converter",
            })
            dubins_fid += 1
        return dubins_point_fid[key]

    for src_rel_id in sorted(successors.keys()):
        if src_rel_id not in lanelet_fid:
            continue
        src_cl = centerlines.get(src_rel_id)
        if src_cl is None:
            continue

        src_fid = lanelet_fid[src_rel_id]
        src_end = src_cl.coords[-1]
        src_dubins = _get_or_create_dubins(
            src_end, _heading_deg(src_cl.coords[-2], src_cl.coords[-1])
        )

        src_rel = rel_by_id.get(src_rel_id)
        turn_fid = _TURN_FID.get(
            src_rel.tags.get("turn_direction", "straight") if src_rel else "straight", 1
        )

        for dst_rel_id in successors[src_rel_id]:
            if dst_rel_id not in lanelet_fid:
                continue
            dst_cl = centerlines.get(dst_rel_id)
            if dst_cl is None:
                continue

            dst_fid = lanelet_fid[dst_rel_id]
            dst_start = dst_cl.coords[0]
            dst_dubins = _get_or_create_dubins(
                dst_start,
                _heading_deg(dst_cl.coords[0], dst_cl.coords[1]) if len(dst_cl.coords) >= 2 else 0.0,
            )

            conn_geom = LineString([src_end, dst_start])
            if conn_geom.length == 0:
                # degenerate: lanelets share an endpoint — skip, NuPlan requires length > 0
                continue

            conn_rows.append({
                "fid": conn_fid,
                "geom": _geom_blob(conn_geom),
                "exit_lane_group_fid": rs_fid_map.get(src_fid),
                "exit_lane_fid": src_fid,
                "entry_lane_group_fid": rs_fid_map.get(dst_fid),
                "lane_group_connector_fid": None,
                "entry_lane_fid": dst_fid,
                "intersection_fid": None,
                "turn_type_fid": turn_fid,
                "bulb_fids": None,
                "traffic_light_stop_line_fids": None,
                "overlap": 0,
                "creator_id": "vf_converter",
                "reflector_side_fid": None,
                "left_has_reflectors": False,
                "right_has_reflectors": False,
                "speed_limit_mps": None,
            })

            bp_rows.append({
                "fid": bp_fid,
                "geom": _geom_blob(conn_geom),
                "path_type_fid": 2,
                "creator_id": "vf_converter",
                "lane_fid": None,
                "lane_connector_fid": conn_fid,
                "dubins_node_fids": json.dumps([src_dubins, dst_dubins]),
            })
            bp_fid += 1
            conn_fid += 1

    return conn_rows, dubins_rows, bp_rows


def _build_road_segments(
    rs_groups: Dict[int, List[int]],
    lane_poly_map: Dict[int, Polygon],
) -> List[Dict]:
    rows: List[Dict] = []
    for rs_fid, lane_fids in sorted(rs_groups.items()):
        polys = [lane_poly_map[lfid] for lfid in lane_fids if lfid in lane_poly_map]
        if not polys:
            continue
        union = polys[0] if len(polys) == 1 else unary_union(polys)
        if not isinstance(union, Polygon):
            union = union.convex_hull
        if union.is_empty:
            continue
        rows.append({
            "fid": rs_fid,
            "geom": _geom_blob(union),
            "lane_group_fids": json.dumps(lane_fids),
        })
    return rows


def _build_crosswalks(
    crosswalk_rels: List[OSMRelation],
    parsed: ParsedLaneletMap,
) -> List[Dict]:
    rows: List[Dict] = []
    for fid, rel in enumerate(crosswalk_rels, start=1):
        poly = _lane_polygon(rel, parsed)
        if poly is None or poly.is_empty:
            continue
        rows.append({
            "fid": fid,
            "geom": _geom_blob(poly),
            "intersection_fids": None,
            "lane_fids": None,
            "creator_id": "vf_converter",
            "is_marked": True,
        })
    return rows


def _build_carpark_areas(
    parking_rels: List[OSMRelation],
    parking_ways: List[OSMWay],
    parsed: ParsedLaneletMap,
) -> List[Dict]:
    rows: List[Dict] = []
    fid = 1
    for rel in parking_rels:
        poly = _lane_polygon(rel, parsed)
        if poly is not None and not poly.is_empty:
            rows.append({"fid": fid, "geom": _geom_blob(poly), "heading": None, "creator_id": "vf_converter"})
            fid += 1
    for way in parking_ways:
        coords = _way_coords(way, parsed)
        if len(coords) >= 3:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                rows.append({"fid": fid, "geom": _geom_blob(poly), "heading": None, "creator_id": "vf_converter"})
                fid += 1
    return rows


# ---------------------------------------------------------------------------
# Derived routing layers
# ---------------------------------------------------------------------------

def _derive_intersections(lane_connector_rows: List[Dict]) -> List[Dict]:
    """Derive intersection polygons by clustering lane-connector centerlines.

    Algorithm:
    1. Decode each lane-connector LINESTRING (WGS-84) from its WKB blob.
    2. Re-project to UTM Zone 48N (EPSG:32648) for metric buffering.
    3. Buffer each line by *lane_buffer_m* to get an approximate lane ribbon.
    4. Union all ribbons → connected components.
    5. Retain only components involving ≥ *min_road_segments* distinct RS IDs.
    6. Take the convex hull of each retained component, re-project to WGS-84.
    """
    if not lane_connector_rows:
        return []

    lane_buffer_m = 3.5
    min_road_segments = 3

    _wgs_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    _utm_to_wgs = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)

    def _to_utm(geom: BaseGeometry) -> BaseGeometry:
        return transform(_wgs_to_utm.transform, geom)

    def _to_wgs(geom: BaseGeometry) -> BaseGeometry:
        return transform(_utm_to_wgs.transform, geom)

    geoms_utm: List[BaseGeometry] = []
    rs_sets: List[frozenset] = []
    for lc in lane_connector_rows:
        raw = lc.get("geom")
        if not raw:
            continue
        try:
            wkb_data = raw[8:] if isinstance(raw, (bytes, bytearray)) and raw[:2] == b"GP" else raw
            line_wgs = wkb_loads(wkb_data)
        except Exception:
            continue
        if line_wgs.is_empty:
            continue
        geoms_utm.append(_to_utm(line_wgs))
        entry = lc.get("entry_lane_group_fid")
        exit_ = lc.get("exit_lane_group_fid")
        rs_sets.append(frozenset(x for x in (entry, exit_) if x is not None))

    if not geoms_utm:
        return []

    buffered = [g.buffer(lane_buffer_m) for g in geoms_utm]
    merged = unary_union(buffered)
    components = list(merged.geoms) if hasattr(merged, "geoms") else [merged]

    intersection_rows: List[Dict] = []
    fid = 1
    for component in components:
        if component.is_empty or not component.is_valid:
            continue
        member_rs: set = set()
        for idx, buf in enumerate(buffered):
            if component.intersects(buf):
                member_rs.update(rs_sets[idx])
        if len(member_rs) < min_road_segments:
            continue
        hull_wgs = _to_wgs(component.convex_hull)
        if hull_wgs.is_empty or not isinstance(hull_wgs, Polygon):
            continue
        intersection_rows.append({
            "fid": fid,
            "geom": _geom_blob(hull_wgs),
            "intersection_type_fid": None,
            "creator_id": "vf_converter",
        })
        fid += 1

    return intersection_rows


def _derive_routing_layers(
    road_segment_rows: List[Dict],
    lane_connector_rows: List[Dict],
) -> Dict[str, List[Dict]]:
    """Derive NuPlan ROADBLOCK / ROADBLOCK_CONNECTOR layers from base layer data."""
    lgp_rows: List[Dict] = [
        {
            "fid": rs["fid"],
            "geom": rs["geom"],
            "creator_id": None,
            "left_boundary_fid": None,
            "right_boundary_fid": None,
            "from_edge_fid": None,
            "to_edge_fid": None,
        }
        for rs in road_segment_rows
    ]

    seen: set = set()
    lgc_rows: List[Dict] = []
    fid = 1
    for lc in lane_connector_rows:
        from_lg = lc.get("exit_lane_group_fid")
        to_lg = lc.get("entry_lane_group_fid")
        if from_lg is None or to_lg is None:
            continue
        key = (from_lg, to_lg)
        if key in seen:
            continue
        seen.add(key)
        lgc_rows.append({
            "fid": fid,
            "geom": lc["geom"],
            "intersection_fid": lc.get("intersection_fid"),
            "from_lane_group_fid": from_lg,
            "to_lane_group_fid": to_lg,
            "creator_id": None,
            "from_edge_fid": None,
            "to_edge_fid": None,
            "left_boundary_fid": None,
            "right_boundary_fid": None,
        })
        fid += 1

    return {
        "lane_groups_polygons": lgp_rows,
        "lane_group_connectors": lgc_rows,
        "stop_polygons": [],
        "intersections": _derive_intersections(lane_connector_rows),
        "gen_lane_connectors_scaled_width_polygons": [],
    }


# ---------------------------------------------------------------------------
# Public converter class
# ---------------------------------------------------------------------------

class LaneletConverter:
    """Convert a :class:`ParsedLaneletMap` into NuPlan-compatible GPKG layer rows."""

    def convert(self, parsed: ParsedLaneletMap) -> Dict[str, List[Dict[str, Any]]]:
        lanelets = [
            rel for rel in parsed.relations.values()
            if rel.tags.get("type") == "lanelet" and rel.tags.get("subtype") != "crosswalk"
        ]
        crosswalk_rels = [
            rel for rel in parsed.relations.values()
            if rel.tags.get("type") == "lanelet" and rel.tags.get("subtype") == "crosswalk"
        ]
        parking_rels = [
            rel for rel in parsed.relations.values()
            if rel.tags.get("type") == "lanelet" and rel.tags.get("subtype") == "parking_lot"
        ]
        parking_ways = [
            way for way in parsed.ways.values()
            if way.tags.get("type") == "parking_lot" or way.tags.get("area") == "yes"
        ]

        way_fid: Dict[int, int] = {
            way_id: i for i, way_id in enumerate(sorted(parsed.ways), start=1)
        }
        lanelet_fid: Dict[int, int] = {
            rel.relation_id: i
            for i, rel in enumerate(sorted(lanelets, key=lambda r: r.relation_id), start=1)
        }

        topology = _build_topology(lanelets, parsed)
        rs_groups = _assign_road_segments(lanelets, topology, lanelet_fid)
        rs_fid_map: Dict[int, int] = {
            lfid: rs_fid for rs_fid, lane_fids in rs_groups.items() for lfid in lane_fids
        }

        boundaries = _build_boundaries(parsed, way_fid)
        lane_rows, bp_lane_rows = _build_lanes_and_baselines(
            lanelets, parsed, lanelet_fid, way_fid, rs_fid_map
        )

        lane_poly_map: Dict[int, Polygon] = {
            lanelet_fid[rel.relation_id]: poly
            for rel in lanelets
            if (poly := _lane_polygon(rel, parsed)) is not None
        }

        conn_rows, dubins_rows, bp_conn_rows = _build_connectors_dubins_and_connector_paths(
            lanelets, parsed, lanelet_fid, rs_fid_map, topology,
            first_bp_fid=len(bp_lane_rows) + 1,
        )

        road_segments = _build_road_segments(rs_groups, lane_poly_map)
        crosswalks = _build_crosswalks(crosswalk_rels, parsed)
        carpark_areas = _build_carpark_areas(parking_rels, parking_ways, parsed)
        routing_layers = _derive_routing_layers(road_segments, conn_rows)

        return {
            "boundaries": boundaries,
            "baseline_paths": bp_lane_rows + bp_conn_rows,
            "lanes_polygons": lane_rows,
            "lane_connectors": conn_rows,
            "road_segments": road_segments,
            "dubins_nodes": dubins_rows,
            "crosswalks": crosswalks,
            "carpark_areas": carpark_areas,
            **routing_layers,
        }
