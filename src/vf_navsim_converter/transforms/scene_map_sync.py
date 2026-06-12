"""Stage 4.5: resolve scene-map synchronization context."""

import json
import sqlite3
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from shapely import wkb
from shapely.geometry import Point
from shapely.strtree import STRtree

from ..contracts.models_navsim import SceneMapSyncContext
from ..contracts.models_raw import RawSensorBundle, RawTimeSeriesRow


class DefaultSceneMapSyncResolver:
    """Resolve map identity and route roadblock IDs from Stage 0 map report and NAV rows."""

    def __init__(
        self,
        *,
        output_root: Path,
        default_map_location: str,
        map_sync_min_overlap: float,
    ) -> None:
        self._output_root = output_root
        self._default_map_location = default_map_location
        self._map_sync_min_overlap = map_sync_min_overlap

    def resolve(self, raw_bundle: RawSensorBundle) -> SceneMapSyncContext:
        default_context = SceneMapSyncContext(map_location=self._default_map_location)

        report_path = self._output_root / ".conversion_metadata" / "stage0_map_conversion_report.json"
        if not report_path.exists():
            return default_context

        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default_context

        bbox = _parse_bbox(payload.get("bbox"))
        overlap_ratio = _compute_nav_bbox_overlap(raw_bundle.nav_rows, bbox)

        configured_map_location = payload.get("map_location") or self._default_map_location
        if overlap_ratio <= 0.0:
            return SceneMapSyncContext(
                map_location=configured_map_location,
                roadblock_ids=[],
                overlap_ratio=overlap_ratio,
            )

        roadblock_ids = _extract_route_roadblock_ids_from_nav(
            payload.get("output_path"),
            raw_bundle.nav_rows,
        )
        if overlap_ratio < self._map_sync_min_overlap:
            # Keep synchronized map identity but avoid forcing a weak route assignment.
            roadblock_ids = []

        return SceneMapSyncContext(
            map_location=configured_map_location,
            roadblock_ids=roadblock_ids,
            overlap_ratio=overlap_ratio,
        )


def _parse_bbox(raw_bbox: object) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        min_lon, min_lat, max_lon, max_lat = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None
    return (min_lon, min_lat, max_lon, max_lat)


def _pick_float(row: RawTimeSeriesRow, keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = row.values.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _compute_nav_bbox_overlap(
    nav_rows: Sequence[RawTimeSeriesRow],
    bbox: Optional[Tuple[float, float, float, float]],
) -> float:
    if not nav_rows or bbox is None:
        return 0.0

    min_lon, min_lat, max_lon, max_lat = bbox
    considered = 0
    inside = 0

    for row in nav_rows:
        lat = _pick_float(row, ("Latitude", "latitude", "lat"))
        lon = _pick_float(row, ("Longitude", "longitude", "lon"))
        if lat is None or lon is None:
            continue

        considered += 1
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            inside += 1

    if considered == 0:
        return 0.0
    return inside / considered


def _extract_route_roadblock_ids_from_nav(
    map_gpkg_path: object,
    nav_rows: Sequence[RawTimeSeriesRow],
    max_nav_samples: int = 400,
    max_bridge_hops: int = 40,
) -> List[str]:
    if not isinstance(map_gpkg_path, str) or not map_gpkg_path:
        return []

    candidate = Path(map_gpkg_path)
    if not candidate.exists() or not nav_rows:
        return []

    nav_points = _sample_nav_points(nav_rows, max_nav_samples)
    if not nav_points:
        return []

    try:
        with sqlite3.connect(candidate) as conn:
            road_segments = conn.execute('SELECT fid, geom FROM road_segments ORDER BY fid').fetchall()
            lane_group_rows = conn.execute('SELECT fid, lane_group_fids FROM road_segments ORDER BY fid').fetchall()
            connector_rows = conn.execute(
                'SELECT entry_lane_group_fid, exit_lane_group_fid FROM lane_connectors '
                'WHERE entry_lane_group_fid IS NOT NULL AND exit_lane_group_fid IS NOT NULL'
            ).fetchall()
    except sqlite3.Error:
        return []

    segment_ids: List[str] = []
    segment_geoms = []
    for fid, geom_blob in road_segments:
        geom = _decode_gpkg_geometry(geom_blob)
        if fid is None or geom is None or geom.is_empty:
            continue
        segment_ids.append(str(fid))
        segment_geoms.append(geom)

    if not segment_ids:
        return []

    tree = STRtree(segment_geoms)
    geom_index_to_segment = {idx: segment_ids[idx] for idx in range(len(segment_ids))}

    matched_segment_ids: List[str] = []
    for lon, lat in nav_points:
        point = Point(lon, lat)
        nearest_idx = tree.nearest(point)
        if nearest_idx is None:
            continue
        matched_id = geom_index_to_segment.get(int(nearest_idx))
        if matched_id is None:
            continue
        if not matched_segment_ids or matched_segment_ids[-1] != matched_id:
            matched_segment_ids.append(matched_id)

    if not matched_segment_ids:
        return []

    lane_group_to_segment = _build_lane_group_to_segment_index(lane_group_rows)
    adjacency = _build_segment_adjacency(connector_rows, lane_group_to_segment)
    stitched = _stitch_segment_route(matched_segment_ids, adjacency, max_bridge_hops=max_bridge_hops)
    return _dedupe_preserve_order(stitched)


def _sample_nav_points(nav_rows: Sequence[RawTimeSeriesRow], max_samples: int) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for row in nav_rows:
        lat = _pick_float(row, ("Latitude", "latitude", "lat"))
        lon = _pick_float(row, ("Longitude", "longitude", "lon"))
        if lat is None or lon is None:
            continue
        points.append((lon, lat))

    if len(points) <= max_samples:
        return points

    stride = max(1, len(points) // max_samples)
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _decode_gpkg_geometry(blob: object):
    if not isinstance(blob, (bytes, bytearray)):
        return None
    if len(blob) < 9:
        return None
    if blob[0:2] != b"GP":
        return None

    flags = blob[3]
    envelope_indicator = (flags >> 1) & 0b111
    envelope_bytes_lookup = {
        0: 0,
        1: 32,
        2: 48,
        3: 48,
        4: 64,
    }
    envelope_bytes = envelope_bytes_lookup.get(envelope_indicator)
    if envelope_bytes is None:
        return None

    wkb_offset = 8 + envelope_bytes
    if wkb_offset >= len(blob):
        return None

    try:
        return wkb.loads(bytes(blob[wkb_offset:]))
    except Exception:
        return None


def _parse_lane_group_ids(value: object) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            ids = []
            for item in parsed:
                try:
                    ids.append(int(item))
                except (TypeError, ValueError):
                    continue
            return ids
    except json.JSONDecodeError:
        pass

    tokens = text.replace("[", "").replace("]", "").split(",")
    ids = []
    for token in tokens:
        token = token.strip().strip("'").strip('"')
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError:
            continue
    return ids


def _build_lane_group_to_segment_index(rows: Sequence[Tuple[object, object]]) -> dict[int, str]:
    index: dict[int, str] = {}
    for fid, lane_group_fids in rows:
        if fid is None:
            continue
        segment_id = str(fid)
        for lane_group_id in _parse_lane_group_ids(lane_group_fids):
            index[lane_group_id] = segment_id
    return index


def _build_segment_adjacency(
    connector_rows: Sequence[Tuple[object, object]],
    lane_group_to_segment: dict[int, str],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for entry_lane_group, exit_lane_group in connector_rows:
        try:
            entry_group = int(entry_lane_group)
            exit_group = int(exit_lane_group)
        except (TypeError, ValueError):
            continue

        src = lane_group_to_segment.get(entry_group)
        dst = lane_group_to_segment.get(exit_group)
        if src is None or dst is None:
            continue
        if src == dst:
            continue

        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
    return adjacency


def _find_path_bfs(adjacency: dict[str, set[str]], start: str, goal: str, max_hops: int) -> List[str]:
    if start == goal:
        return [start]

    frontier: deque[Tuple[str, List[str]]] = deque([(start, [start])])
    visited = {start}

    while frontier:
        node, path = frontier.popleft()
        if len(path) > max_hops:
            continue

        for neighbor in adjacency.get(node, set()):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            frontier.append((neighbor, next_path))

    return []


def _stitch_segment_route(observed: Sequence[str], adjacency: dict[str, set[str]], max_bridge_hops: int) -> List[str]:
    if not observed:
        return []

    stitched = [observed[0]]
    for target in observed[1:]:
        current = stitched[-1]
        if current == target:
            continue

        bridge = _find_path_bfs(adjacency, current, target, max_hops=max_bridge_hops)
        if bridge:
            stitched.extend(bridge[1:])
        else:
            stitched.append(target)
    return stitched


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def save_scene_map_sync_context(context: SceneMapSyncContext, output_path: Path) -> None:
    """Save the scene map sync context to a JSON file."""
    manifest_data = {
        "map_location": context.map_location,
        "roadblock_ids": context.roadblock_ids,
        "overlap_ratio": context.overlap_ratio,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)