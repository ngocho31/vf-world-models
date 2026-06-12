"""Write NuPlan-compatible GeoPackage map artifact for NAVSIM consumption.

Handles the SQLite / GeoPackage I/O layer:
  - DDL column schemas (verified against sg-one-north/9.17.1964/map.gpkg)
  - gpkg_spatial_ref_sys / gpkg_contents / gpkg_geometry_columns bookkeeping
  - ``meta`` table with coordinate-system identifiers for GPKGMapsDB
  - Companion JSON (vf-maps-v1.0.json) with raster-layer stubs
"""

from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

from ..contracts.gpkg_models import MapConversionSummary
from ..contracts.osm_models import ParsedLaneletMap

# ---------------------------------------------------------------------------
# NuPlan column schema DDL (verified against sg-one-north/9.17.1964/map.gpkg)
# ---------------------------------------------------------------------------

_DDL: Dict[str, str] = {
    "boundaries": (
        'CREATE TABLE "boundaries" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" LINESTRING,'
        '"boundary_segment_fids" TEXT(256),'
        '"has_reflectors" BOOLEAN,'
        '"boundary_type_fid" INTEGER,'
        '"creator_id" TEXT(256))'
    ),
    "baseline_paths": (
        'CREATE TABLE "baseline_paths" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" LINESTRING,'
        '"path_type_fid" INTEGER,'
        '"creator_id" TEXT(256),'
        '"lane_fid" INTEGER,'
        '"lane_connector_fid" INTEGER,'
        '"dubins_node_fids" TEXT(256))'
    ),
    "lanes_polygons": (
        'CREATE TABLE "lanes_polygons" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POLYGON,'
        '"lane_group_fid" INTEGER,'
        '"lane_fid" INTEGER,'
        '"lane_index" MEDIUMINT,'
        '"creator_id" TEXT(256),'
        '"name" TEXT(256),'
        '"road_type_fid" INTEGER,'
        '"lane_type_fid" INTEGER,'
        '"width" MEDIUMINT,'
        '"left_offset" REAL,'
        '"right_offset" REAL,'
        '"min_speed" REAL,'
        '"max_speed" REAL,'
        '"stops" MEDIUMINT,'
        '"left_boundary_fid" INTEGER,'
        '"right_boundary_fid" INTEGER,'
        '"speed_limit_mps" REAL,'
        '"left_has_reflectors" BOOLEAN,'
        '"right_has_reflectors" BOOLEAN,'
        '"from_edge_fid" INTEGER,'
        '"to_edge_fid" INTEGER)'
    ),
    "lane_connectors": (
        'CREATE TABLE "lane_connectors" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" LINESTRING,'
        '"exit_lane_group_fid" INTEGER,'
        '"exit_lane_fid" INTEGER,'
        '"entry_lane_group_fid" INTEGER,'
        '"lane_group_connector_fid" INTEGER,'
        '"entry_lane_fid" INTEGER,'
        '"intersection_fid" INTEGER,'
        '"turn_type_fid" INTEGER,'
        '"bulb_fids" TEXT(256),'
        '"traffic_light_stop_line_fids" TEXT(256),'
        '"overlap" MEDIUMINT,'
        '"creator_id" TEXT(256),'
        '"reflector_side_fid" INTEGER,'
        '"left_has_reflectors" BOOLEAN,'
        '"right_has_reflectors" BOOLEAN,'
        '"speed_limit_mps" REAL)'
    ),
    "road_segments": (
        'CREATE TABLE "road_segments" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POLYGON,'
        '"lane_group_fids" TEXT(256))'
    ),
    "lane_groups_polygons": (
        'CREATE TABLE "lane_groups_polygons" ('
        '"fid" INTEGER PRIMARY KEY NOT NULL,'
        '"geom" POLYGON,'
        '"creator_id" TEXT(256),'
        '"left_boundary_fid" INTEGER,'
        '"right_boundary_fid" INTEGER,'
        '"from_edge_fid" INTEGER,'
        '"to_edge_fid" INTEGER)'
    ),
    "lane_group_connectors": (
        'CREATE TABLE "lane_group_connectors" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POLYGON,'
        '"intersection_fid" INTEGER,'
        '"from_lane_group_fid" INTEGER,'
        '"to_lane_group_fid" INTEGER,'
        '"creator_id" TEXT(256),'
        '"from_edge_fid" INTEGER,'
        '"to_edge_fid" INTEGER,'
        '"left_boundary_fid" INTEGER,'
        '"right_boundary_fid" INTEGER)'
    ),
    "stop_polygons": (
        'CREATE TABLE "stop_polygons" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POLYGON,'
        '"stop_polygon_type_fid" INTEGER,'
        '"lane_group_fid" INTEGER,'
        '"lane_fids" TEXT(256),'
        '"lane_connector_fids" TEXT(256),'
        '"traffic_light_fids" TEXT(256),'
        '"crosswalk_fids" TEXT(256),'
        '"precedence_area_fids" TEXT(256),'
        '"creator_id" TEXT(256))'
    ),
    "intersections": (
        'CREATE TABLE "intersections" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POLYGON,'
        '"creator_id" TEXT(256),'
        '"intersection_type_fid" INTEGER,'
        '"is_mini" BOOLEAN)'
    ),
    "gen_lane_connectors_scaled_width_polygons": (
        'CREATE TABLE "gen_lane_connectors_scaled_width_polygons" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POLYGON,'
        '"baseline_path_fid" INTEGER,'
        '"lane_connector_fid" INTEGER,'
        '"from_edge_fid" INTEGER,'
        '"to_edge_fid" INTEGER,'
        '"left_boundary_fid" INTEGER,'
        '"right_boundary_fid" INTEGER)'
    ),
    "dubins_nodes": (
        'CREATE TABLE "dubins_nodes" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POINT,'
        '"radius" REAL,'
        '"heading" REAL,'
        '"creator_id" TEXT(256))'
    ),
    "crosswalks": (
        'CREATE TABLE "crosswalks" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POLYGON,'
        '"intersection_fids" TEXT(256),'
        '"lane_fids" TEXT(256),'
        '"creator_id" TEXT(256),'
        '"is_marked" BOOLEAN)'
    ),
    "carpark_areas": (
        'CREATE TABLE "carpark_areas" ('
        '"fid" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,'
        '"geom" POLYGON,'
        '"heading" REAL,'
        '"creator_id" TEXT(256))'
    ),
}

_GEOM_TYPE: Dict[str, str] = {
    "boundaries": "LINESTRING",
    "baseline_paths": "LINESTRING",
    "lanes_polygons": "POLYGON",
    "lane_connectors": "LINESTRING",
    "road_segments": "POLYGON",
    "dubins_nodes": "POINT",
    "crosswalks": "POLYGON",
    "carpark_areas": "POLYGON",
    "lane_groups_polygons": "POLYGON",
    "lane_group_connectors": "POLYGON",
    "stop_polygons": "POLYGON",
    "intersections": "POLYGON",
    "gen_lane_connectors_scaled_width_polygons": "POLYGON",
}


# ---------------------------------------------------------------------------
# GPKG SQLite helpers
# ---------------------------------------------------------------------------

def _initialize_gpkg(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA application_id=0x47504B47;")
    conn.execute("PRAGMA user_version=10300;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT UNIQUE,
            description TEXT DEFAULT '',
            last_change TEXT NOT NULL,
            min_x DOUBLE,
            min_y DOUBLE,
            max_x DOUBLE,
            max_y DOUBLE,
            srs_id INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            PRIMARY KEY (table_name, column_name)
        )
        """
    )
    _wgs84_def = (
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    conn.executemany(
        "INSERT OR REPLACE INTO gpkg_spatial_ref_sys"
        " (srs_name, srs_id, organization, organization_coordsys_id, definition, description)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("WGS 84 geodetic", 4326, "EPSG", 4326, _wgs84_def, "WGS 84"),
            ("Undefined Cartesian", 0, "NONE", 0, "undefined", "Undefined Cartesian coordinate reference system"),
        ],
    )
    # NuPlan requires a 'meta' table so GPKGMapsDB can reproject geometries into the local CRS.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            key   TEXT NOT NULL,
            value TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        [
            ("geographicCoordSystem", "epsg:4326"),
            ("projectedCoordSystem",  "epsg:32648"),  # UTM Zone 48N – Vietnam
        ],
    )


def _write_layer(
    conn: sqlite3.Connection,
    layer_name: str,
    rows: List[Dict[str, Any]],
    bbox: Optional[Tuple[float, float, float, float]],
    srs_id: int = 4326,
) -> None:
    conn.execute(_DDL[layer_name])

    for row in rows:
        cols = list(row.keys())
        vals = list(row.values())
        placeholders = ", ".join("?" * len(cols))
        col_list = ", ".join(f'"{c}"' for c in cols)
        conn.execute(
            f'INSERT INTO "{layer_name}" ({col_list}) VALUES ({placeholders})',
            vals,
        )

    min_x, min_y, max_x, max_y = bbox if bbox is not None else (0.0, 0.0, 0.0, 0.0)
    conn.execute(
        """
        INSERT OR REPLACE INTO gpkg_contents
        (table_name, data_type, identifier, description, last_change,
         min_x, min_y, max_x, max_y, srs_id)
        VALUES (?, 'features', ?, '', datetime('now'), ?, ?, ?, ?, ?)
        """,
        (layer_name, layer_name, min_x, min_y, max_x, max_y, srs_id),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO gpkg_geometry_columns
        (table_name, column_name, geometry_type_name, srs_id, z, m)
        VALUES (?, 'geom', ?, ?, 0, 0)
        """,
        (layer_name, _GEOM_TYPE[layer_name], srs_id),
    )


def _calc_bbox_from_nodes(
    parsed: ParsedLaneletMap,
) -> Optional[Tuple[float, float, float, float]]:
    if not parsed.nodes:
        return None
    lons = [n.lon for n in parsed.nodes.values()]
    lats = [n.lat for n in parsed.nodes.values()]
    return (min(lons), min(lats), max(lons), max(lats))


# ---------------------------------------------------------------------------
# Public writer class
# ---------------------------------------------------------------------------

class NuplanGpkgWriter:
    """Write NuPlan-compatible GPKG map file from pre-built layer row-dicts."""

    def write(
        self,
        layers: Dict[str, List[Dict[str, Any]]],
        parsed: ParsedLaneletMap,
        output_path: Path,
    ) -> MapConversionSummary:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        bbox = _calc_bbox_from_nodes(parsed)

        if output_path.exists():
            output_path.unlink()

        with sqlite3.connect(output_path) as conn:
            _initialize_gpkg(conn)
            for layer_name, rows in layers.items():
                _write_layer(conn, layer_name, rows, bbox)
            conn.commit()

        return MapConversionSummary(
            source_path=parsed.source_path,
            output_path=output_path,
            map_location="",
            map_version="",
            node_count=len(parsed.nodes),
            way_count=len(parsed.ways),
            relation_count=len(parsed.relations),
            boundary_count=len(layers.get("boundaries", [])),
            baseline_path_count=len(layers.get("baseline_paths", [])),
            lane_polygon_count=len(layers.get("lanes_polygons", [])),
            crosswalk_count=len(layers.get("crosswalks", [])),
            carpark_count=len(layers.get("carpark_areas", [])),
            lane_connector_count=len(layers.get("lane_connectors", [])),
            road_segment_count=len(layers.get("road_segments", [])),
            dubins_node_count=len(layers.get("dubins_nodes", [])),
            bbox=bbox,
        )


# ---------------------------------------------------------------------------
# Companion JSON helpers (nuplan-maps-v1.0.json)
# ---------------------------------------------------------------------------

def save_map_conversion_report(summary: MapConversionSummary, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    payload["source_path"] = str(summary.source_path)
    payload["output_path"] = str(summary.output_path)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_nuplan_maps_json(summary: MapConversionSummary, maps_root: Path) -> Path:
    """Create or update ``vf-maps-v1.0.json`` at *maps_root*.

    All other locations already present in the file are preserved.
    Returns the path of the (created or updated) JSON file.
    """
    maps_root = Path(maps_root)
    json_path = maps_root / "vf-maps-v1.0.json"

    if json_path.exists():
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {}

    layers: dict = {}
    if summary.bbox is not None:
        layers = _build_raster_layer_stubs(summary.bbox)

    data[summary.map_location] = {
        "version": summary.map_version,
        "layers": layers,
    }

    maps_root.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
        fh.write("\n")

    return json_path


def _build_raster_layer_stubs(
    bbox: Tuple[float, float, float, float],
    pixel_size_m: float = 1.0,
) -> dict:
    """Build nuplan-style raster-layer stubs from a WGS-84 bounding box."""
    min_lon, min_lat, max_lon, max_lat = bbox
    try:
        tf = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
        e_min, n_min = tf.transform(min_lon, min_lat)
        e_max, n_max = tf.transform(max_lon, max_lat)
    except Exception:
        return {}

    width_m = e_max - e_min
    height_m = n_max - n_min
    cols = max(1, int(width_m / pixel_size_m))
    rows = max(1, int(height_m / pixel_size_m))

    transform_matrix = [
        [pixel_size_m,  0.0,          0.0, e_min],
        [0.0,          -pixel_size_m, 0.0, n_max],
        [0.0,           0.0,          1.0, 0.0  ],
        [0.0,           0.0,          0.0, 1.0  ],
    ]

    layer_names = [
        "drivable_area", "drivable_area_distance_px",
        "semantic_prior", "semantic_prior_distance_px",
        "walkway", "walkway_distance_px",
    ]
    return {
        name: {"shape": [rows, cols], "transform_matrix": transform_matrix}
        for name in layer_names
    }
