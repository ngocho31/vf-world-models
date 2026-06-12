"""Orchestrate the HD map conversion workflow."""

from datetime import datetime
from pathlib import Path

from ..config import HdMapConversionConfig
from .reader import Lanelet2OSMReader
from .converter import LaneletConverter
from .writer import NuplanGpkgWriter, save_map_conversion_report, write_nuplan_maps_json


class HdMapConversionRunner:
    """Convert a lanelet2 OSM map into a NAVSIM-style GeoPackage."""

    def __init__(self, config: HdMapConversionConfig) -> None:
        self._config = config

    def run(self):
        source_path = self._resolve_source_path(self._config.input_root, self._config.source_map_filename)

        reader = Lanelet2OSMReader()
        parsed = reader.read(source_path)

        converter = LaneletConverter()
        layers = converter.convert(parsed)

        map_version = self._config.map_version or datetime.now().strftime("%Y-%m-%d")

        output_path = (
            self._config.output_root
            / "maps"
            / self._config.map_location
            / map_version
            / self._config.output_filename
        )

        writer = NuplanGpkgWriter()
        summary = writer.write(layers, parsed, output_path)
        summary.map_location = self._config.map_location
        summary.map_version = map_version

        metadata_dir = self._config.output_root / ".conversion_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        save_map_conversion_report(summary, metadata_dir / "stage0_map_conversion_report.json")

        maps_root = self._config.output_root / "maps"
        write_nuplan_maps_json(summary, maps_root)

        return summary

    @staticmethod
    def _resolve_source_path(source_root: Path, source_filename: str) -> Path:
        source_root = Path(source_root)
        if source_root.is_file():
            return source_root
        if source_root.is_dir():
            candidate = source_root / source_filename
            if candidate.exists():
                return candidate
            osm_files = sorted(source_root.glob("*.osm"))
            if osm_files:
                return osm_files[0]
        raise FileNotFoundError(f"Unable to resolve lanelet2 OSM source from: {source_root}")
