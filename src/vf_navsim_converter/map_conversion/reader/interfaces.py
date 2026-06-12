"""OSM reader interface contract."""

from pathlib import Path
from typing import Protocol

from ..contracts.osm_models import ParsedLaneletMap


class OSMReader(Protocol):
    def read(self, source_path: Path) -> ParsedLaneletMap:
        """Parse an OSM / Lanelet2 file and return the structured map data."""
        ...
