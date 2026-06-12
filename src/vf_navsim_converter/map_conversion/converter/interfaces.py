"""Map converter interface contract."""

from typing import Any, Dict, List, Protocol

from ..contracts.osm_models import ParsedLaneletMap


class MapConverter(Protocol):
    def convert(self, parsed: ParsedLaneletMap) -> Dict[str, List[Dict[str, Any]]]:
        """Convert a parsed OSM map into NuPlan-compatible GPKG layer row-dicts."""
        ...
