"""GPKG writer interface contract."""

from pathlib import Path
from typing import Any, Dict, List, Protocol

from ..contracts.gpkg_models import MapConversionSummary
from ..contracts.osm_models import ParsedLaneletMap


class GpkgWriter(Protocol):
    def write(
        self,
        layers: Dict[str, List[Dict[str, Any]]],
        parsed: ParsedLaneletMap,
        output_path: Path,
    ) -> MapConversionSummary:
        """Write NuPlan-compatible layer rows to a GeoPackage file."""
        ...
