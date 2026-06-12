"""Transform interface contracts."""

from typing import Protocol, Sequence

from ..contracts.models_canonical import CanonicalBatch, CanonicalFrame
from ..contracts.models_navsim import NavsimSceneArtifact, SceneMapSyncContext
from ..contracts.models_raw import RawSensorBundle


class TimestampAligner(Protocol):
    def align(self, data: RawSensorBundle) -> Sequence[str]:
        """Return synchronized frame tokens."""
        ...


class CanonicalFrameBuilder(Protocol):
    def build(self, data: RawSensorBundle, tokens: Sequence[str]) -> CanonicalBatch:
        """Build canonical frames for synchronized tokens."""
        ...


class SceneMapSyncResolver(Protocol):
    def resolve(self, raw_bundle: RawSensorBundle) -> SceneMapSyncContext:
        """Resolve scene-map synchronization context for NAVSIM assembly."""
        ...


class NavsimAssembler(Protocol):
    def assemble(
        self,
        frames: Sequence[CanonicalFrame],
        scene_map_context: SceneMapSyncContext | None = None,
    ) -> NavsimSceneArtifact:
        """Assemble NAVSIM-compatible scene artifacts."""
        ...
