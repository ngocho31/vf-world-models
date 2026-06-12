"""Pipeline stage interfaces."""

from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, Protocol, TypeVar

T = TypeVar("T")
I = TypeVar("I")
O = TypeVar("O")


@dataclass
class StageResult(Generic[T]):
    payload: T
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StageContext:
    run_id: str
    dry_run: bool = False
    source_fingerprint: Optional[str] = None


class TransformStage(Protocol[I, O]):
    name: str

    def run(self, data: I, context: StageContext) -> StageResult[O]:
        """Run one stage and return payload, warnings, and metrics."""
        ...
