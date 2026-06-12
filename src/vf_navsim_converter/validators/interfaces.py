"""Validation interface contracts."""

from dataclasses import dataclass, field
from typing import List, Protocol, Sequence

from ..contracts.models_navsim import NavsimFrameRecord


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str  # warning | error


@dataclass
class ValidationResult:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


class SchemaValidator(Protocol):
    def validate(self, frames: Sequence[NavsimFrameRecord]) -> ValidationResult:
        ...


class TemporalValidator(Protocol):
    def validate(self, frames: Sequence[NavsimFrameRecord]) -> ValidationResult:
        ...


class AssetValidator(Protocol):
    def validate(self, frames: Sequence[NavsimFrameRecord]) -> ValidationResult:
        ...
