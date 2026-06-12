"""Validators for Stage 4 Canonical Frames output."""

import json
import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence

from ..contracts.models_canonical import CanonicalBatch, CanonicalFrame
from ..constants import VF_REQUIRED_CAMERAS, VF_TO_NAVSIM_CAMERA_MAP
from .interfaces import ValidationIssue, ValidationResult

logger = logging.getLogger(__name__)


class CanonicalFrameValidator:
    """Validates CanonicalBatch output from Stage 4."""

    def __init__(self, check_file_paths: bool = False) -> None:
        """
        Initialize validator.

        Args:
            check_file_paths: If True, verify that referenced camera files exist on disk.
        """
        self._check_file_paths = check_file_paths

    def validate(self, batch: CanonicalBatch) -> ValidationResult:
        """Validate a CanonicalBatch and return all issues found."""
        issues: List[ValidationIssue] = []

        if not batch.frames:
            issues.append(
                ValidationIssue(
                    code="EMPTY_BATCH",
                    message="CanonicalBatch is empty (no frames)",
                    severity="error",
                )
            )
            return ValidationResult(issues=issues)

        # Validate structure and content
        issues.extend(self._validate_frame_structure(batch.frames))
        issues.extend(self._validate_ego_states(batch.frames))
        issues.extend(self._validate_cameras(batch.frames))
        issues.extend(self._validate_timestamps(batch.frames))
        issues.extend(self._validate_transforms(batch.frames))

        if self._check_file_paths:
            issues.extend(self._validate_file_paths(batch.frames))

        return ValidationResult(issues=issues)

    def _validate_frame_structure(self, frames: Sequence[CanonicalFrame]) -> List[ValidationIssue]:
        """Validate basic frame structure."""
        issues: List[ValidationIssue] = []

        for idx, frame in enumerate(frames):
            if not frame.token or not isinstance(frame.token, str):
                issues.append(
                    ValidationIssue(
                        code="INVALID_TOKEN",
                        message=f"Frame {idx}: token must be non-empty string, got {type(frame.token)}",
                        severity="error",
                    )
                )

            if frame.timestamp_ns <= 0:
                issues.append(
                    ValidationIssue(
                        code="INVALID_TIMESTAMP",
                        message=f"Frame {idx} ({frame.token}): timestamp_ns must be positive, got {frame.timestamp_ns}",
                        severity="error",
                    )
                )

            if frame.ego_state is None:
                issues.append(
                    ValidationIssue(
                        code="MISSING_EGO_STATE",
                        message=f"Frame {idx} ({frame.token}): ego_state is None",
                        severity="error",
                    )
                )

            if not frame.cameras:
                issues.append(
                    ValidationIssue(
                        code="MISSING_CAMERAS",
                        message=f"Frame {idx} ({frame.token}): cameras dict is empty",
                        severity="error",
                    )
                )

        return issues

    def _validate_ego_states(self, frames: Sequence[CanonicalFrame]) -> List[ValidationIssue]:
        """Validate ego state values and ranges."""
        issues: List[ValidationIssue] = []

        for idx, frame in enumerate(frames):
            ego = frame.ego_state
            if ego is None:
                continue

            # Validate translation (meters, reasonable global bounds)
            translation = ego.translation_xyz_m
            if len(translation) != 3:
                issues.append(
                    ValidationIssue(
                        code="INVALID_TRANSLATION_LEN",
                        message=f"Frame {idx} ({frame.token}): translation must have 3 elements, got {len(translation)}",
                        severity="error",
                    )
                )
            else:
                # Check for NaN and extreme values (±1e6 meters ~ ±1000km)
                for i, val in enumerate(translation):
                    if not isinstance(val, (int, float)) or math.isnan(val):
                        issues.append(
                            ValidationIssue(
                                code="INVALID_TRANSLATION_VALUE",
                                message=f"Frame {idx} ({frame.token}): translation[{i}] is NaN or not numeric",
                                severity="error",
                            )
                        )
                    elif abs(val) > 1e6:
                        issues.append(
                            ValidationIssue(
                                code="EXTREME_TRANSLATION",
                                message=f"Frame {idx} ({frame.token}): translation[{i}]={val} exceeds bounds",
                                severity="warning",
                            )
                        )

            # Validate rotation (quaternion wxyz, should be normalized)
            rotation = ego.rotation_wxyz
            if len(rotation) != 4:
                issues.append(
                    ValidationIssue(
                        code="INVALID_ROTATION_LEN",
                        message=f"Frame {idx} ({frame.token}): rotation must have 4 elements (wxyz), got {len(rotation)}",
                        severity="error",
                    )
                )
            else:
                norm = sum(x * x for x in rotation) ** 0.5
                if abs(norm - 1.0) > 0.01:  # Allow small denormalization
                    issues.append(
                        ValidationIssue(
                            code="DENORMALIZED_QUATERNION",
                            message=f"Frame {idx} ({frame.token}): quaternion norm={norm:.4f}, expected ~1.0",
                            severity="warning",
                        )
                    )

            # Validate dynamic state [vx, vy, ax, ay]
            dyn = ego.dynamic_state
            if len(dyn) != 4:
                issues.append(
                    ValidationIssue(
                        code="INVALID_DYNAMIC_STATE_LEN",
                        message=f"Frame {idx} ({frame.token}): dynamic_state must have 4 elements, got {len(dyn)}",
                        severity="error",
                    )
                )
            else:
                for i, val in enumerate(dyn):
                    if math.isnan(val):
                        issues.append(
                            ValidationIssue(
                                code="NAN_DYNAMIC_STATE",
                                message=f"Frame {idx} ({frame.token}): dynamic_state[{i}] is NaN",
                                severity="error",
                            )
                        )

        return issues

    def _validate_cameras(self, frames: Sequence[CanonicalFrame]) -> List[ValidationIssue]:
        """Validate camera samples."""
        issues: List[ValidationIssue] = []
        expected_cameras = set(VF_TO_NAVSIM_CAMERA_MAP.values())

        for idx, frame in enumerate(frames):
            actual_cameras = set(frame.cameras.keys())

            # Check required cameras present
            missing = expected_cameras - actual_cameras
            if missing:
                issues.append(
                    ValidationIssue(
                        code="MISSING_REQUIRED_CAMERAS",
                        message=f"Frame {idx} ({frame.token}): missing cameras {missing}",
                        severity="error",
                    )
                )

            # Check for extra unexpected cameras
            extra = actual_cameras - expected_cameras
            if extra:
                issues.append(
                    ValidationIssue(
                        code="EXTRA_CAMERAS",
                        message=f"Frame {idx} ({frame.token}): unexpected cameras {extra}",
                        severity="warning",
                    )
                )

            # Validate each camera sample
            for cam_name, cam_sample in frame.cameras.items():
                self._validate_camera_sample(idx, frame.token, cam_name, cam_sample, issues)

        return issues

    def _validate_camera_sample(
        self,
        frame_idx: int,
        frame_token: str,
        cam_name: str,
        cam_sample: object,
        issues: List[ValidationIssue],
    ) -> None:
        """Validate a single camera sample."""
        if not hasattr(cam_sample, "relative_path"):
            issues.append(
                ValidationIssue(
                    code="MISSING_CAMERA_PATH",
                    message=f"Frame {frame_idx} ({frame_token}): camera {cam_name} missing relative_path",
                    severity="error",
                )
            )

        if not hasattr(cam_sample, "cam_intrinsic"):
            issues.append(
                ValidationIssue(
                    code="MISSING_INTRINSICS",
                    message=f"Frame {frame_idx} ({frame_token}): camera {cam_name} missing cam_intrinsic",
                    severity="error",
                )
            )
        else:
            intrinsic = cam_sample.cam_intrinsic
            if not isinstance(intrinsic, (list, tuple)) or len(intrinsic) != 3:
                issues.append(
                    ValidationIssue(
                        code="INVALID_INTRINSIC_LEN",
                        message=f"Frame {frame_idx} ({frame_token}): camera {cam_name} intrinsic must be 3x3 matrix",
                        severity="error",
                    )
                )
            elif len(intrinsic) == 3 and all(len(row) == 3 for row in intrinsic):
                # Check for reasonable focal lengths
                fx = intrinsic[0][0]
                fy = intrinsic[1][1]
                if fx <= 0 or fy <= 0:
                    issues.append(
                        ValidationIssue(
                            code="INVALID_FOCAL_LENGTH",
                            message=f"Frame {frame_idx} ({frame_token}): camera {cam_name} focal length must be positive",
                            severity="error",
                        )
                    )

        if not hasattr(cam_sample, "sensor2lidar_rotation") or not hasattr(cam_sample, "sensor2lidar_translation"):
            issues.append(
                ValidationIssue(
                    code="MISSING_EXTRINSICS",
                    message=f"Frame {frame_idx} ({frame_token}): camera {cam_name} missing sensor2lidar transform",
                    severity="error",
                )
            )

    def _validate_timestamps(self, frames: Sequence[CanonicalFrame]) -> List[ValidationIssue]:
        """Validate timestamp monotonicity."""
        issues: List[ValidationIssue] = []

        prev_ts = -1
        for idx, frame in enumerate(frames):
            if frame.timestamp_ns <= prev_ts:
                issues.append(
                    ValidationIssue(
                        code="NON_MONOTONIC_TIMESTAMPS",
                        message=f"Frame {idx} ({frame.token}): timestamp {frame.timestamp_ns} not strictly increasing (prev: {prev_ts})",
                        severity="error",
                    )
                )
            prev_ts = frame.timestamp_ns

        return issues

    def _validate_transforms(self, frames: Sequence[CanonicalFrame]) -> List[ValidationIssue]:
        """Validate rotation matrix and translation vectors in camera samples."""
        issues: List[ValidationIssue] = []

        for idx, frame in enumerate(frames):
            for cam_name, cam_sample in frame.cameras.items():
                if hasattr(cam_sample, "sensor2lidar_rotation"):
                    rot = cam_sample.sensor2lidar_rotation
                    # Check 3x3 rotation matrix
                    if not isinstance(rot, (list, tuple)) or len(rot) != 3:
                        issues.append(
                            ValidationIssue(
                                code="INVALID_ROTATION_MATRIX_LEN",
                                message=f"Frame {idx} ({frame.token}): camera {cam_name} rotation must be 3x3",
                                severity="error",
                            )
                        )
                    elif not all(len(row) == 3 for row in rot):
                        issues.append(
                            ValidationIssue(
                                code="INVALID_ROTATION_MATRIX_SHAPE",
                                message=f"Frame {idx} ({frame.token}): camera {cam_name} rotation rows have inconsistent length",
                                severity="error",
                            )
                        )

                if hasattr(cam_sample, "sensor2lidar_translation"):
                    trans = cam_sample.sensor2lidar_translation
                    if not isinstance(trans, (list, tuple)) or len(trans) != 3:
                        issues.append(
                            ValidationIssue(
                                code="INVALID_TRANSLATION_VECTOR_LEN",
                                message=f"Frame {idx} ({frame.token}): camera {cam_name} translation must be 3D vector",
                                severity="error",
                            )
                        )

        return issues

    def _validate_file_paths(self, frames: Sequence[CanonicalFrame]) -> List[ValidationIssue]:
        """Validate that referenced camera files exist (if check_file_paths=True)."""
        issues: List[ValidationIssue] = []

        for idx, frame in enumerate(frames):
            for cam_name, cam_sample in frame.cameras.items():
                if hasattr(cam_sample, "relative_path"):
                    path_str = cam_sample.relative_path
                    if isinstance(path_str, str) and path_str:
                        path = Path(path_str)
                        if not path.exists():
                            issues.append(
                                ValidationIssue(
                                    code="MISSING_CAMERA_FILE",
                                    message=f"Frame {idx} ({frame.token}): camera {cam_name} file not found: {path_str}",
                                    severity="error",
                                )
                            )

        return issues


def save_canonical_validation_report(
    batch: CanonicalBatch,
    result: ValidationResult,
    output_path: Path,
) -> None:
    """Save validation report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "frame_count": len(batch.frames),
        "validation_status": "passed" if not result.has_errors else "failed",
        "error_count": sum(1 for issue in result.issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in result.issues if issue.severity == "warning"),
        "issues": [
            {"code": issue.code, "message": issue.message, "severity": issue.severity}
            for issue in result.issues
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation report saved to {output_path}")
    logger.info(f"Status: {report['validation_status']} ({report['error_count']} errors, {report['warning_count']} warnings)")
