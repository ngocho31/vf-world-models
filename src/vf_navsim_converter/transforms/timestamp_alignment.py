"""Timestamp alignment implementation for Stage 3."""

import json
import logging
from pathlib import Path
from bisect import bisect_left
from typing import List, Sequence, Set

from ..constants import (
    VF_REQUIRED_ALIGNMENT_BUNDLE_FIELDS,
    VF_REQUIRED_CAMERAS,
)
from ..contracts.models_raw import RawSensorBundle

logger = logging.getLogger(__name__)


class TokenIntersectionTimestampAligner:
    """Align frames by intersection of camera tokens across required streams."""

    def __init__(
        self,
        max_sensor_nav_drift_ms: int = 80,
        frame_stride: int = 1,
        max_frames: int = 0,
    ) -> None:
        self._required_sensors = tuple(VF_REQUIRED_CAMERAS)
        self._required_alignment_fields = tuple(VF_REQUIRED_ALIGNMENT_BUNDLE_FIELDS)
        self._max_drift_ns = max_sensor_nav_drift_ms * 1_000_000
        self._frame_stride = max(1, frame_stride)
        self._max_frames = max(0, max_frames)

    def align(self, data: RawSensorBundle) -> Sequence[str]:
        token_sets: List[Set[str]] = []
        for sensor in self._required_sensors:
            frames = data.camera_frames_by_sensor.get(sensor, [])
            tokens = set(frame.token for frame in frames)
            if not tokens:
                raise ValueError("missing camera frames for sensor: %s" % sensor)
            token_sets.append(tokens)

        common = set.intersection(*token_sets)
        sorted_tokens = sorted(common)
        required_timestamps = {}
        for field_name in self._required_alignment_fields:
            timestamps = self._read_timestamps_for_field(data, field_name)
            if not timestamps:
                raise ValueError("%s is required for timestamp alignment" % field_name)
            required_timestamps[field_name] = timestamps

        aligned_tokens: List[str] = []
        for token in sorted_tokens:
            token_ns = self._parse_timestamp_token_to_ns(token)

            is_aligned = True

            for timestamps in required_timestamps.values():
                drift = self._nearest_abs_diff_ns(token_ns, timestamps)
                if drift > self._max_drift_ns:
                    is_aligned = False
                    break

            if not is_aligned:
                continue

            aligned_tokens.append(token)

        if self._frame_stride > 1:
            aligned_tokens = aligned_tokens[:: self._frame_stride]

        if self._max_frames > 0:
            aligned_tokens = aligned_tokens[: self._max_frames]

        return aligned_tokens

    @staticmethod
    def _read_timestamps_for_field(data: RawSensorBundle, field_name: str) -> List[int]:
        rows = getattr(data, field_name, None)
        if rows is None:
            return []
        return sorted(row.timestamp_ns for row in rows)

    @staticmethod
    def _parse_timestamp_token_to_ns(token: str) -> int:
        try:
            sec_str, subsec_str = token.split("-", 1)
        except ValueError as exc:
            raise ValueError("Invalid timestamp token: %s" % token) from exc

        if not sec_str.isdigit():
            raise ValueError("Invalid second component in token: %s" % token)

        subsec_digits = "".join(ch for ch in subsec_str if ch.isdigit())
        if not subsec_digits:
            subsec_digits = "0"

        ns_component = int((subsec_digits + "000000000")[:9])
        return int(sec_str) * 1_000_000_000 + ns_component

    @staticmethod
    def _nearest_abs_diff_ns(target_ns: int, sorted_timestamps_ns: Sequence[int]) -> int:
        idx = bisect_left(sorted_timestamps_ns, target_ns)
        candidates = []
        if idx < len(sorted_timestamps_ns):
            candidates.append(abs(sorted_timestamps_ns[idx] - target_ns))
        if idx > 0:
            candidates.append(abs(sorted_timestamps_ns[idx - 1] - target_ns))
        if not candidates:
            return 2**63 - 1
        return min(candidates)


def save_timestamp_alignment_manifest(aligned_tokens: Sequence[str], output_path: Path) -> None:
    """Save a JSON manifest of the timestamp alignment results for debugging and visibility."""
    manifest_data = {
        "stage": "align_timestamps",
        "status": "ok",
        "aligned_token_count": len(aligned_tokens),
        "sample": {
            "first_token": aligned_tokens[0],
            "last_token": aligned_tokens[-1],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_data, handle, indent=2)

    logger.info(f"Raw sensor bundle manifest saved to {output_path}")
