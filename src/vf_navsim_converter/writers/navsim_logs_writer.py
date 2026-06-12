"""Default implementation for writing NAVSIM pickle artifacts (Stage 7)."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from ..contracts.models_navsim import NavsimFrameRecord
from .interfaces import NavsimLogsWriter


class DefaultNavsimLogsWriter(NavsimLogsWriter):
    """Write NAVSIM frame dictionaries to a pickle artifact."""

    def write_frames(self, frames: Sequence[NavsimFrameRecord], output_pkl_path: Path) -> None:
        output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(frame) for frame in frames]
        with output_pkl_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_stage7_artifact_manifest(
    frames: Sequence[NavsimFrameRecord],
    output_pkl_path: Path,
    manifest_path: Path,
) -> None:
    """Persist a compact Stage 7 manifest for debugging and auditability."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame = frames[0] if frames else None
    sample_cameras = sorted(first_frame.cams.keys()) if first_frame else []
    payload = {
        "frame_count": len(frames),
        "log_name": first_frame.log_name if first_frame else "",
        "output_pkl_path": str(output_pkl_path),
        "sample_frame": {
            "token": first_frame.token,
            "timestamp": first_frame.timestamp,
            "camera_keys": sample_cameras,
            "lidar_path": first_frame.lidar_path,
        }
        if first_frame
        else None,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_conversion_report(
    frames: Sequence[NavsimFrameRecord],
    output_pkl_path: Path,
    report_path: Path,
) -> None:
    """Write a conversion report summarizing known gaps vs. full NAVSIM supervision."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = frames[0] if frames else None
    route_count = len(first_frame.roadblock_ids) if first_frame else 0
    map_location = first_frame.map_location if first_frame else ""

    missing_summary = {
        "roadblock_ids": (
            f"route-aware road segment sequence inferred from NAV trajectory + map graph (count={route_count})"
            if route_count > 0
            else "empty (map overlap below threshold or route inference unavailable)"
        ),
        "traffic_lights": "not derived from traffic light state or signal timeline",
        "anns": "3D box, class, velocity, instance, and track annotations are not loaded from VF raw inputs",
        "map_location": (
            f"synchronized value='{map_location}'"
            if map_location
            else "empty"
        ),
    }

    payload = {
        "frame_count": len(frames),
        "log_name": first_frame.log_name if first_frame else "",
        "output_pkl_path": str(output_pkl_path),
        "synchronization": {
            "map_location": map_location,
            "route_roadblock_count": route_count,
        },
        "known_gaps_vs_navsim": missing_summary,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
