#!/usr/bin/env python3
"""Convert a VF recording into a synchronized CaMo-JEPA dataset.

Input
-----
``--vf-root`` must contain the selected camera under
``CAMERA/<camera>/*.jpg`` and these telemetry streams under ``OTHERS/``:
``NAV``, ``IMU``, ``VEHICLE_INFO``, and ``VEHICLE_STEER``. Camera frames are
matched to the nearest row in every stream by timestamp. Frames are skipped
when any match exceeds ``--max-drift-ms``.

Output
------
The converter writes to ``--output-root`` using the CaMo-JEPA v1 layout:

* ``images/<split>/<log-name>/<timestamp>.jpg``
* ``episodes/<split>/<log-name>.npz``
* ``manifest.jsonl``
* ``dataset_info.json``

The episode stores timestamps, source timestamps, image paths, an 18-value
``can_bus`` vector, and ``ego_motion`` as
``[yaw_rate_radps, longitudinal_acceleration_mps2]``. Output is downsampled
to at most ``--target-hz`` (10 Hz by default) without changing retained
source timestamps. Images are symlinked by default; use ``--image-mode`` to
select hard links or copies.

Usage
-----
From the repository root, validate synchronization without writing output:

    python3 src/utils/convert_vf_to_camo.py --vf-root dataset_vf/data --dry-run

Convert the recording and write the default dataset to
``dataset_camo/vf``:

    python3 src/utils/convert_vf_to_camo.py --vf-root dataset_vf/data

Use ``python src/utils/convert_vf_to_camo.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VF_ROOT = REPO_ROOT / "dataset_vf" / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "dataset_camo" / "vf"
EGO_MOTION_NAMES = ("yaw_rate_radps", "longitudinal_acceleration_mps2")
REQUIRED_STREAMS = ("NAV", "IMU", "VEHICLE_INFO", "VEHICLE_STEER")


@dataclass(frozen=True)
class TelemetryRow:
    timestamp_ns: int
    values: Mapping[str, str]


@dataclass(frozen=True)
class FrameSample:
    timestamp_us: int
    source_timestamp_us: int
    token: str
    source_image: Path
    image_path: Path
    can_bus: np.ndarray
    ego_motion: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert synchronized VF images and CAN-bus telemetry into CaMo-JEPA episodes."
    )
    parser.add_argument(
        "--vf-root",
        type=Path,
        default=DEFAULT_VF_ROOT,
        help=f"VF data root containing CAMERA and OTHERS (default: {DEFAULT_VF_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination CaMo dataset root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--camera", default="CAM_P_F", help="VF camera directory (default: CAM_P_F)")
    parser.add_argument("--split", default="train", help="Output split name (default: train)")
    parser.add_argument(
        "--log-name",
        default="vf_recording_000",
        help="Output episode name for this VF recording (default: vf_recording_000)",
    )
    parser.add_argument(
        "--max-drift-ms",
        type=float,
        default=80.0,
        help="Maximum timestamp delta to every telemetry stream (default: 80)",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How output image files refer to source images (default: symlink)",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=10.0,
        help="Maximum output frequency in Hz; source timestamps are preserved (default: 10)",
    )
    parser.add_argument("--max-frames", type=int, help="Convert at most this many frames")
    parser.add_argument("--dry-run", action="store_true", help="Validate synchronization without writing files")
    return parser.parse_args()


def token_to_ns(token: str) -> int:
    try:
        seconds, subsecond = token.split("-", 1)
    except ValueError as error:
        raise ValueError(f"invalid VF timestamp token: {token}") from error
    if not seconds.isdigit():
        raise ValueError(f"invalid VF seconds component: {token}")
    digits = "".join(character for character in subsecond if character.isdigit())
    return int(seconds) * 1_000_000_000 + int((digits + "000000000")[:9] or "0")


def read_stream(stream_dir: Path) -> list[TelemetryRow]:
    rows: list[TelemetryRow] = []
    for csv_path in sorted(stream_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as csv_file:
            for row in csv.DictReader(csv_file):
                timestamp = (row.get("Timestamp") or "").strip()
                if timestamp:
                    rows.append(TelemetryRow(token_to_ns(timestamp), row))
    rows.sort(key=lambda row: row.timestamp_ns)
    return rows


def nearest_row(
    rows: list[TelemetryRow], timestamps: list[int], target_ns: int
) -> tuple[TelemetryRow, int] | None:
    index = bisect_left(timestamps, target_ns)
    candidates = rows[max(0, index - 1):index + 1]
    if not candidates:
        return None
    selected = min(candidates, key=lambda row: abs(row.timestamp_ns - target_ns))
    return selected, abs(selected.timestamp_ns - target_ns)


def value(row: Mapping[str, str], name: str, default: float = 0.0) -> float:
    try:
        result = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def quaternion(imu: Mapping[str, str]) -> tuple[float, float, float, float]:
    result = (value(imu, "qw", 1.0), value(imu, "qx"), value(imu, "qy"), value(imu, "qz"))
    norm = math.sqrt(sum(component * component for component in result))
    return tuple(component / norm for component in result) if norm else (1.0, 0.0, 0.0, 0.0)


def local_position(nav: Mapping[str, str], origin: tuple[float, float]) -> tuple[float, float, float]:
    latitude = value(nav, "Latitude")
    longitude = value(nav, "Longitude")
    origin_latitude, origin_longitude = origin
    earth_radius_m = 6_378_137.0
    northing = math.radians(latitude - origin_latitude) * earth_radius_m
    easting = math.radians(longitude - origin_longitude) * earth_radius_m * math.cos(math.radians(origin_latitude))
    return easting, northing, value(nav, "Altitude")


def build_can_bus(
    nav: Mapping[str, str],
    imu: Mapping[str, str],
    vehicle: Mapping[str, str],
    steer: Mapping[str, str],
    origin: tuple[float, float],
) -> np.ndarray:
    position = local_position(nav, origin)
    rotation = quaternion(imu)
    return np.asarray(
        (
            *position,
            *rotation,
            value(vehicle, "acceleration_x"),
            value(vehicle, "acceleration_y"),
            value(vehicle, "yaw_rate"),
            value(nav, "Ve"),
            value(nav, "Vn"),
            value(imu, "ax"),
            value(imu, "ay"),
            value(steer, "steer_angle"),
            value(steer, "steer_speed"),
            0.0,
            0.0,
        ),
        dtype=np.float32,
    )


def resample_samples(samples: list[FrameSample], target_hz: float) -> list[FrameSample]:
    """Downsample while retaining original image, CAN-bus, and timestamp values."""
    if len(samples) < 2:
        return samples
    interval_us = round(1_000_000 / target_hz)
    selected = [samples[0]]
    next_timestamp_us = samples[0].timestamp_us + interval_us
    for sample in samples[1:]:
        if sample.timestamp_us >= next_timestamp_us:
            selected.append(sample)
            next_timestamp_us = sample.timestamp_us + interval_us
    return selected


def build_samples(args: argparse.Namespace) -> tuple[list[FrameSample], int]:
    camera_root = args.vf_root / "CAMERA" / args.camera
    streams = {name: read_stream(args.vf_root / "OTHERS" / name) for name in REQUIRED_STREAMS}
    missing = [name for name, rows in streams.items() if not rows]
    if missing:
        raise ValueError(f"required VF telemetry streams are empty: {', '.join(missing)}")
    stream_timestamps = {
        name: [row.timestamp_ns for row in rows]
        for name, rows in streams.items()
    }
    image_paths = sorted(camera_root.glob("*.jpg"), key=lambda path: token_to_ns(path.stem))
    if not image_paths:
        raise ValueError(f"no JPG files found for camera: {camera_root}")

    first_nav = streams["NAV"][0].values
    origin = (value(first_nav, "Latitude"), value(first_nav, "Longitude"))
    max_drift_ns = int(args.max_drift_ms * 1_000_000)
    image_root = args.output_root / "images" / args.split / args.log_name
    samples: list[FrameSample] = []
    skipped = 0
    for source_image in image_paths:
        timestamp_ns = token_to_ns(source_image.stem)
        matches = {
            name: nearest_row(rows, stream_timestamps[name], timestamp_ns)
            for name, rows in streams.items()
        }
        if any(match is None or match[1] > max_drift_ns for match in matches.values()):
            skipped += 1
            continue
        nav, imu, vehicle, steer = (matches[name][0].values for name in REQUIRED_STREAMS)
        can_bus = build_can_bus(nav, imu, vehicle, steer, origin)
        samples.append(
            FrameSample(
                timestamp_us=timestamp_ns // 1_000,
                source_timestamp_us=timestamp_ns // 1_000,
                token=source_image.stem,
                source_image=source_image,
                image_path=image_root / source_image.name,
                can_bus=can_bus,
                ego_motion=np.asarray((can_bus[9], can_bus[7]), dtype=np.float32),
            )
        )
        if args.max_frames is not None and len(samples) >= args.max_frames:
            break
    return samples, skipped


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    if mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def write_output(args: argparse.Namespace, samples: list[FrameSample]) -> None:
    output_root = args.output_root
    episode_path = output_root / "episodes" / args.split / f"{args.log_name}.npz"
    for sample in samples:
        materialize_image(sample.source_image, sample.image_path, args.image_mode)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        episode_path,
        timestamps_us=np.asarray([sample.timestamp_us for sample in samples], dtype=np.int64),
        source_timestamps_us=np.asarray(
            [sample.source_timestamp_us for sample in samples], dtype=np.int64
        ),
        tokens=np.asarray([sample.token for sample in samples]),
        image_paths=np.asarray([sample.image_path.relative_to(output_root).as_posix() for sample in samples]),
        can_bus=np.stack([sample.can_bus for sample in samples]),
        ego_motion=np.stack([sample.ego_motion for sample in samples]),
        ego_motion_names=np.asarray(EGO_MOTION_NAMES),
    )
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as manifest_file:
        for index, sample in enumerate(samples):
            row = {
                "episode": episode_path.relative_to(output_root).as_posix(),
                "frame_index": index,
                "timestamp_us": sample.timestamp_us,
                "source_timestamp_us": sample.source_timestamp_us,
                "token": sample.token,
                "image_path": sample.image_path.relative_to(output_root).as_posix(),
                "ego_motion": sample.ego_motion.tolist(),
                "ego_motion_names": list(EGO_MOTION_NAMES),
            }
            manifest_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    dataset_info = {
        "format": "camo-jepa-v1",
        "source_dataset": "vf",
        "camera": args.camera,
        "ego_motion_names": list(EGO_MOTION_NAMES),
        "ego_motion_source": "can_bus[9], can_bus[7]",
        "can_bus_shape": [18],
        "image_mode": args.image_mode,
        "target_hz": args.target_hz,
        "resampling": "downsample_preserve_source_timestamp",
    }
    (output_root / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.vf_root = args.vf_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.max_drift_ms < 0:
        raise SystemExit("--max-drift-ms must be non-negative")
    if args.target_hz <= 0:
        raise SystemExit("--target-hz must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be positive")
    if not (args.vf_root / "CAMERA").is_dir() or not (args.vf_root / "OTHERS").is_dir():
        raise SystemExit("VF root must contain both CAMERA/ and OTHERS/")

    samples, skipped = build_samples(args)
    samples = resample_samples(samples, args.target_hz)
    if not samples:
        raise SystemExit("no camera frames met the telemetry synchronization tolerance")
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_output(args, samples)
    mode = "validated" if args.dry_run else "converted"
    print(f"[DONE] {mode} {len(samples)} synchronized frames, {skipped} skipped frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())