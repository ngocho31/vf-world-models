#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from navsim.agents.drive_jepa_perception_free.drive_jepa_agent import DriveJEPAAgent
from navsim.common.dataclasses import AgentInput, Camera, Cameras, EgoStatus, Lidar


@dataclass
class NavSample:
    timestamp_ns: int
    ve: float
    vn: float
    acc_x: float
    acc_y: float


@dataclass
class SteerSample:
    timestamp_ns: int
    steer_angle: float


def parse_timestamp_ns(token: str) -> int:
    return int(token.strip().replace("-", ""))


def load_nav_samples(nav_dir: Path) -> List[NavSample]:
    samples: List[NavSample] = []
    csv_files = sorted(nav_dir.glob("*.csv"))

    for csv_file in csv_files:
        with csv_file.open("r", newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = None
                for k in row.keys():
                    if k and k.strip().lower() in ["timestamp", "time", "ts", "sys_time"]:
                        ts = row[k]
                        break
                if not ts:
                    continue
                try:
                    sample = NavSample(
                        timestamp_ns=parse_timestamp_ns(ts),
                        ve=float(row.get("vx", row.get("Ve", 0.0)) or 0.0),
                        vn=float(row.get("vy", row.get("Vn", 0.0)) or 0.0),
                        acc_x=float(row.get("ax", row.get("AccX", 0.0)) or 0.0),
                        acc_y=float(row.get("ay", row.get("AccY", 0.0)) or 0.0),
                    )
                    samples.append(sample)
                except ValueError:
                    continue

    samples.sort(key=lambda item: item.timestamp_ns)
    return samples


def load_steer_samples(steer_dir: Path) -> List[SteerSample]:
    samples: List[SteerSample] = []
    csv_files = sorted(steer_dir.glob("*.csv"))

    for csv_file in csv_files:
        with csv_file.open("r", newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                # Cần ít nhất 2 cột: timestamp và value
                if len(row) < 2:
                    continue
                
                # Heuristic: cột đầu tiên thường là Timestamp, cột cuối thường là Value
                ts_str = row[0]
                val_str = row[-1]
                
                try:
                    ts = parse_timestamp_ns(ts_str)
                    val = float(val_str)
                    samples.append(SteerSample(timestamp_ns=ts, steer_angle=val))
                except ValueError:
                    # Gặp dòng header không parse được sang số thì bỏ qua
                    pass

    samples.sort(key=lambda item: item.timestamp_ns)
    return samples


def _find_closest_sample(samples: List, timestamp_ns: int):
    if not samples:
        return None

    ts_list = [sample.timestamp_ns for sample in samples]
    idx = bisect.bisect_left(ts_list, timestamp_ns)

    if idx <= 0:
        return samples[0]
    if idx >= len(samples):
        return samples[-1]

    before = samples[idx - 1]
    after = samples[idx]
    if abs(before.timestamp_ns - timestamp_ns) <= abs(after.timestamp_ns - timestamp_ns):
        return before
    return after

def find_closest_nav_sample(samples: List[NavSample], timestamp_ns: int) -> Optional[NavSample]:
    return _find_closest_sample(samples, timestamp_ns)

def find_closest_steer_sample(samples: List[SteerSample], timestamp_ns: int) -> Optional[SteerSample]:
    return _find_closest_sample(samples, timestamp_ns)


def make_cameras_from_front(front_rgb: np.ndarray) -> Cameras:
    return Cameras(
        cam_f0=Camera(image=front_rgb),
        cam_l0=Camera(),
        cam_l1=Camera(),
        cam_l2=Camera(),
        cam_r0=Camera(),
        cam_r1=Camera(),
        cam_r2=Camera(),
        cam_b0=Camera(),
    )


def build_agent_input(
    prev_image_path: Path,
    curr_image_path: Path,
    nav_samples: List[NavSample],
) -> AgentInput:
    prev_rgb = np.array(Image.open(prev_image_path).convert("RGB"))
    curr_rgb = np.array(Image.open(curr_image_path).convert("RGB"))

    curr_ts = parse_timestamp_ns(curr_image_path.stem)
    prev_ts = parse_timestamp_ns(prev_image_path.stem)

    curr_nav = find_closest_nav_sample(nav_samples, curr_ts)
    prev_nav = find_closest_nav_sample(nav_samples, prev_ts)

    driving_command = np.array([0, 1, 0, 0], dtype=np.int64)

    def ego_status_from_nav(sample: Optional[NavSample]) -> EgoStatus:
        if sample is None:
            velocity = np.array([0.0, 0.0], dtype=np.float32)
            acceleration = np.array([0.0, 0.0], dtype=np.float32)
        else:
            velocity = np.array([sample.ve, sample.vn], dtype=np.float32)
            acceleration = np.array([sample.acc_x, sample.acc_y], dtype=np.float32)

        return EgoStatus(
            ego_pose=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ego_velocity=velocity,
            ego_acceleration=acceleration,
            driving_command=driving_command,
            in_global_frame=False,
        )

    ego_statuses = [ego_status_from_nav(prev_nav), ego_status_from_nav(curr_nav)]
    cameras = [make_cameras_from_front(prev_rgb), make_cameras_from_front(curr_rgb)]
    lidars = [Lidar(), Lidar()]

    return AgentInput(ego_statuses=ego_statuses, cameras=cameras, lidars=lidars)


def draw_trajectory_bev(trajectory_xyh: np.ndarray, width: int, height: int, scale_px_per_meter: float) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    origin_x = width // 2
    origin_y = int(height * 0.9)

    # Thêm trục đơn vị đo lường (grid & ticks)
    grid_color = (220, 220, 220)
    text_color = (130, 130, 130)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Vẽ vạch mét cho trục X (forward) - khoảng cách 2.5m (tăng độ chi tiết vì tăng scale x4)
    for x_m in np.arange(0, 120, 2.5):
        py = int(origin_y - x_m * scale_px_per_meter)
        if py < 0:
            break
        # Vẽ nét đứt nhạt cho các vạch 2.5m, nét liền cho vạch 10m
        is_major = (x_m % 10 == 0)
        color = (180, 180, 180) if is_major else (235, 235, 235)
        thickness = 2 if is_major else 1
        cv2.line(canvas, (0, py), (width, py), color, thickness)
        if x_m != 0 and is_major:
            cv2.putText(canvas, f"{int(x_m)}m", (origin_x + 5, py - 5), font, 0.5, (100, 100, 100), 1)

    # Vẽ vạch mét cho trục Y (left/right) - khoảng cách 2.5m
    for y_m in np.arange(-50, 55, 2.5):
        if y_m == 0:
            continue
        px = int(origin_x - y_m * scale_px_per_meter)
        if px < 0 or px >= width:
            continue
        is_major = (y_m % 10 == 0)
        color = (180, 180, 180) if is_major else (235, 235, 235)
        thickness = 2 if is_major else 1
        cv2.line(canvas, (px, 0), (px, height), color, thickness)
        if is_major:
            cv2.putText(canvas, f"{int(y_m)}m", (px + 5, origin_y - 5), font, 0.5, (100, 100, 100), 1)

    # Vẽ trục chính X, Y
    cv2.line(canvas, (origin_x, 0), (origin_x, height), (150, 150, 150), 2)
    cv2.line(canvas, (0, origin_y), (width, origin_y), (150, 150, 150), 2)
    cv2.circle(canvas, (origin_x, origin_y), 5, (0, 0, 0), -1)

    # Plot các điểm quỹ đạo
    points: List[Tuple[int, int]] = []
    for x_m, y_m, _ in trajectory_xyh:
        px = int(origin_x - y_m * scale_px_per_meter)
        py = int(origin_y - x_m * scale_px_per_meter)
        points.append((px, py))

    for i in range(1, len(points)):
        cv2.line(canvas, points[i - 1], points[i], (0, 0, 255), 2)

    for p in points:
        cv2.circle(canvas, p, 2, (255, 0, 0), -1)

    cv2.putText(canvas, "BEV predicted trajectory", (20, 30), font, 0.8, (40, 40, 40), 2)
    cv2.putText(canvas, "x: forward (m), y: left (m)", (20, 60), font, 0.6, (80, 80, 80), 1)
    
    return canvas


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhanced Qualitative VF evaluation with Telemetry HUD and Metric BEV.")
    # Required arguments
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Planner checkpoint path (.ckpt).")
    parser.add_argument("--encoder-path", type=Path, required=True, help="V-JEPA encoder checkpoint path (.pt).")
    parser.add_argument("--agent-config", type=Path, required=True, help="Path to the agent Hydra YAML config.")
    parser.add_argument("--camera-dir", type=Path, required=True, help="Path to VF front camera JPG directory (e.g. CAM_P_F).")

    # Optional data inputs
    parser.add_argument("--nav-dir", type=Path, default=None, help="Path to VF NAV csv directory. Optional.")
    parser.add_argument("--steer-dir", type=Path, default=None, help="Path to VF VEHICLE_STEER csv dir. Optional.")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/vf_qualitative_eval_enhanced"), help="Output directory.")

    # Video/rendering parameters
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum number of output frames.")
    parser.add_argument("--frame-stride", type=int, default=3, help="Take one every N frames.")
    parser.add_argument(
        "--sample-gap-seconds",
        type=float,
        default=0.0,
        help="Minimum timestamp gap (seconds) between selected frames. 0 disables time-based sampling.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for frame window selection. Omit for a different random window each run.")
    parser.add_argument("--scale", type=float, default=40.0, help="Pixels per meter for BEV drawing.")

    return parser


def validate_torch_artifact(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size < 1024:
        raise RuntimeError(f"{label} looks too small ({path.stat().st_size} bytes): {path}")

    with path.open("rb") as f:
        head = f.read(512)

    head_lower = head.lower()
    if head.startswith(b"<") or b"<!doctype html" in head_lower or b"<html" in head_lower:
        raise RuntimeError(
            f"{label} is HTML, not a PyTorch artifact: {path}\n"
            "This usually means the download URL returned a web page (403/404/login) instead of the checkpoint file."
        )


def select_frame_indices(image_paths: List[Path], frame_stride: int, sample_gap_seconds: float) -> List[int]:
    if len(image_paths) == 0:
        return []

    stride = max(1, frame_stride)

    if sample_gap_seconds <= 0:
        return list(range(0, len(image_paths), stride))

    min_gap_ns = int(sample_gap_seconds * 1e9)
    selected: List[int] = [0]
    last_ts = parse_timestamp_ns(image_paths[0].stem)

    for idx in range(stride, len(image_paths), stride):
        ts = parse_timestamp_ns(image_paths[idx].stem)
        if ts - last_ts >= min_gap_ns:
            selected.append(idx)
            last_ts = ts

    if selected[-1] != len(image_paths) - 1:
        selected.append(len(image_paths) - 1)

    return selected


def main() -> None:
    args = make_parser().parse_args()

    # ── Step 1: Validate inputs ───────────────────────────────────────────────
    validate_torch_artifact(args.checkpoint_path, "Planner checkpoint")
    validate_torch_artifact(args.encoder_path, "Encoder checkpoint")
    if not args.agent_config.exists():
        raise FileNotFoundError(f"Agent config not found: {args.agent_config}")

    image_paths = sorted(args.camera_dir.glob("*.jpg"))
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least 2 images in {args.camera_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir = args.output_dir / "frames"
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 2: Load NAV & STEER samples ──────────────────────────────────────
    nav_samples: List[NavSample] = []
    if args.nav_dir is not None and args.nav_dir.exists():
        nav_samples = load_nav_samples(args.nav_dir)
        print(f"[INFO] Loaded {len(nav_samples)} NAV samples from {args.nav_dir}")
    else:
        print("[INFO] NAV directory not provided or missing. Using zero velocity/acceleration fallback.")
        
    steer_samples: List[SteerSample] = []
    if args.steer_dir is not None and args.steer_dir.exists():
        steer_samples = load_steer_samples(args.steer_dir)
        print(f"[INFO] Loaded {len(steer_samples)} STEER samples from {args.steer_dir}")
    else:
        print("[INFO] STEER directory not provided or missing.")

    # ── Step 3: Build agent from YAML config ─────────────────────────────────
    agent_cfg = OmegaConf.load(args.agent_config)
    OmegaConf.update(agent_cfg, "checkpoint_path", str(args.checkpoint_path))
    OmegaConf.update(agent_cfg, "pretrain_pt_path", str(args.encoder_path))
    agent: DriveJEPAAgent = instantiate(agent_cfg)
    trajectory_sampling = agent._trajectory_sampling
    print(f"[INFO] Agent config:    {args.agent_config}")
    print(f"[INFO] double_image={agent._double_image} ({'prev + current frame' if agent._double_image else 'current frame only'}), horizon={trajectory_sampling.time_horizon}s")

    # ── Step 4: Load checkpoint weights ──────────────────────────────────────
    agent.initialize()
    print(f"[INFO] Planner checkpoint: {args.checkpoint_path}")
    print(f"[INFO] Encoder checkpoint: {args.encoder_path}")

    # ── Step 5: Select frame pairs ────────────────────────────────────────────
    selected_indices = select_frame_indices(
        image_paths=image_paths,
        frame_stride=args.frame_stride,
        sample_gap_seconds=args.sample_gap_seconds,
    )
    if len(selected_indices) < 2:
        raise RuntimeError("Not enough selected frames after sampling.")

    pair_indices: List[Tuple[int, int]] = [
        (selected_indices[i - 1], selected_indices[i]) for i in range(1, len(selected_indices))
    ]
    if args.max_frames > 0 and len(pair_indices) > args.max_frames:
        seed = args.seed if args.seed is not None else random.randint(0, 2**31)
        rng = random.Random(seed)
        max_start = len(pair_indices) - args.max_frames
        start = rng.randint(0, max_start)
        pair_indices = pair_indices[start : start + args.max_frames]
        print(f"[INFO] Random window: seed={seed}, start_pair={start}, pairs={len(pair_indices)}")

    print(
        f"[INFO] Sampling: {len(selected_indices)} anchor frames → "
        f"{len(pair_indices)} pairs "
        f"(frame_stride={args.frame_stride})"
    )

    # ── Step 6: Run inference and render ─────────────────────────────────────
    video_writer: Optional[cv2.VideoWriter] = None
    records = []

    for out_idx, (prev_idx, curr_idx) in enumerate(pair_indices):
        prev_image_path = image_paths[prev_idx]
        curr_image_path = image_paths[curr_idx]
        curr_ts = parse_timestamp_ns(curr_image_path.stem)

        agent_input = build_agent_input(prev_image_path, curr_image_path, nav_samples)
        trajectory = agent.compute_trajectory(agent_input)
        poses = trajectory.poses.astype(np.float32)

        curr_bgr = cv2.imread(str(curr_image_path))
        if curr_bgr is None:
            continue

        # Extract Telemetry Ground Truth
        curr_nav = find_closest_nav_sample(nav_samples, curr_ts)
        curr_steer = find_closest_steer_sample(steer_samples, curr_ts)

        speed_mps = np.hypot(curr_nav.ve, curr_nav.vn) if curr_nav else 0.0
        speed_kph = speed_mps * 3.6
        acc = np.hypot(curr_nav.acc_x, curr_nav.acc_y) if curr_nav else 0.0
        steer_angle = curr_steer.steer_angle if curr_steer else 0.0

        # Draw Telemetry HUD
        hud_x, hud_y = 30, 40
        overlay = curr_bgr.copy()
        cv2.rectangle(overlay, (hud_x - 10, hud_y - 30), (hud_x + 350, hud_y + 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, curr_bgr, 0.5, 0, curr_bgr) # Semi-transparent background
        
        cv2.putText(curr_bgr, "Telemetry GT", (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(curr_bgr, f"Speed: {speed_kph:.1f} km/h ({speed_mps:.1f} m/s)", (hud_x, hud_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(curr_bgr, f"Accel: {acc:.2f} m/s^2", (hud_x, hud_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(curr_bgr, f"Steer: {steer_angle:.2f} deg", (hud_x, hud_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(curr_bgr, f"Timestamp: {curr_image_path.stem}", (hud_x, curr_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Build final canvas
        h, w = curr_bgr.shape[:2]
        bev = draw_trajectory_bev(poses, width=w, height=h, scale_px_per_meter=args.scale)
        canvas = np.concatenate([curr_bgr, bev], axis=1)

        text = f"pred_horizon={trajectory_sampling.time_horizon:.1f}s"
        cv2.putText(canvas, text, (curr_bgr.shape[1] + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 220, 20), 2)

        frame_out_path = output_frames_dir / f"{out_idx:06d}.jpg"
        cv2.imwrite(str(frame_out_path), canvas)

        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = args.output_dir / "vf_qualitative_eval_enhanced.mp4"
            video_writer = cv2.VideoWriter(str(video_path), fourcc, float(args.fps), (canvas.shape[1], canvas.shape[0]))

        video_writer.write(canvas)

        records.append(
            {
                "frame": curr_image_path.stem,
                "prev_frame": prev_image_path.stem,
                "trajectory_xyh": poses.tolist(),
                "telemetry": {
                    "speed_mps": float(speed_mps),
                    "acc_mps2": float(acc),
                    "steer_angle_deg": float(steer_angle),
                }
            }
        )

        if (out_idx + 1) % 20 == 0:
            print(f"[INFO] Processed {out_idx + 1}/{len(pair_indices)} pairs")

    if video_writer is not None:
        video_writer.release()

    trajectories_path = args.output_dir / "predicted_trajectories_enhanced.json"
    with trajectories_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("[INFO] Done.")
    print(f"[INFO] Frames:       {output_frames_dir}")
    print(f"[INFO] Video:        {args.output_dir / 'vf_qualitative_eval_enhanced.mp4'}")
    print(f"[INFO] Trajectories: {trajectories_path}")

if __name__ == "__main__":
    main()
