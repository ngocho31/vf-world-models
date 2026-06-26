#!/usr/bin/env python3
"""
VF Qualitative Evaluation — GT vs Prediction Overlay
=====================================================
Overlay Ground Truth (RED) and Prediction (BLUE) trajectories onto a video.

GT is reconstructed via dead-reckoning from NAV/IMU velocity data at future timestamps.
Prediction comes from DriveJEPA agent.compute_trajectory().
"""
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NavSample:
    timestamp_ns: int
    ve: float       # velocity east (m/s) hoặc vx forward
    vn: float       # velocity north (m/s) hoặc vy lateral
    acc_x: float
    acc_y: float


@dataclass
class SteerSample:
    timestamp_ns: int
    steer_angle: float


# ═══════════════════════════════════════════════════════════════════════════════
# TIMESTAMP & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

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
                    ts_val = parse_timestamp_ns(ts)
                    # Support both VEHICLE_STEER format and fallback
                    steer_val = float(row.get("steer_angle", 0.0))
                    samples.append(SteerSample(timestamp_ns=ts_val, steer_angle=steer_val))
                except ValueError:
                    pass
    samples.sort(key=lambda item: item.timestamp_ns)
    return samples


def _find_closest_sample(samples: List, timestamp_ns: int):
    if not samples:
        return None
    ts_list = [s.timestamp_ns for s in samples]
    idx = bisect.bisect_left(ts_list, timestamp_ns)
    if idx <= 0:
        return samples[0]
    if idx >= len(samples):
        return samples[-1]
    before, after = samples[idx - 1], samples[idx]
    return before if abs(before.timestamp_ns - timestamp_ns) <= abs(after.timestamp_ns - timestamp_ns) else after


def find_closest_nav_sample(samples: List[NavSample], timestamp_ns: int) -> Optional[NavSample]:
    return _find_closest_sample(samples, timestamp_ns)

def find_closest_steer_sample(samples: List[SteerSample], timestamp_ns: int) -> Optional[SteerSample]:
    return _find_closest_sample(samples, timestamp_ns)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH TRAJECTORY (Dead Reckoning)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gt_trajectory(
    nav_samples: List[NavSample],
    curr_ts_ns: int,
    future_image_paths: List[Path],
    num_poses: int = 8,
    interval_s: float = 0.5,
) -> np.ndarray:
    """
    Tính quỹ đạo Ground Truth bằng Dead Reckoning từ dữ liệu NAV/IMU.

    Phương pháp:
    1. Từ frame hiện tại (thời điểm t), nhìn về phía trước 4 giây
       (8 waypoints × 0.5s/waypoint)
    2. Tại mỗi thời điểm tương lai, lấy vận tốc (ve, vn) từ NAV data
    3. Tích phân vận tốc × dt → displacement → tích lũy thành tọa độ

    Kết quả:
    - GT trajectory trong hệ tọa độ cục bộ (local frame)
    - x = forward (hướng xe đi), y = left
    - shape: [num_poses, 3] (x, y, heading)

    Args:
        nav_samples: Danh sách dữ liệu NAV đã sort theo timestamp
        curr_ts_ns: Timestamp hiện tại (nanoseconds)
        future_image_paths: Danh sách path ảnh tương lai (để lấy timestamp chính xác)
        num_poses: Số waypoints GT cần tính (default 8 = 4s / 0.5s)
        interval_s: Khoảng cách thời gian giữa các waypoints (default 0.5s)

    Returns:
        np.ndarray shape [num_poses, 3]: GT trajectory (x_forward, y_left, heading)
    """
    if not nav_samples:
        return np.zeros((num_poses, 3), dtype=np.float32)

    # Lấy heading hiện tại từ vận tốc
    curr_nav = find_closest_nav_sample(nav_samples, curr_ts_ns)
    if curr_nav is None:
        return np.zeros((num_poses, 3), dtype=np.float32)

    # heading_0 = hướng di chuyển hiện tại (dùng để chuyển global → local)
    speed_curr = np.hypot(curr_nav.ve, curr_nav.vn)
    if speed_curr > 0.1:
        heading_0 = np.arctan2(curr_nav.ve, curr_nav.vn)  # heading relative to North
    else:
        heading_0 = 0.0  # Xe đang đứng yên → giả sử hướng Bắc

    cos_h = np.cos(heading_0)
    sin_h = np.sin(heading_0)

    # Tích phân vận tốc trong hệ global (East/North), rồi xoay về local
    east_acc = 0.0   # Displacement tích lũy hướng Đông
    north_acc = 0.0  # Displacement tích lũy hướng Bắc
    poses = []

    prev_ts = curr_ts_ns

    for pose_idx in range(1, num_poses + 1):
        # Thời điểm mục tiêu
        target_ts = curr_ts_ns + int(pose_idx * interval_s * 1e9)

        # Lấy vận tốc gần nhất tại thời điểm mục tiêu
        nav = find_closest_nav_sample(nav_samples, target_ts)

        if nav is not None:
            # dt thực tế giữa 2 mốc
            dt = interval_s
            east_acc += nav.ve * dt
            north_acc += nav.vn * dt

        # Chuyển từ (East, North) sang (Forward, Left) bằng rotation -heading_0
        # Forward = component dọc theo heading_0
        # Left = component vuông góc với heading_0
        x_fwd = east_acc * sin_h + north_acc * cos_h
        y_left = -east_acc * cos_h + north_acc * sin_h

        # Heading cục bộ
        if nav and np.hypot(nav.ve, nav.vn) > 0.1:
            heading_local = np.arctan2(nav.ve, nav.vn) - heading_0
        else:
            heading_local = 0.0

        poses.append([x_fwd, y_left, heading_local])

    return np.array(poses, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT INPUT
# ═══════════════════════════════════════════════════════════════════════════════

def make_cameras_from_front(front_rgb: np.ndarray) -> Cameras:
    return Cameras(
        cam_f0=Camera(image=front_rgb),
        cam_l0=Camera(), cam_l1=Camera(), cam_l2=Camera(),
        cam_r0=Camera(), cam_r1=Camera(), cam_r2=Camera(),
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
            ego_velocity=velocity, ego_acceleration=acceleration,
            driving_command=driving_command, in_global_frame=False,
        )

    return AgentInput(
        ego_statuses=[ego_status_from_nav(prev_nav), ego_status_from_nav(curr_nav)],
        cameras=[make_cameras_from_front(prev_rgb), make_cameras_from_front(curr_rgb)],
        lidars=[Lidar(), Lidar()],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: BEV WITH GT + PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def draw_trajectory_bev_dual(
    pred_xyh: np.ndarray,
    gt_xyh: Optional[np.ndarray],
    width: int,
    height: int,
    scale_px_per_meter: float,
) -> np.ndarray:
    """
    Vẽ BEV canvas chồng 2 quỹ đạo:
    - GT: Curve ĐỎ + chấm tròn đỏ
    - Prediction: Curve XANH + chấm tròn xanh
    """
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    origin_x = width // 2
    origin_y = int(height * 0.9)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Grid lines
    for x_m in np.arange(0, 120, 2.5):
        py = int(origin_y - x_m * scale_px_per_meter)
        if py < 0:
            break
        is_major = (x_m % 10 == 0)
        color = (180, 180, 180) if is_major else (235, 235, 235)
        thickness = 2 if is_major else 1
        cv2.line(canvas, (0, py), (width, py), color, thickness)
        if x_m != 0 and is_major:
            cv2.putText(canvas, f"{int(x_m)}m", (origin_x + 5, py - 5), font, 0.5, (100, 100, 100), 1)

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

    # Axes
    cv2.line(canvas, (origin_x, 0), (origin_x, height), (150, 150, 150), 2)
    cv2.line(canvas, (0, origin_y), (width, origin_y), (150, 150, 150), 2)
    cv2.circle(canvas, (origin_x, origin_y), 6, (0, 0, 0), -1)

    def xyh_to_px(xyh: np.ndarray) -> List[Tuple[int, int]]:
        pts = []
        for x_m, y_m, _ in xyh:
            px = int(origin_x - y_m * scale_px_per_meter)
            py = int(origin_y - x_m * scale_px_per_meter)
            pts.append((px, py))
        return pts

    # ── GT (ĐỎ) ──
    if gt_xyh is not None and len(gt_xyh) > 0:
        gt_pts = xyh_to_px(gt_xyh)
        for i in range(1, len(gt_pts)):
            cv2.line(canvas, gt_pts[i - 1], gt_pts[i], (0, 0, 255), 3)  # BGR: đỏ
        for p in gt_pts:
            cv2.circle(canvas, p, 4, (0, 0, 200), -1)

    # ── Prediction (XANH DƯƠNG) ──
    pred_pts = xyh_to_px(pred_xyh)
    for i in range(1, len(pred_pts)):
        cv2.line(canvas, pred_pts[i - 1], pred_pts[i], (255, 100, 0), 3)  # BGR: xanh dương
    for p in pred_pts:
        cv2.circle(canvas, p, 4, (255, 80, 0), -1)

    # ── Legend ──
    legend_y = 30
    cv2.putText(canvas, "BEV: GT vs Prediction", (20, legend_y), font, 0.8, (40, 40, 40), 2)
    # GT legend
    cv2.line(canvas, (20, legend_y + 25), (60, legend_y + 25), (0, 0, 255), 3)
    cv2.putText(canvas, "GT (Ground Truth)", (70, legend_y + 30), font, 0.6, (0, 0, 200), 2)
    # Prediction legend
    cv2.line(canvas, (20, legend_y + 50), (60, legend_y + 50), (255, 100, 0), 3)
    cv2.putText(canvas, "Prediction (Model)", (70, legend_y + 55), font, 0.6, (200, 80, 0), 2)
    # Scale info
    cv2.putText(canvas, "x: forward (m), y: left (m)", (20, legend_y + 80), font, 0.5, (80, 80, 80), 1)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GT vs Prediction Overlay — VF Qualitative Eval")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--camera-dir", type=Path, required=True)
    parser.add_argument("--nav-dir", type=Path, default=None)
    parser.add_argument("--steer-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/vf_gt_vs_pred"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--sample-gap-seconds", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scale", type=float, default=40.0)
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
        raise RuntimeError(f"{label} is HTML, not a PyTorch artifact: {path}")


def select_frame_indices(image_paths: List[Path], frame_stride: int, sample_gap_seconds: float) -> List[int]:
    if not image_paths:
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = make_parser().parse_args()

    # ── Validate ──
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

    # ── Load NAV & STEER ──
    nav_samples: List[NavSample] = []
    if args.nav_dir and args.nav_dir.exists():
        nav_samples = load_nav_samples(args.nav_dir)
        print(f"[INFO] Loaded {len(nav_samples)} NAV samples")
    else:
        print("[WARN] NAV directory missing → GT trajectory sẽ bằng 0 (đường thẳng)")

    steer_samples: List[SteerSample] = []
    if args.steer_dir and args.steer_dir.exists():
        steer_samples = load_steer_samples(args.steer_dir)
        print(f"[INFO] Loaded {len(steer_samples)} STEER samples")

    # ── Build agent ──
    agent_cfg = OmegaConf.load(args.agent_config)
    OmegaConf.update(agent_cfg, "checkpoint_path", str(args.checkpoint_path))
    OmegaConf.update(agent_cfg, "pretrain_pt_path", str(args.encoder_path))
    agent: DriveJEPAAgent = instantiate(agent_cfg)
    traj_sampling = agent._trajectory_sampling
    print(f"[INFO] Agent: double_image={agent._double_image}, horizon={traj_sampling.time_horizon}s, num_poses={traj_sampling.num_poses}")

    agent.initialize()
    print(f"[INFO] Agent initialized.")

    # ── Select frame pairs ──
    selected_indices = select_frame_indices(image_paths, args.frame_stride, args.sample_gap_seconds)
    if len(selected_indices) < 2:
        raise RuntimeError("Not enough selected frames.")

    pair_indices = [(selected_indices[i - 1], selected_indices[i]) for i in range(1, len(selected_indices))]
    if args.max_frames > 0 and len(pair_indices) > args.max_frames:
        seed = args.seed if args.seed is not None else random.randint(0, 2**31)
        rng = random.Random(seed)
        start = rng.randint(0, len(pair_indices) - args.max_frames)
        pair_indices = pair_indices[start:start + args.max_frames]
        print(f"[INFO] Random window: seed={seed}, start={start}, pairs={len(pair_indices)}")

    print(f"[INFO] Processing {len(pair_indices)} frame pairs")

    # ── Inference + Render ──
    video_writer: Optional[cv2.VideoWriter] = None
    records = []

    for out_idx, (prev_idx, curr_idx) in enumerate(pair_indices):
        prev_path = image_paths[prev_idx]
        curr_path = image_paths[curr_idx]
        curr_ts = parse_timestamp_ns(curr_path.stem)

        # ── Prediction trajectory ──
        agent_input = build_agent_input(prev_path, curr_path, nav_samples)
        trajectory = agent.compute_trajectory(agent_input)
        pred_poses = trajectory.poses.astype(np.float32)

        # ── GT trajectory (Dead Reckoning) ──
        # Lấy danh sách future frames để tính GT
        future_paths = image_paths[curr_idx + 1:curr_idx + 1 + traj_sampling.num_poses * 5]
        gt_poses = compute_gt_trajectory(
            nav_samples=nav_samples,
            curr_ts_ns=curr_ts,
            future_image_paths=future_paths,
            num_poses=traj_sampling.num_poses,
            interval_s=0.5,
        )

        # ── Read camera image ──
        curr_bgr = cv2.imread(str(curr_path))
        if curr_bgr is None:
            continue

        # ── Telemetry HUD on camera image ──
        curr_nav = find_closest_nav_sample(nav_samples, curr_ts)
        curr_steer = find_closest_steer_sample(steer_samples, curr_ts)
        speed_mps = np.hypot(curr_nav.ve, curr_nav.vn) if curr_nav else 0.0
        speed_kph = speed_mps * 3.6
        acc = np.hypot(curr_nav.acc_x, curr_nav.acc_y) if curr_nav else 0.0
        steer_angle = curr_steer.steer_angle if curr_steer else 0.0

        # Semi-transparent HUD background
        hud_x, hud_y = 30, 40
        overlay = curr_bgr.copy()
        cv2.rectangle(overlay, (hud_x - 10, hud_y - 30), (hud_x + 350, hud_y + 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, curr_bgr, 0.5, 0, curr_bgr)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(curr_bgr, "Telemetry GT", (hud_x, hud_y), font, 0.8, (0, 255, 255), 2)
        cv2.putText(curr_bgr, f"Speed: {speed_kph:.1f} km/h ({speed_mps:.1f} m/s)", (hud_x, hud_y + 35), font, 0.7, (0, 255, 0), 2)
        cv2.putText(curr_bgr, f"Accel: {acc:.2f} m/s^2", (hud_x, hud_y + 70), font, 0.7, (0, 255, 0), 2)
        cv2.putText(curr_bgr, f"Steer: {steer_angle:.2f} deg", (hud_x, hud_y + 105), font, 0.7, (0, 255, 0), 2)
        cv2.putText(curr_bgr, f"TS: {curr_path.stem}", (hud_x, curr_bgr.shape[0] - 20), font, 0.5, (255, 255, 255), 1)

        # ── Build BEV canvas with both trajectories ──
        h, w = curr_bgr.shape[:2]
        bev = draw_trajectory_bev_dual(
            pred_xyh=pred_poses,
            gt_xyh=gt_poses,
            width=w,
            height=h,
            scale_px_per_meter=args.scale,
        )

        # Combine camera + BEV side by side
        canvas = np.concatenate([curr_bgr, bev], axis=1)

        # Prediction horizon label
        cv2.putText(canvas, f"pred_horizon={traj_sampling.time_horizon:.1f}s",
                    (w + 20, h - 30), font, 0.7, (20, 180, 20), 2)

        # ── Compute per-frame error metrics ──
        if gt_poses is not None and np.any(gt_poses):
            # ADE = Average Displacement Error
            disp = np.sqrt(np.sum((pred_poses[:, :2] - gt_poses[:, :2])**2, axis=1))
            ade = float(np.mean(disp))
            fde = float(disp[-1])  # Final Displacement Error
            cv2.putText(canvas, f"ADE: {ade:.2f}m | FDE: {fde:.2f}m",
                        (w + 20, h - 60), font, 0.6, (0, 0, 180), 2)

        # Save frame
        frame_out_path = output_frames_dir / f"{out_idx:06d}.jpg"
        cv2.imwrite(str(frame_out_path), canvas)

        # Init video writer
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = args.output_dir / "gt_vs_prediction.mp4"
            video_writer = cv2.VideoWriter(str(video_path), fourcc, float(args.fps), (canvas.shape[1], canvas.shape[0]))
        video_writer.write(canvas)

        # Record for JSON
        records.append({
            "frame": curr_path.stem,
            "prev_frame": prev_path.stem,
            "prediction_xyh": pred_poses.tolist(),
            "gt_xyh": gt_poses.tolist(),
            "telemetry": {
                "speed_mps": float(speed_mps),
                "acc_mps2": float(acc),
                "steer_angle_deg": float(steer_angle),
            },
            "metrics": {
                "ade_m": float(np.mean(np.sqrt(np.sum((pred_poses[:, :2] - gt_poses[:, :2])**2, axis=1)))),
                "fde_m": float(np.sqrt(np.sum((pred_poses[-1, :2] - gt_poses[-1, :2])**2))),
            } if np.any(gt_poses) else {},
        })

        if (out_idx + 1) % 20 == 0:
            print(f"[INFO] Processed {out_idx + 1}/{len(pair_indices)} pairs")

    if video_writer is not None:
        video_writer.release()

    # Save trajectories JSON
    traj_path = args.output_dir / "gt_vs_prediction_trajectories.json"
    with traj_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Summary statistics
    ade_list = [r["metrics"]["ade_m"] for r in records if r.get("metrics")]
    fde_list = [r["metrics"]["fde_m"] for r in records if r.get("metrics")]
    if ade_list:
        print(f"\n{'='*50}")
        print(f"SUMMARY — GT vs Prediction")
        print(f"{'='*50}")
        print(f"Total frames:      {len(records)}")
        print(f"Mean ADE:          {np.mean(ade_list):.3f} m")
        print(f"Mean FDE:          {np.mean(fde_list):.3f} m")
        print(f"Max ADE:           {np.max(ade_list):.3f} m")
        print(f"Max FDE:           {np.max(fde_list):.3f} m")

    print(f"\n[DONE] Output:")
    print(f"  Video:        {args.output_dir / 'gt_vs_prediction.mp4'}")
    print(f"  Frames:       {output_frames_dir}")
    print(f"  Trajectories: {traj_path}")


if __name__ == "__main__":
    main()
