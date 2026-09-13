"""LiDAR-tier loader for Stray Scanner exports (and our synthetic clone of that format).

Folder: camera_matrix.csv, odometry.csv (timestamp, frame, x, y, z, qx, qy, qz, qw; ARKit y-up world,
OpenGL camera), depth/NNNNNN.png (16-bit mm, 192x256), confidence/NNNNNN.png (0/1/2), rgb.mp4 or rgb/*.jpg.
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np
from PIL import Image

from floorplan.core.types import Frame

# ARKit world is y-up; we work z-up. Camera: OpenGL (x right, y up, z back) → OpenCV (x right, y down, z fwd).
_R_WORLD = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)   # y-up → z-up
_R_CAM = np.diag([1.0, -1.0, -1.0])                                             # GL cam → CV cam


def _quat_to_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def stray_pose_to_zup(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """ARKit (y-up world, GL camera) camera-to-world → z-up world, OpenCV camera, 4x4."""
    R = _quat_to_R(qx, qy, qz, qw)
    T = np.eye(4)
    T[:3, :3] = _R_WORLD @ R @ _R_CAM
    T[:3, 3] = _R_WORLD @ np.array([x, y, z], dtype=np.float64)
    return T


def is_stray_capture(capture_dir: str) -> bool:
    return os.path.exists(os.path.join(capture_dir, "odometry.csv")) and os.path.isdir(os.path.join(capture_dir, "depth"))


def load_stray_capture(capture_dir: str, every: int = 1, max_frames: int = 3000) -> list[Frame]:
    K_rgb = np.loadtxt(os.path.join(capture_dir, "camera_matrix.csv"), delimiter=",").reshape(3, 3)
    poses = {}
    with open(os.path.join(capture_dir, "odometry.csv")) as f:
        rd = csv.reader(f)
        header = next(rd)
        for row in rd:
            if not row or row[0].strip().startswith("#"):
                continue
            t, fi = float(row[0]), int(float(row[1]))
            x, y, z, qx, qy, qz, qw = map(float, row[2:9])
            poses[fi] = (t, stray_pose_to_zup(x, y, z, qx, qy, qz, qw))
    depth_files = sorted(f for f in os.listdir(os.path.join(capture_dir, "depth")) if f.endswith(".png"))
    rgb_dir = os.path.join(capture_dir, "rgb")
    mp4 = os.path.join(capture_dir, "rgb.mp4")
    cap = cv2.VideoCapture(mp4) if os.path.exists(mp4) and not os.path.isdir(rgb_dir) else None
    frames: list[Frame] = []
    for i, df in enumerate(depth_files):
        fi = int(os.path.splitext(df)[0])
        if fi not in poses or (i % every):
            continue
        depth = np.asarray(Image.open(os.path.join(capture_dir, "depth", df)), dtype=np.float32) / 1000.0
        cf = os.path.join(capture_dir, "confidence", df)
        conf = np.asarray(Image.open(cf), dtype=np.uint8) if os.path.exists(cf) else np.full(depth.shape, 2, np.uint8)
        h, w = depth.shape
        rgb = None
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if ok:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        elif os.path.isdir(rgb_dir):
            p = os.path.join(rgb_dir, f"{fi:06d}.jpg")
            if os.path.exists(p):
                rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        if rgb is None:
            rgb = np.zeros((h * 2, w * 2, 3), np.uint8)
        # K for the depth resolution (geometry works at depth res; rgb kept at its own res with K_rgb)
        K_d = K_rgb.copy()
        K_d[0] *= w / rgb.shape[1] if rgb.shape[1] else 1.0
        K_d[1] *= h / rgb.shape[0] if rgb.shape[0] else 1.0
        t, T = poses[fi]
        frames.append(Frame(path=os.path.join(capture_dir, "depth", df), rgb=rgb, K=K_d, depth=depth, conf=conf,
                            pose=T, t=t, depth_source="lidar"))
        if len(frames) >= max_frames:
            break
    return frames
