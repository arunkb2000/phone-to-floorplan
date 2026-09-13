"""Loader for Stray Scanner exports (App Store, free): the LiDAR tier's input format.

camera_matrix.csv holds RGB-resolution intrinsics, odometry.csv the per-frame camera-to-world pose,
depth/ and confidence/ the 256x192 maps, rgb.mp4 one frame per depth frame. RGB is decoded lazily,
because the geometry needs only depth and that is the difference between a 6 second and a 4 minute run.

The exported rotation is already in OpenCV camera axes, not ARKit's OpenGL ones; applying the
textbook flip breaks everything downstream. scripts/check_convention.py is the evidence.
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np
from PIL import Image

from floorplan.core.types import Frame

# ARKit world is y-up; we work z-up.
_R_WORLD = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)   # y-up -> z-up
# Stray Scanner writes the rotation already in OpenCV camera axes (x right, y down, z forward), not
# ARKit's OpenGL ones. Verified empirically on the sample captures: with the OpenGL flip the floor
# smears over 3 m of height; without it the vertical-normal points collapse into a single 2 cm bin
# 1.47 m below the ARKit origin, which is the phone's carry height. scripts/check_convention.py
# reproduces that comparison.
_R_CAM = np.eye(3)


def _quat_to_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def stray_pose_to_zup(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """ARKit (y-up world, GL camera) camera-to-world -> z-up world, OpenCV camera, as a 4x4."""
    T = np.eye(4)
    T[:3, :3] = _R_WORLD @ _quat_to_R(qx, qy, qz, qw) @ _R_CAM
    T[:3, 3] = _R_WORLD @ np.array([x, y, z], dtype=np.float64)
    return T


def is_stray_capture(capture_dir: str) -> bool:
    return (os.path.exists(os.path.join(capture_dir, "odometry.csv"))
            and os.path.isdir(os.path.join(capture_dir, "depth")))


def read_odometry(capture_dir: str) -> dict[int, tuple[float, np.ndarray]]:
    poses = {}
    with open(os.path.join(capture_dir, "odometry.csv")) as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if len(row) < 9 or row[0].strip().startswith("#"):
                continue
            t, fi = float(row[0]), int(float(row[1]))
            poses[fi] = (t, stray_pose_to_zup(*(float(v) for v in row[2:9])))
    return poses


class RGBSource:
    """Lazy access to rgb.mp4 (or an rgb/ folder of stills) by frame index."""

    def __init__(self, capture_dir: str):
        self.dir = os.path.join(capture_dir, "rgb")
        self.mp4 = os.path.join(capture_dir, "rgb.mp4")
        self._cap = None
        self._pos = -1

    @property
    def available(self) -> bool:
        return os.path.isdir(self.dir) or os.path.exists(self.mp4)

    def get(self, fi: int, max_side: int = 1280) -> np.ndarray | None:
        if os.path.isdir(self.dir):
            p = os.path.join(self.dir, f"{fi:06d}.jpg")
            if not os.path.exists(p):
                return None
            img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        else:
            if self._cap is None:
                if not os.path.exists(self.mp4):
                    return None
                self._cap = cv2.VideoCapture(self.mp4)
            if fi != self._pos + 1:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, bgr = self._cap.read()
            self._pos = fi
            if not ok:
                return None
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        return img


def load_stray_capture(capture_dir: str, every: int = 1, max_frames: int = 4000,
                       target_fps: float | None = None, with_rgb: bool = False) -> list[Frame]:
    """Frames with depth (metres), confidence, intrinsics at depth resolution and z-up poses.

    `target_fps` subsamples by timestamp, which is what you want on a 46 fps capture; `every`
    subsamples by index and is used when timestamps are unavailable.
    """
    K_rgb = np.loadtxt(os.path.join(capture_dir, "camera_matrix.csv"), delimiter=",").reshape(3, 3)
    poses = read_odometry(capture_dir)
    depth_dir = os.path.join(capture_dir, "depth")
    conf_dir = os.path.join(capture_dir, "confidence")
    depth_files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".png"))
    if not depth_files:
        return []
    probe = np.asarray(Image.open(os.path.join(depth_dir, depth_files[0])))
    dh, dw = probe.shape[:2]
    # the depth map is the RGB frame downscaled, so the intrinsics scale with it. The RGB width is
    # 2*cx to within a pixel on every Stray export, which is how we recover it without decoding video.
    rgb_w, rgb_h = 2.0 * K_rgb[0, 2], 2.0 * K_rgb[1, 2]
    K_d = K_rgb.copy()
    K_d[0] *= dw / rgb_w
    K_d[1] *= dh / rgb_h
    rgb = RGBSource(capture_dir) if with_rgb else None

    keep: list[str] = []
    if target_fps is not None and len(poses) > 1:
        ts = sorted(t for t, _ in poses.values())
        span = ts[-1] - ts[0]
        if span > 0:
            step = 1.0 / target_fps
            next_t, chosen = ts[0], set()
            for f in depth_files:
                fi = int(os.path.splitext(f)[0])
                if fi not in poses:
                    continue
                t = poses[fi][0]
                if t + 1e-9 >= next_t:
                    chosen.add(f)
                    next_t = t + step
            keep = [f for f in depth_files if f in chosen]
    if not keep:
        keep = [f for i, f in enumerate(depth_files) if i % max(1, every) == 0]
    if len(keep) > max_frames:
        idx = np.linspace(0, len(keep) - 1, max_frames).round().astype(int)
        keep = [keep[i] for i in sorted(set(idx.tolist()))]

    frames: list[Frame] = []
    for f in keep:
        fi = int(os.path.splitext(f)[0])
        if fi not in poses:
            continue
        depth = np.asarray(Image.open(os.path.join(depth_dir, f)), dtype=np.float32) / 1000.0
        cp = os.path.join(conf_dir, f)
        conf = np.asarray(Image.open(cp), dtype=np.uint8) if os.path.exists(cp) else np.full(depth.shape, 2, np.uint8)
        t, T = poses[fi]
        img = rgb.get(fi) if rgb is not None else None
        frames.append(Frame(path=f"{capture_dir}#{fi:06d}", rgb=img if img is not None else np.zeros((1, 1, 3), np.uint8),
                            K=K_d, depth=depth, conf=conf, pose=T, t=t, depth_source="lidar"))
        frames[-1].frame_index = fi
    return frames
