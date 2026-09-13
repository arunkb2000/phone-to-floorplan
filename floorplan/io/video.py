"""Video-tier loader: one handheld walkthrough clip.

The protocol asks the walker to cover the lens for two seconds while stepping through a doorway,
which puts an unambiguous, free marker in the stream. When those markers are present the room cuts
are exact; when they are not (a clip shot by some other app) the tier falls back to change detection
on the geometry, and says so in the output.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from floorplan.core.types import Frame
from floorplan.io.photos import MAX_SIDE, intrinsics_from_f35

VID_EXT = {".mov", ".mp4", ".m4v"}
DEFAULT_VIDEO_F35 = 28.0    # iPhone main lens at 1x, allowing for the stabilisation crop
SAMPLE_FPS = 2.0
DARK_MEAN = 22.0
DARK_MIN_S = 0.8


def clip_path(capture_dir: str) -> str:
    if os.path.isfile(capture_dir):
        return capture_dir
    vids = sorted(f for f in os.listdir(capture_dir) if os.path.splitext(f)[1].lower() in VID_EXT)
    if not vids:
        raise ValueError(f"no video file in {capture_dir}")
    return os.path.join(capture_dir, vids[0])


def _grab(cap, fi: int, max_side: int = MAX_SIDE):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, bgr = cap.read()
    if not ok:
        return None
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _brightness_track(path: str, every: int = 5):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx, means = 0, []
    while True:
        if not cap.grab():
            break
        if idx % every == 0:
            ok, fr = cap.retrieve()
            if ok:
                means.append((idx, float(cv2.resize(fr, (48, 27)).mean())))
        idx += 1
    cap.release()
    return fps, idx, means


def dark_cuts(path: str):
    fps, n, means = _brightness_track(path)
    cuts, run = [], None
    for i, m in means:
        if m < DARK_MEAN and run is None:
            run = i
        elif m >= DARK_MEAN and run is not None:
            if (i - run) / fps >= DARK_MIN_S:
                cuts.append((run, i))
            run = None
    return fps, n, cuts


def sample_video_frames(capture_dir: str, a: int = 0, b: int | None = None, room_key: str = "01_space",
                        fps_target: float = SAMPLE_FPS, max_frames: int = 40) -> list[Frame]:
    path = clip_path(capture_dir)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    b = n if b is None else b
    step = max(1, int(round(fps / fps_target)))
    idxs = list(range(a, b, step))
    if len(idxs) > max_frames:
        idxs = [idxs[i] for i in np.linspace(0, len(idxs) - 1, max_frames).round().astype(int)]
    frames = []
    for j, fi in enumerate(idxs):
        rgb = _grab(cap, fi)
        if rgb is None:
            continue
        K = intrinsics_from_f35(DEFAULT_VIDEO_F35, rgb.shape[1], rgb.shape[0])
        role = "entry" if j == 0 else ("exit" if j == len(idxs) - 1 else "corner")
        frames.append(Frame(path=f"{path}#{fi}", rgb=rgb, K=K, t=fi / fps, room_key=room_key, role=role))
    cap.release()
    return frames


def is_video_capture(capture_dir: str) -> bool:
    return os.path.isdir(capture_dir) and any(os.path.splitext(f)[1].lower() in VID_EXT for f in os.listdir(capture_dir))


def load_video_capture(capture_dir: str, room_names: list[str] | None = None):
    """Returns ({room_key: [Frame]}, how) where `how` says which cut method was used."""
    path = clip_path(capture_dir)
    if room_names is None:
        p = os.path.join(capture_dir if os.path.isdir(capture_dir) else os.path.dirname(capture_dir), "rooms.txt")
        if os.path.exists(p):
            room_names = [ln.strip() for ln in open(p) if ln.strip()]
    fps, n, cuts = dark_cuts(path)
    if not cuts:
        return {}, "none"
    bounds, prev = [], 0
    for a, b in cuts:
        bounds.append((prev, a))
        prev = b
    bounds.append((prev, n))
    bounds = [(a, b) for a, b in bounds if (b - a) / fps >= 3.0]
    groups = {}
    for i, (a, b) in enumerate(bounds):
        key = room_names[i] if room_names and i < len(room_names) else f"{i + 1:02d}_space"
        key = key if key not in groups else f"{key}#{i}"
        fr = sample_video_frames(path, a, b, room_key=key)
        if fr:
            groups[key] = fr
    return groups, f"palm-cover cuts ({len(cuts)} found)"
