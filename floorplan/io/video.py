"""Video-tier loader: one handheld walkthrough clip. Rooms are delimited by the protocol's
2-second palm cover (near-black frames). Frames are sampled at ~2 fps at ≤ 1024 px."""
from __future__ import annotations

import os

import cv2
import numpy as np

from floorplan.core.types import Frame
from floorplan.io.photos import MAX_SIDE, intrinsics_from_f35

VID_EXT = {".mov", ".mp4", ".m4v"}
# iPhone video: 26 mm-equivalent main lens with ~8 % stabilisation crop → effective ~28 mm.
DEFAULT_VIDEO_F35 = 28.0
SAMPLE_FPS = 2.0
DARK_MEAN = 22.0   # mean intensity below this = lens covered
DARK_MIN_S = 0.8   # covered for at least this long


def is_video_capture(capture_dir: str) -> bool:
    return any(os.path.splitext(f)[1].lower() in VID_EXT for f in os.listdir(capture_dir))


def _clip_path(capture_dir: str) -> str:
    vids = sorted(f for f in os.listdir(capture_dir) if os.path.splitext(f)[1].lower() in VID_EXT)
    return os.path.join(capture_dir, vids[0])


def load_video_capture(capture_dir: str, room_names: list[str] | None = None) -> dict[str, list[Frame]]:
    path = _clip_path(capture_dir)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / SAMPLE_FPS)))
    # pass 1: brightness per frame (cheap: every 3rd frame, small)
    means = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % 3 == 0:
            ok, fr = cap.retrieve()
            if ok:
                small = cv2.resize(fr, (64, 36))
                means.append((idx, float(small.mean())))
        idx += 1
    # segments: runs of dark frames ≥ DARK_MIN_S split rooms
    dark = [(i, m < DARK_MEAN) for i, m in means]
    cuts = []
    run_start = None
    for i, d in dark:
        if d and run_start is None:
            run_start = i
        if not d and run_start is not None:
            if (i - run_start) / fps >= DARK_MIN_S:
                cuts.append((run_start, i))
            run_start = None
    bounds = []
    prev = 0
    for a, b in cuts:
        bounds.append((prev, a))
        prev = b
    bounds.append((prev, idx))
    bounds = [(a, b) for a, b in bounds if (b - a) / fps >= 3.0]  # ignore stubs < 3 s
    # pass 2: sample frames per segment
    rooms: dict[str, list[Frame]] = {}
    cap.release()
    cap = cv2.VideoCapture(path)
    for si, (a, b) in enumerate(bounds):
        key = (room_names[si] if room_names and si < len(room_names) else f"{si + 1:02d}_space")
        frames = []
        for fi in range(a, b, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            h, w = fr.shape[:2]
            s = min(1.0, MAX_SIDE / max(w, h))
            if s < 1.0:
                fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            K = intrinsics_from_f35(DEFAULT_VIDEO_F35, rgb.shape[1], rgb.shape[0])
            frames.append(Frame(path=f"{path}#{fi}", rgb=rgb, K=K, t=fi / fps, room_key=key, role="corner"))
        if frames:
            frames[0].role = "entry"
            if len(frames) > 1:
                frames[-1].role = "exit"
            rooms[key] = frames
    cap.release()
    return rooms
