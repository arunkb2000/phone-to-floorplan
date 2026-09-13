from __future__ import annotations

import os

from floorplan.io.video import load_video_capture
from floorplan.tiers.common import rooms_from_frames


def run_video(capture_dir: str, *, depth_size: str, device: str, use_cache: bool, log: dict):
    names = None
    p = os.path.join(capture_dir, "rooms.txt")
    if os.path.exists(p):
        names = [l.strip() for l in open(p) if l.strip()]
    rooms = load_video_capture(capture_dir, names)
    log["n_rooms"] = len(rooms)
    log["n_frames"] = sum(len(v) for v in rooms.values())
    return rooms_from_frames(rooms, "video", depth_size=depth_size, device=device, use_cache=use_cache, log=log)
