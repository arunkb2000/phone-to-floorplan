from __future__ import annotations

import os
import time

import numpy as np

from floorplan.core.types import RoomEstimate
from floorplan.geometry.frame import rotz
from floorplan.geometry.lidar import _global_yaw, correct_drift, frame_geom_world, segment_rooms
from floorplan.geometry.room import fuse_room
from floorplan.io.photos import label_from_key
from floorplan.io.stray import load_stray_capture


def run_lidar(capture_dir: str, *, drift_correction: bool, log: dict, every: int = 2):
    t0 = time.time()
    frames = load_stray_capture(capture_dir, every=every)
    log["n_frames"] = len(frames)
    log["load_s"] = round(time.time() - t0, 2)
    yaw = _global_yaw(frames)
    drift = {"correction": "on" if drift_correction else "off"}
    t0 = time.time()
    if drift_correction:
        drift.update(correct_drift(frames, yaw))
    else:
        drift["method"] = "poses used as-is (ablation)"
    log["drift_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    geoms = [frame_geom_world(fr, yaw, seed=i) for i, fr in enumerate(frames)]
    Rm = rotz(-yaw)
    cams = np.array([(Rm @ fr.pose[:3, 3])[:2] for fr in frames])
    room_idx, n_rooms, transitions = segment_rooms(frames, geoms, cams)
    names = None
    p = os.path.join(capture_dir, "rooms.txt")
    if os.path.exists(p):
        names = [l.strip() for l in open(p) if l.strip()]
    rooms: list[RoomEstimate] = []
    for r in range(n_rooms):
        idx = [i for i in range(len(frames)) if room_idx[i] == r]
        key = names[r] if names and r < len(names) else f"{r + 1:02d}_space"
        fr_r = [frames[i] for i in idx]
        g_r = [geoms[i] for i in idx]
        c_r = [cams[i] for i in idx]
        room = fuse_room(fr_r, g_r, key, label_from_key(key), "lidar", cam_xy=c_r)
        room._frames, room._geoms = fr_r, g_r
        rooms.append(room)
    log["geometry_s"] = round(time.time() - t0, 2)
    log["transitions"] = transitions
    return rooms, drift
