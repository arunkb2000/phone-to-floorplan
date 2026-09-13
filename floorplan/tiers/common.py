"""Shared driver pieces for the no-pose tiers (photo, video)."""
from __future__ import annotations

import time

import numpy as np

from floorplan.core.types import Frame, RoomEstimate
from floorplan.geometry.depth import predict_depth
from floorplan.geometry.room import frame_geometry, fuse_room, orient_frames
from floorplan.io.photos import label_from_key


def exposure_score(rgb: np.ndarray) -> float:
    g = rgb.mean(-1)
    dark = float((g < 25).mean()); bright = float((g > 235).mean())
    return float(max(0.0, 1.0 - 2.0 * dark - 2.0 * bright))


def rooms_from_frames(rooms: dict[str, list[Frame]], tier: str, *, depth_size: str, device: str,
                      use_cache: bool, log: dict) -> list[RoomEstimate]:
    t0 = time.time()
    for key, frames in rooms.items():
        for fr in frames:
            if fr.depth is None:
                fr.depth = predict_depth(fr.rgb, size=depth_size, device=device, use_cache=use_cache)
                fr.depth_source = "monocular"
    log["depth_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    out = []
    for key, frames in rooms.items():
        geoms = [frame_geometry(fr, tier, seed=i) for i, fr in enumerate(frames)]
        geoms = orient_frames(frames, geoms, tier)
        room = fuse_room(frames, geoms, key, label_from_key(key), tier)
        room.exposure = float(np.mean([exposure_score(fr.rgb) for fr in frames]))
        if room.exposure < 0.6:
            room.warnings.append("low light / clipped exposure: intervals widened")
            for m in (room.length_x, room.length_y, room.ceiling):
                m.sigma *= 1.5
        room._frames = frames
        room._geoms = geoms
        out.append(room)
    log["geometry_s"] = round(time.time() - t0, 2)
    return out
