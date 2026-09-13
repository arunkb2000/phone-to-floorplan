"""Video tier: one handheld walkthrough clip, no depth, no poses.

Rooms are cut out of the clip two ways. If the walker followed the protocol and covered the lens
between rooms, the near-black runs are the cuts and they are exact. If not (a clip recorded by some
other app, say), we fall back to change detection on the per-frame wall signature: when the set of
wall distances stops being consistent with the room we have been accumulating, a new room starts.
"""
from __future__ import annotations

import os

import numpy as np

from floorplan.core.types import Frame
from floorplan.io.video import load_video_capture, sample_video_frames
from floorplan.tiers.mono import chain_stitch, rooms_from_grouped_frames


def _signature(fr: Frame, tier: str, seed: int):
    from floorplan.geometry.room import frame_geometry
    g = frame_geometry(fr, tier, seed=seed)
    if g is None:
        return None
    return g


def segment_by_signature(frames: list[Frame], log: dict, min_len: int = 4) -> list[list[Frame]]:
    """Cut the clip where the observed room stops matching. Cheap proxy for a doorway transit:
    the running estimate of the room's two extents jumps and does not come back."""
    from floorplan.geometry.depth import predict_depth
    from floorplan.geometry.room import frame_geometry
    sigs = []
    for i, fr in enumerate(frames):
        if fr.depth is None:
            fr.depth = predict_depth(fr.rgb, size=log.get("_depth_size", "small"), device=log.get("_device", "auto"),
                                     use_cache=log.get("_use_cache", True))
            fr.depth_source = "monocular"
        g = frame_geometry(fr, "video", seed=i)
        sigs.append(None if g is None else sorted(round(w["dist"], 2) for w in g.walls.values()))
    groups, cur = [], []
    ref = None
    for fr, s in zip(frames, sigs):
        if s is None or ref is None:
            cur.append(fr)
            if s is not None and ref is None:
                ref = s
            continue
        near = sum(1 for v in s if any(abs(v - r) < 0.35 for r in ref))
        if near == 0 and len(cur) >= min_len:
            groups.append(cur)
            cur, ref = [fr], s
        else:
            cur.append(fr)
            ref = s if len(s) >= len(ref) else ref
    if cur:
        groups.append(cur)
    return [g for g in groups if len(g) >= min_len] or [frames]


def run_video(capture_dir: str, *, depth_size: str, device: str, use_cache: bool, log: dict):
    groups, how = load_video_capture(capture_dir)
    log["room_cuts"] = how
    if how == "none":
        frames = sample_video_frames(capture_dir)
        log["_depth_size"], log["_device"], log["_use_cache"] = depth_size, device, use_cache
        segs = segment_by_signature(frames, log)
        groups = {f"{i + 1:02d}_space": s for i, s in enumerate(segs)}
        log["room_cuts"] = "wall-signature change detection (no palm-cover cuts found)"
    log["n_frames"] = sum(len(v) for v in groups.values())
    for k in ("_depth_size", "_device", "_use_cache"):
        log.pop(k, None)
    rooms = rooms_from_grouped_frames(groups, "video", depth_size=depth_size, device=device,
                                      use_cache=use_cache, log=log)
    return rooms, chain_stitch(rooms)
