"""LiDAR tier: Stray Scanner capture -> global cloud -> rooms -> dimensioned layout.

Drift handling is a first-class, ablatable stage (see floorplan/geometry/drift.py).
"""
from __future__ import annotations

import os
import time

import numpy as np

from floorplan.core.room import RoomOut
from floorplan.core.types import Measurement, Opening
from floorplan.geometry.cloud import build_cloud, build_grid, segment_rooms, wall_mask
from floorplan.geometry.drift import correct_drift
from floorplan.geometry.layout import (
    close_polygon,
    orient_edges,
    rectilinear,
    refine_edges,
    room_ceiling,
    room_contour,
    trim_edges,
    wall_openings,
)
from floorplan.io.photos import label_from_key
from floorplan.io.stray import load_stray_capture

CEILING_PRIOR = (2.70, 0.30)     # metres: used only when the ceiling was never observed


def run_lidar(capture_dir: str, *, drift_correction: bool, log: dict, target_fps: float = 4.0,
              max_frames: int = 1200, with_rgb: bool = False, room_seeds: str = "hmaxima",
              drift_datum: str = "robust"):
    t0 = time.time()
    frames = load_stray_capture(capture_dir, target_fps=target_fps, max_frames=max_frames, with_rgb=with_rgb)
    if not frames:
        raise ValueError(f"no frames loaded from {capture_dir}")
    log["n_frames"] = len(frames)
    log["load_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    drift = {"correction": "on" if drift_correction else "off"}
    if drift_correction:
        drift.update(correct_drift(frames, datum=drift_datum))
    else:
        drift["method"] = "poses used as-is (ablation only; not the shipped default)"
    log["drift_s"] = round(time.time() - t0, 2)

    t0 = time.time()
    cloud = build_cloud(frames)
    g = build_grid(cloud)
    lab = segment_rooms(g, seeds=room_seeds)
    log["room_seeds"] = room_seeds
    free = g.free & ~wall_mask(g)
    log["cloud_s"] = round(time.time() - t0, 2)
    log["n_points"] = int(len(cloud.P))
    log["yaw_deg"] = round(float(np.degrees(cloud.yaw)), 2)
    log["floor_z"] = round(float(cloud.floor_z), 4)
    log["ceil_z"] = None if not np.isfinite(cloud.ceil_z) else round(float(cloud.ceil_z), 4)

    names = None
    p = os.path.join(capture_dir, "rooms.txt")
    if os.path.exists(p):
        names = [ln.strip() for ln in open(p) if ln.strip()]

    t0 = time.time()
    rooms: list[RoomOut] = []
    for r in range(1, int(lab.max()) + 1):
        mask = lab == r
        poly = room_contour(g, mask, simplify_m=0.22)
        if poly is None or len(poly) < 4:
            continue
        edges = rectilinear(poly, min_edge=0.5)
        if len(edges) < 4:
            continue
        orient_edges(edges, poly.mean(0))
        refine_edges(edges, cloud)
        trim_edges(edges)
        ring = close_polygon(edges)
        key = names[r - 1] if names and r - 1 < len(names) else f"{r:02d}_space"
        room = RoomOut(key=key, label=label_from_key(key), polygon=ring)
        measured = 0.0
        for k, e in enumerate(edges):
            room.wall_ids.append(chr(ord("A") + k) if len(edges) <= 26 else f"W{k}")
            src = "measured" if e.refined else "inferred"
            e.sigma if e.refined else max(0.06, 0.02 * e.length)
            # a wall's length is set by where its two neighbours sit, so its uncertainty is theirs
            n0, n1 = edges[k - 1], edges[(k + 1) % len(edges)]
            lsig = float(np.hypot(n0.sigma if n0.refined else 0.06, n1.sigma if n1.refined else 0.06))
            room.wall_len.append(Measurement(e.length, max(lsig, 0.005), src,
                                             "rectilinear contour with edges snapped to the measured wall planes",
                                             e.n_support))
            if e.refined:
                measured += e.length
            for o in wall_openings(e, cloud, g, free):
                op = Opening(wall_id=room.wall_ids[k], type=o["type"],
                             offset=Measurement(abs(o["u0"] - min(e.u0, e.u1)), max(0.01, 1.5 * e.sigma), "measured",
                                                "half-maximum edge of the wall-material profile", o["support"]),
                             width=Measurement(o["width"], 0.012, "measured",
                                               "half-maximum edges of the gap in the wall-material profile", o["support"]),
                             confidence=float(min(0.95, 0.55 + 0.00002 * o["support"])))
                if np.isfinite(o["top"]):
                    op.height = Measurement(float(o["top"] - o["sill"]), 0.03, "measured", "vertical extent of the gap", o["support"])
                if o["type"] == "window":
                    op.sill = Measurement(float(o["sill"]), 0.03, "measured", "top of the material below the gap", o["support"])
                room.openings.append((k, op))
        total_len = sum(e.length for e in edges)
        room.coverage = float(measured / total_len) if total_len else 0.0
        ch = room_ceiling(cloud, g, mask)
        if ch is not None:
            room.ceiling = Measurement(ch[0], ch[1], "measured", "room-local ceiling plane minus floor plane", ch[2])
            room.ceiling_spread = ch[3]
        elif np.isfinite(cloud.ceil_z):
            room.ceiling = Measurement(float(cloud.ceil_z - cloud.floor_z), max(0.02, cloud.ceil_sigma * 3), "inferred",
                                       "property-wide ceiling plane; this room's ceiling was not directly observed",
                                       cloud.ceil_support)
            room.warnings.append("ceiling not observed inside this room; property-wide ceiling used, interval widened")
        else:
            room.ceiling = Measurement(CEILING_PRIOR[0], CEILING_PRIOR[1], "prior",
                                       "no ceiling returns anywhere in the capture; regional prior", 0)
            room.warnings.append("the capture never looked at a ceiling: height is a prior, not a measurement")
        a = room.polygon_area()
        rel = float(np.mean([m.sigma / max(m.value, 1e-6) for m in room.wall_len])) if room.wall_len else 0.05
        room.area = Measurement(a, max(0.01, a * 2 * rel), "measured" if room.coverage > 0.5 else "inferred",
                                "shoelace area of the dimensioned polygon", len(edges))
        room.n_frames = int(np.sum(lab[tuple(g.to_cell(cloud.cams[:, 0], cloud.cams[:, 1]))] == r)) if len(cloud.cams) else 0
        if room.coverage < 0.5:
            room.warnings.append(f"only {room.coverage:.0%} of this room's wall length was measured against a sensed plane")
        room._mask = mask
        rooms.append(room)
    log["geometry_s"] = round(time.time() - t0, 2)
    log["n_rooms"] = len(rooms)
    return rooms, drift, dict(cloud=cloud, grid=g, labels=lab, frames=frames)
