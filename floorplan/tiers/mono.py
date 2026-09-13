"""Photo and video tiers: RGB only, no depth, no poses.

Both tiers run the same core. A monocular metric-depth network turns each still into a depth map;
gravity comes from the floor and ceiling planes in that map, the Manhattan frame from the wall
normals, and the room's extent from frames that see two opposite walls at once. The tiers differ in
how the frames are grouped into rooms and in how many frames vote, which is why the video tier's
intervals are tighter.

Scale is the honest weak point. The network is metric, but a monocular metric prediction carries a
few per cent of scale error that no amount of averaging removes, so the per-tier interval floor in
floorplan/geometry/assemble.py never lets these tiers claim LiDAR-class precision.
"""
from __future__ import annotations

import time

import numpy as np

from floorplan.core.room import RoomOut
from floorplan.core.types import Frame, Measurement, Opening
from floorplan.geometry.depth import predict_depth
from floorplan.geometry.room import TIER_DEPTH_REL_ERR, frame_geometry, fuse_room, orient_frames
from floorplan.io.photos import label_from_key

CEILING_PRIOR = (2.70, 0.30)


def exposure_score(rgb: np.ndarray) -> float:
    g = rgb.mean(-1)
    dark, bright = float((g < 25).mean()), float((g > 235).mean())
    blur = 1.0
    try:
        import cv2
        lap = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        blur = float(np.clip(lap / 120.0, 0.0, 1.0))
    except Exception:
        pass
    return float(max(0.0, min(1.0, (1.0 - 2.0 * dark - 2.0 * bright) * (0.5 + 0.5 * blur))))


def rooms_from_grouped_frames(groups: dict[str, list[Frame]], tier: str, *, depth_size: str, device: str,
                              use_cache: bool, log: dict) -> list[RoomOut]:
    t0 = time.time()
    n_depth = 0
    for frames in groups.values():
        for fr in frames:
            if fr.depth is None:
                fr.depth = predict_depth(fr.rgb, size=depth_size, device=device, use_cache=use_cache)
                fr.depth_source = "monocular"
                n_depth += 1
    log["depth_s"] = round(time.time() - t0, 2)
    log["n_depth_inferences"] = n_depth

    t0 = time.time()
    out: list[RoomOut] = []
    for key, frames in groups.items():
        geoms = [frame_geometry(fr, tier, seed=i) for i, fr in enumerate(frames)]
        geoms = orient_frames(frames, geoms, tier)
        est = fuse_room(frames, geoms, key, label_from_key(key), tier)
        room = _to_roomout(est, tier)
        room.exposure = float(np.mean([exposure_score(fr.rgb) for fr in frames]))
        room.frames, room.geoms = frames, geoms
        if room.exposure < 0.55:
            room.warnings.append("low light or motion blur: every interval in this room is widened 1.5x")
            for m in room.wall_len + ([room.ceiling] if room.ceiling else []):
                m.sigma *= 1.5
        out.append(room)
    log["geometry_s"] = round(time.time() - t0, 2)
    log["n_rooms"] = len(out)
    return out


def _to_roomout(est, tier: str) -> RoomOut:
    """A fused rectangle, in its own frame: entrance on wall C at y = 0, wall A ahead at y = Ly."""
    Lx, Ly = est.length_x, est.length_y
    room = RoomOut(key=est.key, label=est.label)
    room.polygon = np.array([[0.0, 0.0], [Lx.value, 0.0], [Lx.value, Ly.value], [0.0, Ly.value]])
    room.wall_ids = ["C", "B", "A", "D"]
    room.wall_len = [Measurement(Lx.value, Lx.sigma, Lx.source, Lx.method, Lx.n_support),
                     Measurement(Ly.value, Ly.sigma, Ly.source, Ly.method, Ly.n_support),
                     Measurement(Lx.value, Lx.sigma, Lx.source, Lx.method, Lx.n_support),
                     Measurement(Ly.value, Ly.sigma, Ly.source, Ly.method, Ly.n_support)]
    idx = {"C": 0, "B": 1, "A": 2, "D": 3}
    for o in est.openings:
        k = idx.get(o.wall_id)
        if k is None:
            continue
        L = room.wall_len[k].value
        # our detector measures offsets clockwise from the wall's far corner; the JSON measures them
        # from the polygon ring's start point, so flip once, here.
        o2 = Opening(wall_id=o.wall_id, type=o.type,
                     offset=Measurement(max(0.0, L - (o.offset.value + o.width.value)), o.offset.sigma,
                                        o.offset.source, o.offset.method, o.offset.n_support),
                     width=o.width, height=o.height, sill=o.sill, confidence=o.confidence)
        room.openings.append((k, o2))
    room.ceiling = est.ceiling
    room.ceiling_spread = est.ceiling_spread
    a = Lx.value * Ly.value
    room.area = Measurement(a, float(np.hypot(Lx.sigma * Ly.value, Ly.sigma * Lx.value)),
                            "measured" if (Lx.source == "measured" and Ly.source == "measured") else "inferred",
                            "length x width of the fused rectangle", min(Lx.n_support, Ly.n_support))
    room.coverage = est.coverage
    room.n_frames = est.n_frames
    room.warnings = list(est.warnings)
    room.warnings.append("rectangular room model: this tier has no poses, so alcoves and bays are not resolved")
    return room


# --------------------------------------------------------------------------- stitching without poses

WALL_OUT = {"C": (0, -1), "B": (1, 0), "A": (0, 1), "D": (-1, 0)}


def _ring(room: RoomOut, k: int, tx: float, ty: float):
    p = np.asarray(room.polygon, float)
    c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
    R = np.array([[c, -s], [s, c]])
    return p @ R.T + np.array([tx, ty])


def _door_centre_local(room: RoomOut, edge_k: int, o: Opening) -> np.ndarray:
    p = np.asarray(room.polygon, float)
    a, b = p[edge_k], p[(edge_k + 1) % len(p)]
    d = b - a
    L = float(np.hypot(*d))
    if L < 1e-6:
        return (a + b) / 2
    return a + d * ((o.offset.value + 0.5 * o.width.value) / L)


def chain_stitch(rooms: list[RoomOut]) -> list[dict]:
    """Lay the rooms out in walk order: each room's entrance door meets the previous room's exit door,
    and the rooms sit on opposite sides of that wall. Placements that overlap are penalised, so a
    capture that ignores the protocol still produces a plan, just a low-confidence one."""
    from shapely.geometry import Polygon
    edges = []
    if not rooms:
        return edges
    placed = []
    for i, room in enumerate(rooms):
        if i == 0:
            room.placement_confidence = 1.0
            room._place = (0, 0.0, 0.0)
            placed.append(Polygon(_ring(room, 0, 0, 0)))
            continue
        prev = rooms[i - 1]
        pk, ptx, pty = prev._place
        entr = next((o for k, o in room.openings if room.wall_ids[k] == "C" and o.type in ("door", "passage")), None)
        ek = 0
        ec = _door_centre_local(room, ek, entr) if entr else np.array([room.wall_len[0].value / 2, 0.0])
        cands = [(k, o) for k, o in prev.openings if o.type in ("door", "passage") and prev.wall_ids[k] != "C"]
        options = [(prev.wall_ids[k], _door_centre_local(prev, k, o), o) for k, o in cands]
        for letter, k in (("A", 2), ("B", 1), ("D", 3)):
            if not any(w == letter for w, _, _ in options):
                p = np.asarray(prev.polygon, float)
                options.append((letter, (p[k] + p[(k + 1) % 4]) / 2, None))
        best = None
        for letter, pc_local, op in options:
            c, s = np.cos(pk * np.pi / 2), np.sin(pk * np.pi / 2)
            R = np.array([[c, -s], [s, c]])
            pc = R @ pc_local + np.array([ptx, pty])
            nout = R @ np.array(WALL_OUT[letter], float)
            # rotate the new room so its entrance wall C (outward -y) faces back at the previous room
            kk = next(q for q in range(4)
                      if np.allclose(np.array([[np.cos(q * np.pi / 2), -np.sin(q * np.pi / 2)],
                                               [np.sin(q * np.pi / 2), np.cos(q * np.pi / 2)]]) @ np.array([0.0, -1.0]),
                                     -nout, atol=1e-6))
            Rk = np.array([[np.cos(kk * np.pi / 2), -np.sin(kk * np.pi / 2)],
                           [np.sin(kk * np.pi / 2), np.cos(kk * np.pi / 2)]])
            t = pc - Rk @ ec
            poly = Polygon(_ring(room, kk, t[0], t[1]))
            ov = sum(poly.intersection(q).area for q in placed) / max(poly.area, 1e-6)
            score = 10 * ov + (0.0 if op is not None else 1.0) + (0.0 if op is None else (1 - op.confidence) * 0.5)
            if best is None or score < best[0]:
                best = (score, kk, float(t[0]), float(t[1]), letter, op, ov)
        _, kk, tx, ty, letter, op, ov = best
        room._place = (kk, tx, ty)
        room.placement_confidence = float(max(0.15, (0.85 if op is not None else 0.45) - 2 * ov))
        placed.append(Polygon(_ring(room, kk, tx, ty)))
        if ov > 0.03:
            room.warnings.append(f"placement overlaps an already-placed room by {ov:.0%}; adjacency is uncertain")
        edges.append({"room_a": prev.key, "room_b": room.key,
                      "via_opening_id": f"{prev.key}:{letter}:door" if op is not None else "",
                      "evidence": "visual_match" if op is not None else "protocol_order",
                      "confidence": round(room.placement_confidence, 2)})
    for room in rooms:
        k, tx, ty = room._place
        room.polygon = _ring(room, k, tx, ty)
    return edges
