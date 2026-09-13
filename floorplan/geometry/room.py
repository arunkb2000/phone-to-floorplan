"""Fuse per-frame observations of one room into a rectangle with uncertainties.

Room frame (matches the protocol and the ground truth): standing in the entrance door facing in,
wall A is ahead (+y), B right (+x), C behind (-y, holds the entrance), D left (-x). Offsets along a
wall run clockwise seen from above: A left→right, B far→near, C right→left, D near→far.
"""
from __future__ import annotations

import numpy as np

from floorplan.core.types import Frame, Measurement, Opening, RoomEstimate
from floorplan.geometry.frame import (FrameGeom, analyse_frame, backproject, best_rotation, openings_from_uz,
                                      rotate_geom, view_dir_xy, wall_openings, wall_uz)

TIER_SIGMA_FLOOR = {"photo": 0.035, "video": 0.02, "lidar": 0.004}   # relative σ floor per wall length
TIER_DEPTH_REL_ERR = {"photo": 0.05, "video": 0.045, "lidar": 0.01}
STAND_OFF = 0.45          # corner photos: camera ≈ 0.45 m from both walls of the corner
DOOR_STAND_OFF = 0.10     # entry photo: camera ≈ 0.1 m inside wall C
CORNER_VIEWS = [(1, 1), (1, -1), (-1, -1), (-1, 1)]        # protocol: near-left, far-left, far-right, near-right
CORNER_POS = [(0, 0), (0, 1), (1, 1), (1, 0)]              # (x side, y side) 0 = min wall, 1 = max wall
LETTER = {"+x": "B", "-x": "D", "+y": "A", "-y": "C"}


def frame_geometry(fr: Frame, tier: str, seed: int = 0) -> FrameGeom | None:
    P = backproject(fr.depth, fr.K, stride=2 if fr.depth.shape[0] > 400 else 1)
    if fr.pose is not None:
        # LiDAR: gravity from the pose; express points in a z-up frame centred on the camera
        R = fr.pose[:3, :3]
        Pw = P @ R.T
        up_hint = R.T @ np.array([0, 0, 1.0])
        g = analyse_frame(P, up_hint=up_hint, seed=seed, depth_rel_err=TIER_DEPTH_REL_ERR[tier])
        return g
    return analyse_frame(P, up_hint=np.array([0, -1.0, 0]), seed=seed, depth_rel_err=TIER_DEPTH_REL_ERR[tier])


def orient_frames(frames: list[Frame], geoms: list[FrameGeom | None], tier: str) -> list[FrameGeom | None]:
    """Resolve each frame's 90° Manhattan ambiguity into the room frame."""
    out = [None] * len(frames)
    corner_i = 0
    prev_view = None
    for i, (fr, g) in enumerate(zip(frames, geoms)):
        if g is None:
            continue
        if fr.pose is not None:
            # absolute: rotate so the frame's Manhattan x matches the world Manhattan x (set by caller via yaw)
            out[i] = g
            continue
        if tier == "video":
            # continuity: keep the rotation closest to the previous frame's view direction
            if prev_view is None:
                k = best_rotation(g, (0, 1))
            else:
                k = best_rotation(g, prev_view)
            g2 = rotate_geom(g, k)
            prev_view = view_dir_xy(g2)
            out[i] = g2
            continue
        # photos: protocol roles
        if fr.role == "entry":
            k = best_rotation(g, (0, 1))
        elif fr.role == "corner":
            k = best_rotation(g, CORNER_VIEWS[corner_i % 4]); corner_i += 1
        else:  # exit: unknown view; keep the identity, resolved later by extents/doors
            k = 0
        out[i] = rotate_geom(g, k)
    return out


def _cam_position(g: FrameGeom, fr: Frame, corner_idx: int | None, Lx: float, Ly: float):
    """Camera (x, y) in room coords with σ, from visible walls, else from the protocol role."""
    cx = cy = np.nan; sx = sy = 9.0
    rel = g.depth_rel_err
    if "-x" in g.walls:
        cx, sx = g.walls["-x"]["dist"], rel * g.walls["-x"]["dist"] + 0.03
    elif "+x" in g.walls and np.isfinite(Lx):
        cx, sx = Lx - g.walls["+x"]["dist"], rel * g.walls["+x"]["dist"] + 0.06
    if "-y" in g.walls:
        cy, sy = g.walls["-y"]["dist"], rel * g.walls["-y"]["dist"] + 0.03
    elif "+y" in g.walls and np.isfinite(Ly):
        cy, sy = Ly - g.walls["+y"]["dist"], rel * g.walls["+y"]["dist"] + 0.06
    if fr.role == "entry":
        if not np.isfinite(cy):
            cy, sy = DOOR_STAND_OFF, 0.15
    elif fr.role == "corner" and corner_idx is not None and np.isfinite(Lx) and np.isfinite(Ly):
        xs, ys = CORNER_POS[corner_idx % 4]
        if not np.isfinite(cx):
            cx, sx = (STAND_OFF if xs == 0 else Lx - STAND_OFF), 0.3
        if not np.isfinite(cy):
            cy, sy = (STAND_OFF if ys == 0 else Ly - STAND_OFF), 0.3
    return cx, cy, sx, sy


def _fuse(vals, sigs, floor_abs=0.0):
    vals, sigs = np.asarray(vals, float), np.asarray(sigs, float)
    if len(vals) == 0:
        return np.nan, np.nan, 0
    w = 1.0 / np.maximum(sigs, 1e-3) ** 2
    order = np.argsort(vals)
    cw = np.cumsum(w[order]) / w.sum()
    med = vals[order][min(len(vals) - 1, np.searchsorted(cw, 0.5))]
    keep = np.abs(vals - med) <= 3 * np.maximum(sigs, 0.05)
    if not keep.any():
        keep[:] = True
    m = float(np.sum(w[keep] * vals[keep]) / np.sum(w[keep]))
    s = float(max(1.0 / np.sqrt(np.sum(w[keep])), floor_abs))
    return m, s, int(keep.sum())


def fuse_room(frames: list[Frame], geoms: list[FrameGeom | None], key: str, label: str, tier: str,
              cam_xy: list | None = None, expected: dict | None = None) -> RoomEstimate:
    """geoms must already be oriented into the room frame. cam_xy: optional known camera positions
    (LiDAR) as (x, y) per frame in the room frame (any origin; walls are then absolute)."""
    floor_rel = TIER_SIGMA_FLOOR[tier]
    # ceiling height
    ch_v, ch_s = [], []
    for g in geoms:
        if g is not None and np.isfinite(g.cam_height) and np.isfinite(g.ceil_above):
            h = g.cam_height + g.ceil_above
            ch_v.append(h); ch_s.append(g.depth_rel_err * h * 0.7 + 0.008)
    ch, chs, chn = _fuse(ch_v, ch_s)
    ceiling_method = "floor–ceiling plane distance"
    if not np.isfinite(ch):
        # same person, same hand height: combine camera height (frames seeing the floor) with the
        # ceiling clearance (frames seeing the ceiling); extra σ for the hand-height variation
        hs = [g.cam_height for g in geoms if g is not None and np.isfinite(g.cam_height)]
        cs = [g.ceil_above for g in geoms if g is not None and np.isfinite(g.ceil_above)]
        if hs and cs:
            ch = float(np.median(hs) + np.median(cs))
            chs = float(np.hypot(0.05, 0.03 * ch)); chn = min(len(hs), len(cs))
            ceiling_method = "median camera height + median ceiling clearance (different frames)"
            ch_v = [ch]
    chs = max(chs, floor_rel * 0.5 * (ch if np.isfinite(ch) else 2.7))
    spread = float(np.std(ch_v)) if len(ch_v) > 1 else 0.0

    # extents: pair observations (both walls of an axis in one frame) are tight and position-free
    pair = {0: [], 1: []}
    for g in geoms:
        if g is None:
            continue
        rel = g.depth_rel_err
        for axis, (pos, neg) in enumerate((("+x", "-x"), ("+y", "-y"))):
            wp, wn = g.walls.get(pos), g.walls.get(neg)
            if wp and wn:
                v = wp["dist"] + wn["dist"]
                pair[axis].append((v, np.hypot(rel * wp["dist"], rel * wn["dist"]) + 0.5 * (wp["rms"] + wn["rms"])))
    # absolute wall coordinates (camera position known from poses, or from roles)
    absw = {"+x": [], "-x": [], "+y": [], "-y": []}
    if cam_xy is not None:
        for g, c in zip(geoms, cam_xy):
            if g is None or c is None:
                continue
            for d, w in list(g.walls.items()):
                ax = 0 if d[1] == "x" else 1
                sgn = 1 if d[0] == "+" else -1
                coord = c[ax] + sgn * w["dist"]
                if expected is not None and d in expected and abs(coord - expected[d]) > 0.25:
                    del g.walls[d]          # a wall of another room seen through a door: not ours
                    continue
                absw[d].append((coord, g.depth_rel_err * w["dist"] + w["rms"] + 0.005))
        L = {}
        for axis, (pos, neg) in enumerate((("+x", "-x"), ("+y", "-y"))):
            p = _fuse([v for v, s in absw[pos]], [s for v, s in absw[pos]])
            n = _fuse([v for v, s in absw[neg]], [s for v, s in absw[neg]])
            if np.isfinite(p[0]) and np.isfinite(n[0]):
                pair[axis].append((p[0] - n[0], np.hypot(p[1], n[1])))
    else:
        # role-based single-wall estimates (looser)
        corner_i = 0
        for fr, g in zip(frames, geoms):
            if g is None:
                continue
            rel = g.depth_rel_err
            ci = None
            if fr.role == "corner":
                ci = corner_i; corner_i += 1
            for axis, (pos, neg) in enumerate((("+x", "-x"), ("+y", "-y"))):
                wp, wn = g.walls.get(pos), g.walls.get(neg)
                if (wp is None) == (wn is None):
                    continue
                w = wp or wn
                if fr.role == "entry" and axis == 1 and wp is not None:
                    pair[axis].append((w["dist"] + DOOR_STAND_OFF, rel * w["dist"] + 0.12))
                elif fr.role == "corner" and ci is not None:
                    xs, ys = CORNER_POS[ci % 4]
                    far_is_pos = (xs == 0) if axis == 0 else (ys == 0)
                    if (wp is not None) == far_is_pos:
                        pair[axis].append((w["dist"] + STAND_OFF, rel * w["dist"] + 0.25))
    lx, lxs, lxn = _fuse([v for v, s in pair[0]], [s for v, s in pair[0]])
    ly, lys, lyn = _fuse([v for v, s in pair[1]], [s for v, s in pair[1]])
    warnings = []
    src_x = "measured" if lxn else "prior"
    src_y = "measured" if lyn else "prior"
    if not np.isfinite(lx):
        lx, lxs = (ly if np.isfinite(ly) else 3.5), 0.9
        warnings.append("x extent never bounded by two walls; prior used")
    if not np.isfinite(ly):
        ly, lys = (lx if np.isfinite(lx) else 3.5), 0.9
        warnings.append("y extent never bounded by two walls; prior used")
    lxs = max(lxs, floor_rel * lx); lys = max(lys, floor_rel * ly)
    if not np.isfinite(ch):
        ch, chs = 2.7, 0.25
        warnings.append("ceiling never observed with the floor; height is a prior")
    n_walls_seen = len({d for g in geoms if g for d in g.walls})
    room = RoomEstimate(
        key=key, label=label,
        length_x=Measurement(lx, lxs, src_x, "wall-pair distances fused over frames", lxn),
        length_y=Measurement(ly, lys, src_y, "wall-pair distances fused over frames", lyn),
        ceiling=Measurement(ch, chs, "measured" if chn else "prior", ceiling_method, chn),
        n_frames=sum(g is not None for g in geoms), coverage=min(1.0, n_walls_seen / 4.0),
        warnings=warnings, ceiling_spread=spread,
    )
    # wall origin for absolute coordinates (LiDAR): x of wall D, y of wall C
    x0 = y0 = 0.0
    if cam_xy is not None:
        d = _fuse([v for v, s in absw["-x"]], [s for v, s in absw["-x"]])[0]
        c = _fuse([v for v, s in absw["-y"]], [s for v, s in absw["-y"]])[0]
        x0 = d if np.isfinite(d) else 0.0
        y0 = c if np.isfinite(c) else 0.0
    room.openings = _room_openings(frames, geoms, room, cam_xy, x0, y0)
    room.origin_xy = (x0, y0)
    return room


def _accumulated_openings(frames, geoms, room: RoomEstimate, cam_xy, x0, y0) -> list[Opening]:
    """LiDAR: accumulate (u, z) samples of every frame per wall in room coordinates, detect once."""
    Lx, Ly = room.length_x.value, room.length_y.value
    acc = {L: {k: [] for k in ("on_u", "on_z", "cross_u", "cross_z", "near_u", "near_z")} for L in "ABCD"}
    for i, (fr, g) in enumerate(zip(frames, geoms)):
        if g is None or g.P is None or cam_xy[i] is None:
            continue
        cx, cy = cam_xy[i][0] - x0, cam_xy[i][1] - y0
        for d, w in g.walls.items():
            letter = LETTER[d]
            uz = wall_uz(g.P, d, w["dist"], g.cam_height)
            if letter == "A":
                f = lambda u: cx + u
            elif letter == "C":
                f = lambda u: Lx - (cx + u)
            elif letter == "B":
                f = lambda u: Ly - (cy + u)
            else:
                f = lambda u: cy + u
            for k in ("on", "cross", "near"):
                acc[letter][k + "_u"].append(f(uz[k + "_u"])); acc[letter][k + "_z"].append(uz[k + "_z"])
    out = []
    for letter, a in acc.items():
        if not a["on_u"]:
            continue
        uz = {k: (np.concatenate(v) if v else np.array([])) for k, v in a.items()}
        L = Lx if letter in "AC" else Ly
        for op in openings_from_uz(uz, bin_m=0.02, min_support=15, u_range=(0.0, L)):
            u0, u1 = max(0.0, op["u0"]), min(L, op["u1"])
            n_sup = op["support"]
            o = Opening(wall_id=letter, type=op["type"],
                        offset=Measurement(u0, 0.015, "measured", "accumulated ray crossings through the wall plane", n_sup),
                        width=Measurement(u1 - u0, 0.012, "measured", "accumulated hole width in wall point density", n_sup),
                        confidence=min(0.95, 0.5 + 0.0005 * n_sup))
            if np.isfinite(op["top"]):
                o.height = Measurement(op["top"] - (op["bottom"] if np.isfinite(op["bottom"]) else 0.0), 0.06, "measured", "hole vertical extent", n_sup)
            if op["type"] == "window" and np.isfinite(op["bottom"]):
                o.sill = Measurement(op["bottom"], 0.06, "measured", "hole bottom", n_sup)
            out.append(o)
    return out


def _room_openings(frames, geoms, room: RoomEstimate, cam_xy, x0, y0) -> list[Opening]:
    if cam_xy is not None:
        return _accumulated_openings(frames, geoms, room, cam_xy, x0, y0)
    Lx, Ly = room.length_x.value, room.length_y.value
    cands = []
    corner_i = 0
    for i, (fr, g) in enumerate(zip(frames, geoms)):
        if g is None or g.P is None:
            continue
        rel = g.depth_rel_err
        ci = None
        if fr.role == "corner":
            ci = corner_i; corner_i += 1
        if cam_xy is not None and cam_xy[i] is not None:
            cx, cy = cam_xy[i][0] - x0, cam_xy[i][1] - y0
            sx = sy = 0.02
        else:
            cx, cy, sx, sy = _cam_position(g, fr, ci, Lx, Ly)
        for d, w in g.walls.items():
            letter = LETTER[d]
            for op in wall_openings(g.P, d, w["dist"], g.cam_height):
                # crossing coordinates are lateral (u) relative to the camera
                if letter == "A":
                    base, sb, sgn, ref = cx, sx, 1, 0.0            # off = cx + u
                elif letter == "C":
                    base, sb, sgn, ref = cx, sx, -1, Lx            # off = Lx - (cx + u)
                elif letter == "B":
                    base, sb, sgn, ref = cy, sy, -1, Ly            # off = Ly - (cy + u)
                else:
                    base, sb, sgn, ref = cy, sy, 1, 0.0            # off = cy + u
                if np.isfinite(base):
                    lo_ = ref + sgn * (base + op["u0"]); hi_ = ref + sgn * (base + op["u1"])
                    off, off_s = min(lo_, hi_), sb + rel * w["dist"] * 0.3
                    if fr.role == "exit":
                        off_s += 0.5
                else:
                    off, off_s = 0.5 * (Lx if letter in "AC" else Ly) - 0.5 * op["width"], 1.5
                cands.append(dict(wall=letter, type=op["type"], off=off, off_s=off_s, width=op["width"],
                                  width_s=rel * w["dist"] * 0.5 + 0.03, top=op["top"], bottom=op["bottom"],
                                  support=op["support"], role=fr.role))
    merged: list[Opening] = []
    cands.sort(key=lambda c: -c["support"])
    used = [False] * len(cands)
    for i, c in enumerate(cands):
        if used[i]:
            continue
        group = [c]; used[i] = True
        for j in range(i + 1, len(cands)):
            o = cands[j]
            if used[j] or o["wall"] != c["wall"] or o["type"] != c["type"]:
                continue
            tol = max(0.4, 1.5 * max(c["off_s"], o["off_s"]))
            if abs(o["off"] - c["off"]) < tol and abs(o["width"] - c["width"]) < 0.35:
                group.append(o); used[j] = True
        w = np.array([1 / max(x["width_s"], 0.02) ** 2 for x in group])
        width = float(np.sum(w * [x["width"] for x in group]) / w.sum())
        width_s = float(max(1 / np.sqrt(w.sum()), 0.015))
        wo = np.array([1 / max(x["off_s"], 0.02) ** 2 for x in group])
        off = float(np.sum(wo * [x["off"] for x in group]) / wo.sum())
        off_s = float(max(1 / np.sqrt(wo.sum()), 0.03))
        tops = [x["top"] for x in group if np.isfinite(x["top"])]
        bots = [x["bottom"] for x in group if np.isfinite(x["bottom"])]
        conf = min(0.95, 0.4 + 0.12 * len(group) + 0.0004 * sum(x["support"] for x in group))
        opn = Opening(wall_id=c["wall"], type=c["type"],
                      offset=Measurement(max(0.0, off), off_s, "measured", "ray crossings through the wall plane", len(group)),
                      width=Measurement(width, width_s, "measured", "hole width in wall point density", len(group)),
                      confidence=conf)
        if tops:
            b = float(np.median(bots)) if bots else 0.0
            opn.height = Measurement(float(np.median(tops)) - b, 0.08, "measured", "hole vertical extent", len(tops))
        if c["type"] == "window" and bots:
            opn.sill = Measurement(float(np.median(bots)), 0.08, "measured", "hole bottom", len(bots))
        merged.append(opn)
    # drop duplicates that overlap on the same wall (keep the better supported)
    final = []
    for o in merged:
        dup = False
        for f in final:
            if f.wall_id == o.wall_id and abs(f.offset.value - o.offset.value) < 0.5 * (f.width.value + o.width.value):
                dup = True; break
        if not dup:
            final.append(o)
    return final
