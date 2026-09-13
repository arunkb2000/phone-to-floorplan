"""LiDAR tier: frames with poses → drift correction → per-frame geometry in a global Manhattan
frame → trajectory segmentation into rooms → absolute room rectangles.

Drift handling (scored row): (1) plane-anchored correction of yaw, tilt and height per frame using
re-observed wall/floor planes; (2) loop closure: the protocol ends where it started, so the walls
re-observed at the end are matched to the same walls at the start and the residual translation is
distributed along the trajectory. `--drift-correction off` uses the poses as they came.
"""
from __future__ import annotations

import numpy as np

from floorplan.core.types import Frame
from floorplan.geometry.frame import (FrameGeom, analyse_frame, backproject, normals_from_grid, ransac_plane,
                                      rotation_from_up, rotz)
from floorplan.geometry.room import TIER_DEPTH_REL_ERR


def _global_yaw(frames: list[Frame], stride: int = 8) -> float:
    az_all = []
    for fr in frames[::stride]:
        P = backproject(fr.depth, fr.K, stride=2)
        N = normals_from_grid(P)
        R = fr.pose[:3, :3]
        Nw = N @ R.T
        ok = np.isfinite(Nw).all(-1) & (np.abs(Nw[..., 2]) < 0.35)
        n = Nw[ok]
        if len(n):
            az_all.append(np.arctan2(n[:, 1], n[:, 0]))
    if not az_all:
        return 0.0
    az = np.concatenate(az_all)
    a4 = np.mod(az, np.pi / 2)
    hist, _ = np.histogram(a4, bins=180, range=(0, np.pi / 2))
    hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, mode="valid")
    return float((np.argmax(hist) + 0.5) * (np.pi / 2) / 180)


def _floor_and_yaw_obs(fr: Frame, yaw: float, rng):
    """Observed floor (normal, height under camera) and wall-normal yaw residual for one frame, in world."""
    P = backproject(fr.depth, fr.K, stride=3)
    N = normals_from_grid(P)
    R = fr.pose[:3, :3]
    Pw, Nw = P @ R.T, N @ R.T
    ok = np.isfinite(Pw).all(-1) & np.isfinite(Nw).all(-1)
    pts, nrm = Pw[ok], Nw[ok]
    if len(pts) < 300:
        return None, None, None
    z = np.array([0, 0, 1.0])
    r = ransac_plane(pts, nrm, z, 20.0, 0.03, 120, rng, min_inliers=200)
    floor = None
    if r is not None:
        n, d, rms, inl = r
        if 0.6 <= d <= 2.2:
            floor = (n, float(d))
    sel = np.abs(nrm[:, 2]) < 0.3
    dyaw = None
    if sel.sum() > 200:
        az = np.arctan2(nrm[sel, 1], nrm[sel, 0])
        res = np.mod(az - yaw + np.pi / 4, np.pi / 2) - np.pi / 4   # residual to nearest Manhattan axis
        if len(res) > 200:
            dyaw = float(np.median(res))
    return floor, dyaw, len(pts)


def correct_drift(frames: list[Frame], yaw: float, seed: int = 0) -> dict:
    """In-place pose correction. Returns a small report for the JSON."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    tilt = [None] * n; dz = np.full(n, np.nan); dyaw = np.full(n, np.nan)
    for i, fr in enumerate(frames):
        floor, dy, _ = _floor_and_yaw_obs(fr, yaw, rng)
        if floor is not None:
            nrm, h = floor
            tilt[i] = rotation_from_up(nrm)          # rotation that levels the observed floor
            cam_z = fr.pose[2, 3]
            dz[i] = cam_z - h                        # floor height in world: should be 0
        if dy is not None:
            dyaw[i] = dy
    # smooth with a running median (window 15) and fill gaps
    def smooth(v, w=15):
        out = np.copy(v)
        idx = np.where(np.isfinite(v))[0]
        if len(idx) == 0:
            return np.zeros_like(v)
        for i in range(n):
            lo, hi = max(0, i - w), min(n, i + w + 1)
            seg = v[lo:hi]; seg = seg[np.isfinite(seg)]
            out[i] = np.median(seg) if len(seg) else np.nan
        # fill remaining NaNs by nearest
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = np.interp(np.where(bad)[0], idx, v[idx])
        return out
    dz_s, dyaw_s = smooth(dz), smooth(dyaw)
    for i, fr in enumerate(frames):
        Rc = rotz(-dyaw_s[i])
        if tilt[i] is not None:
            Rc = Rc @ tilt[i]
        T = np.eye(4)
        T[:3, :3] = Rc
        fr.pose = T @ fr.pose
        fr.pose[2, 3] -= dz_s[i]
    # loop closure on re-observed walls: compare absolute wall coordinates in the first and last 8 %
    def wall_coords(sub):
        acc = {"+x": [], "-x": [], "+y": [], "-y": []}
        for fr in sub:
            g = frame_geom_world(fr, yaw, seed)
            if g is None:
                continue
            c = rotz(-yaw) @ fr.pose[:3, 3]
            for d, w in g.walls.items():
                ax = 0 if d[1] == "x" else 1
                acc[d].append(c[ax] + (1 if d[0] == "+" else -1) * w["dist"])
        return {k: (float(np.median(v)) if len(v) >= 3 else np.nan) for k, v in acc.items()}
    m = max(5, int(0.08 * n))
    a, b = wall_coords(frames[:m]), wall_coords(frames[-m:])
    dxs = [b[k] - a[k] for k in ("+x", "-x") if np.isfinite(a[k]) and np.isfinite(b[k]) and abs(b[k] - a[k]) < 0.6]
    dys = [b[k] - a[k] for k in ("+y", "-y") if np.isfinite(a[k]) and np.isfinite(b[k]) and abs(b[k] - a[k]) < 0.6]
    loop = np.array([np.mean(dxs) if dxs else 0.0, np.mean(dys) if dys else 0.0])
    loop_world = rotz(yaw)[:2, :2] @ loop
    for i, fr in enumerate(frames):
        fr.pose[:2, 3] -= loop_world * (i / max(1, n - 1))
    return {"method": "plane-anchored yaw/tilt/height + loop closure on re-observed walls",
            "loop_closures": int(bool(dxs or dys)), "loop_residual_m": [round(float(x), 4) for x in loop],
            "median_abs_dyaw_deg": round(float(np.degrees(np.nanmedian(np.abs(dyaw)))), 3) if np.isfinite(dyaw).any() else 0.0,
            "median_abs_dz_m": round(float(np.nanmedian(np.abs(dz))), 4) if np.isfinite(dz).any() else 0.0}


def frame_geom_world(fr: Frame, yaw: float, seed: int = 0) -> FrameGeom | None:
    """Per-frame geometry expressed in the global Manhattan frame (camera at origin)."""
    P = backproject(fr.depth, fr.K, stride=1)
    R = fr.pose[:3, :3]
    up_hint = R.T @ np.array([0, 0, 1.0])
    return analyse_frame(P, up_hint=up_hint, seed=seed, depth_rel_err=TIER_DEPTH_REL_ERR["lidar"],
                         R_override=R, yaw_override=yaw)


def segment_rooms(frames: list[Frame], geoms: list[FrameGeom | None], cams: np.ndarray):
    """Split the walk into room visits using the camera position against the running room box."""
    n = len(frames)
    seg = np.full(n, -1)
    boxes = []   # per segment: dict with fused wall coords
    cur = None; cur_id = -1; outside = 0
    def obs(i):
        g = geoms[i]; c = cams[i]
        o = {}
        if g is None:
            return o
        for d, w in g.walls.items():
            ax = 0 if d[1] == "x" else 1
            o[d] = c[ax] + (1 if d[0] == "+" else -1) * w["dist"]
        return o
    def inside(box, c, margin=0.15):
        for d, v in box.items():
            if not np.isfinite(v):
                continue
            ax = 0 if d[1] == "x" else 1
            if d[0] == "+" and c[ax] > v + margin:
                return False
            if d[0] == "-" and c[ax] < v - margin:
                return False
        return True
    for i in range(n):
        o = obs(i)
        if cur is None:
            cur = {k: [v] for k, v in o.items()}; cur_id += 1; boxes.append(cur); seg[i] = cur_id
            continue
        box = {k: np.median(v) for k, v in cur.items()}
        disagree = any(k in box and abs(box[k] - v) > 0.35 for k, v in o.items())
        if (not inside(box, cams[i])) or disagree:
            outside += 1
        else:
            outside = 0
        if outside >= 4:
            cur = {k: [v] for k, v in o.items()}; cur_id += 1; boxes.append(cur); outside = 0
            seg[i - 3:i + 1] = cur_id
            continue
        for k, v in o.items():
            cur.setdefault(k, []).append(v)
        seg[i] = cur_id
    # merge segments that describe the same box (return to a room)
    fused = [{k: float(np.median(v)) for k, v in b.items()} for b in boxes]
    room_of_seg = list(range(len(boxes)))
    for s in range(len(boxes)):
        for r in range(s):
            if room_of_seg[r] != r:
                continue
            a, b = fused[s], fused[r]
            common = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
            if len(common) >= 2 and all(abs(a[k] - b[k]) < 0.3 for k in common):
                room_of_seg[s] = r
                break
    # drop tiny segments (< 8 frames) into their predecessor
    counts = np.bincount(seg[seg >= 0], minlength=len(boxes))
    for s in range(1, len(boxes)):
        if counts[s] < 8:
            room_of_seg[s] = room_of_seg[s - 1]
    rooms_order = []
    for s in range(len(boxes)):
        r = room_of_seg[s]
        while room_of_seg[r] != r:
            r = room_of_seg[r]
        room_of_seg[s] = r
        if r not in rooms_order:
            rooms_order.append(r)
    room_idx = np.array([rooms_order.index(room_of_seg[s]) for s in seg])
    transitions = []
    for i in range(1, n):
        if room_idx[i] != room_idx[i - 1]:
            transitions.append((int(room_idx[i - 1]), int(room_idx[i]), i))
    return room_idx, len(rooms_order), transitions
