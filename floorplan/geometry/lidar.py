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


def wall_observations(geoms, cams):
    """Absolute wall observations from oriented frames with known camera positions."""
    obs = []
    for i, (g, c) in enumerate(zip(geoms, cams)):
        if g is None:
            continue
        for d, w in g.walls.items():
            ax = 0 if d[1] == "x" else 1
            sgn = 1 if d[0] == "+" else -1
            lat_ax = 1 - ax
            obs.append(dict(frame=i, dir=d, axis=ax, coord=float(c[ax] + sgn * w["dist"]),
                            lat_lo=float(c[lat_ax] + w["lat_min"]), lat_hi=float(c[lat_ax] + w["lat_max"]),
                            dist=float(w["dist"]), rms=float(w["rms"]), n=int(w["n"])))
    return obs


def global_lines(obs, axis, bin_m=0.04, min_frames=6):
    """1-D clustering of wall coordinates along one axis → list of dict(coord, support, spans)."""
    o = [x for x in obs if x["axis"] == axis]
    if not o:
        return []
    v = np.array([x["coord"] for x in o])
    w = np.array([1.0 / (0.01 * x["dist"] + x["rms"] + 0.005) ** 2 for x in o])
    lo, hi = v.min() - 0.2, v.max() + 0.2
    nb = int(np.ceil((hi - lo) / bin_m))
    h, e = np.histogram(v, bins=nb, range=(lo, hi))
    hs = np.convolve(h, np.ones(3), mode="same")
    lines = []
    used = np.zeros(len(o), bool)
    order = np.argsort(-hs)
    for b in order:
        if hs[b] < min_frames:
            break
        centre = lo + (b + 0.5) * bin_m
        if any(abs(centre - L["coord"]) < 0.25 for L in lines):
            continue
        sel = (np.abs(v - centre) < 0.15) & ~used
        if sel.sum() < min_frames:
            continue
        coord = float(np.sum(w[sel] * v[sel]) / np.sum(w[sel]))
        sel = (np.abs(v - coord) < 0.15) & ~used
        used |= sel
        spans = [(o[i]["lat_lo"], o[i]["lat_hi"]) for i in np.where(sel)[0]]
        lines.append(dict(coord=coord, support=int(sel.sum()), spans=spans,
                          sigma=float(1.0 / np.sqrt(np.sum(w[sel])))))
    # drop weakly supported lines (door jambs, furniture edges): need frames AND observed wall length
    if lines:
        smax = max(L["support"] for L in lines)
        lines = [L for L in lines if L["support"] >= max(min_frames, 0.08 * smax)
                 and sum(hi - lo for lo, hi in L["spans"]) >= 8.0]
    lines.sort(key=lambda L: L["coord"])
    return lines


def _coverage(spans, a, b, step=0.05):
    """Fraction of [a, b] covered by the union of spans."""
    if b - a < step:
        return 0.0
    xs = np.arange(a + step / 2, b, step)
    cov = np.zeros(len(xs), bool)
    for lo, hi in spans:
        cov |= (xs >= lo) & (xs <= hi)
    return float(cov.mean())


def anchor_translation(geoms, cams, poses, X, Y, window=8):
    """Plane-anchored translation correction: each frame's wall observations are compared with the
    global wall lines; the smoothed residual is removed from the camera positions (and poses)."""
    n = len(cams)
    res = np.full((n, 2), np.nan)
    acc = [[[] for _ in range(2)] for _ in range(n)]
    for o in wall_observations(geoms, cams):
        lines = X if o["axis"] == 0 else Y
        if not lines:
            continue
        L = min(lines, key=lambda L: abs(L["coord"] - o["coord"]))
        if abs(L["coord"] - o["coord"]) < 0.2:
            acc[o["frame"]][o["axis"]].append(o["coord"] - L["coord"])
    for i in range(n):
        for a in range(2):
            if acc[i][a]:
                res[i, a] = np.mean(acc[i][a])
    corr = np.zeros((n, 2))
    for a in range(2):
        v = res[:, a]
        idx = np.where(np.isfinite(v))[0]
        if len(idx) < 3:
            continue
        sm = np.full(n, np.nan)
        for i in range(n):
            seg = v[max(0, i - window):i + window + 1]
            seg = seg[np.isfinite(seg)]
            if len(seg):
                sm[i] = np.median(seg)
        bad = ~np.isfinite(sm)
        sm[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], sm[~bad])
        corr[:, a] = sm
    cams2 = cams - corr
    return cams2, corr


def rooms_from_cells(geoms, cams, X, Y, min_frames=5):
    """Cells of the wall-line arrangement visited by the camera, merged where no wall separates them."""
    xs = np.array([L["coord"] for L in X]); ys = np.array([L["coord"] for L in Y])
    if len(xs) < 2 or len(ys) < 2:
        return [], np.full(len(cams), -1)
    ci = np.searchsorted(xs, cams[:, 0]) - 1
    cj = np.searchsorted(ys, cams[:, 1]) - 1
    valid = (ci >= 0) & (ci < len(xs) - 1) & (cj >= 0) & (cj < len(ys) - 1)
    counts = {}
    for i in np.where(valid)[0]:
        counts[(ci[i], cj[i])] = counts.get((ci[i], cj[i]), 0) + 1
    cells = [c for c, k in counts.items() if k >= min_frames]
    parent = {c: c for c in cells}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]; c = parent[c]
        return c

    for (i, j) in cells:
        # right neighbour across line x = xs[i+1] over span y ∈ [ys[j], ys[j+1]]
        if (i + 1, j) in parent:
            if _coverage(X[i + 1]["spans"], ys[j], ys[j + 1]) < 0.3:
                parent[find((i, j))] = find((i + 1, j))
        if (i, j + 1) in parent:
            if _coverage(Y[j + 1]["spans"], xs[i], xs[i + 1]) < 0.3:
                parent[find((i, j))] = find((i, j + 1))
    comp = {}
    for c in cells:
        comp.setdefault(find(c), []).append(c)
    # order rooms by first visit
    first = {}
    for i in np.where(valid)[0]:
        c = (ci[i], cj[i])
        if c in parent:
            r = find(c)
            first.setdefault(r, i)
    order = sorted(comp, key=lambda r: first[r])
    rooms = []
    frame_room = np.full(len(cams), -1)
    for k, r in enumerate(order):
        cs = comp[r]
        xi = [c[0] for c in cs]; yj = [c[1] for c in cs]
        bounds = dict(xmin=float(xs[min(xi)]), xmax=float(xs[max(xi) + 1]), ymin=float(ys[min(yj)]), ymax=float(ys[max(yj) + 1]))
        rect = len(cs) == (max(xi) - min(xi) + 1) * (max(yj) - min(yj) + 1)
        rooms.append(dict(bounds=bounds, cells=cs, rectangular=rect))
        for i in np.where(valid)[0]:
            if (ci[i], cj[i]) in parent and find((ci[i], cj[i])) == r:
                frame_room[i] = k
    return rooms, frame_room


def room_heights_from_points(frames, geoms, cams, stride: int = 3, bounds: dict | None = None,
                            global_floor_z: float | None = None):
    """Floor and ceiling z (world) from the accumulated near-horizontal surface points of a room.
    Returns (floor_z, ceil_z, sigma, n_floor, n_ceil) or None."""
    zf, zc = [], []
    for k, (fr, g, c) in enumerate(zip(frames, geoms, cams)):
        if g is None or g.P is None or (k % stride):
            continue
        P = g.P
        N = normals_from_grid(P)
        ok = np.isfinite(P).all(-1) & np.isfinite(N).all(-1)
        if fr.conf is not None and fr.conf.shape == P.shape[:2]:
            ok &= fr.conf >= 1
        z = P[..., 2] + float(fr.pose[2, 3])
        if bounds is not None:
            xw = P[..., 0] + c[0]; yw = P[..., 1] + c[1]
            ok &= (xw > bounds["xmin"] + 0.1) & (xw < bounds["xmax"] - 0.1) & (yw > bounds["ymin"] + 0.1) & (yw < bounds["ymax"] - 0.1)
        up = N[..., 2] > 0.9          # normal toward camera & up → floor
        down = N[..., 2] < -0.9       # ceiling
        zf.append(z[ok & up]); zc.append(z[ok & down])
    zf = np.concatenate(zf) if zf else np.array([]); zc = np.concatenate(zc) if zc else np.array([])
    def mode_median(v, lo, hi):
        h, e = np.histogram(v, bins=int((hi - lo) / 0.02) + 1, range=(lo, hi))
        m = lo + (np.argmax(h) + 0.5) * 0.02
        sel = np.abs(v - m) < 0.06
        return float(np.median(v[sel])), int(sel.sum()), float(np.std(v[sel]))

    if len(zc) < 300:
        return None
    if len(zf) >= 60:
        fz, nf, sf = mode_median(zf, -1.0, 1.0)
    elif global_floor_z is not None:
        fz, nf, sf = global_floor_z, 10 ** 6, 0.004     # shallow room: floor rarely in the LiDAR's field of view
    else:
        return None
    cz, nc, sc = mode_median(zc, 1.8, 4.5)
    sigma = float(np.hypot(sf / np.sqrt(max(nf, 1)), sc / np.sqrt(max(nc, 1))) + 0.003)
    return fz, cz, sigma, nf, nc


def global_floor_z(frames, geoms, stride: int = 4) -> float | None:
    zs = []
    for k, (fr, g) in enumerate(zip(frames, geoms)):
        if g is None or g.P is None or (k % stride):
            continue
        N = normals_from_grid(g.P)
        ok = np.isfinite(g.P).all(-1) & np.isfinite(N).all(-1) & (N[..., 2] > 0.9)
        z = g.P[..., 2][ok] + float(fr.pose[2, 3])
        zs.append(z[np.abs(z) < 0.5])
    zs = np.concatenate(zs) if zs else np.array([])
    if len(zs) < 500:
        return None
    h, e = np.histogram(zs, bins=50, range=(-0.5, 0.5))
    m = e[np.argmax(h)] + 0.01
    return float(np.median(zs[np.abs(zs - m) < 0.06]))
