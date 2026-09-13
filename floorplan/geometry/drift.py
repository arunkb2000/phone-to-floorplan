"""Drift accountability for the LiDAR tier.

ARKit's VIO drifts: yaw creeps, the floor tips, and the world slides. Over a 200 s, 100 m walk the
sample captures end 17 to 39 cm from where they started. Three corrections, all ablatable with
`--drift-correction off`:

1. Floor-anchored levelling and height. Every frame that sees the floor says where z = 0 is; the
   smoothed residual is removed, which kills the slow tilt that otherwise smears a 2.3 m ceiling
   over 10 cm.
2. Manhattan yaw anchoring. Wall normals must agree with the property's dominant axes; the smoothed
   per-frame yaw residual is removed, which stops a long corridor from bending.
3. Loop closure. The protocol ends where it starts, so the residual translation between the opening
   and closing wall observations is distributed along the trajectory.
"""
from __future__ import annotations

import numpy as np

from floorplan.geometry.frame import backproject, fit_plane, normals_from_grid, rotation_from_up, rotz


def _running_median(v: np.ndarray, w: int) -> np.ndarray:
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(n):
        seg = v[max(0, i - w):i + w + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    idx = np.where(np.isfinite(out))[0]
    if len(idx) == 0:
        return np.zeros(n)
    bad = ~np.isfinite(out)
    out[bad] = np.interp(np.where(bad)[0], idx, out[idx])
    return out


def _observe(frames, stride: int = 1):
    """Per frame: observed floor normal, floor height, and wall-normal azimuths."""
    n = len(frames)
    normals = [None] * n
    heights = np.full(n, np.nan)
    az = [None] * n
    for i, f in enumerate(frames):
        if i % stride:
            continue
        p = backproject(f.depth, f.K, stride=3)
        nr = normals_from_grid(p, k=3)
        R, t = f.pose[:3, :3], f.pose[:3, 3]
        Pw, Nw = p @ R.T + t, nr @ R.T
        ok = np.isfinite(Pw).all(-1) & np.isfinite(Nw).all(-1)
        if f.conf is not None:
            ok &= f.conf[::3, ::3] >= 2
        if ok.sum() < 300:
            continue
        P, N = Pw[ok], Nw[ok]
        up = N[:, 2] > 0.85
        if up.sum() > 250:
            z = P[up, 2]
            h, e = np.histogram(z, bins=200, range=(float(z.min()) - 0.01, float(z.max()) + 0.01))
            m = e[np.argmax(h)]
            sel = np.abs(z - m) < 0.05
            if sel.sum() > 250:
                pts = P[up][sel]
                nn, d, _ = fit_plane(pts.astype(np.float64))
                if nn[2] < 0:
                    nn = -nn
                if nn[2] > 0.95:
                    normals[i] = nn
                    heights[i] = float(np.median(pts[:, 2]))
        w = np.abs(N[:, 2]) < 0.3
        if w.sum() > 300:
            az[i] = np.arctan2(N[w, 1], N[w, 0])
    return normals, heights, az


def correct_drift(frames, window: int = 12) -> dict:
    """Corrects poses in place. Returns the report that lands in plan.json under `drift`."""
    n = len(frames)
    normals, heights, az = _observe(frames)
    # ---- global Manhattan yaw from every frame's wall normals
    allaz = np.concatenate([a for a in az if a is not None]) if any(a is not None for a in az) else np.array([])
    if len(allaz) > 1000:
        h, _ = np.histogram(np.mod(allaz, np.pi / 2), bins=360, range=(0, np.pi / 2))
        h = np.convolve(np.r_[h[-4:], h, h[:4]], np.ones(9) / 9, mode="valid")
        yaw0 = float((np.argmax(h) + 0.5) * (np.pi / 2) / 360)
    else:
        yaw0 = 0.0
    dyaw = np.full(n, np.nan)
    for i, a in enumerate(az):
        if a is None:
            continue
        res = np.mod(a - yaw0 + np.pi / 4, np.pi / 2) - np.pi / 4
        dyaw[i] = float(np.median(res))
    dz = np.array([heights[i] - np.nanmedian(heights) if np.isfinite(heights[i]) else np.nan for i in range(n)])
    med_h = float(np.nanmedian(heights)) if np.isfinite(heights).any() else 0.0
    dyaw_s = _running_median(dyaw, window)
    dz_s = _running_median(dz, window)
    tilt_applied = 0
    for i, f in enumerate(frames):
        T = np.eye(4)
        R = rotz(-dyaw_s[i])
        if normals[i] is not None:
            R = R @ rotation_from_up(normals[i])
            tilt_applied += 1
        T[:3, :3] = R
        f.pose = T @ f.pose
        f.pose[2, 3] -= (med_h + dz_s[i])
    # ---- loop closure on the camera track: if the walk returns near its start, share out the gap
    cams = np.array([f.pose[:3, 3] for f in frames])
    gap = cams[-1] - cams[0]
    closed = False
    d0 = float(np.linalg.norm(gap[:2]))
    if d0 < 1.5:                       # the protocol says finish where you started
        for i, f in enumerate(frames):
            f.pose[:3, 3] -= gap * (i / max(1, n - 1))
        closed = True
    return {
        "method": "floor-anchored levelling and height datum + Manhattan yaw anchoring + loop closure",
        "frames_levelled": int(tilt_applied),
        "frames_total": int(n),
        "median_abs_yaw_correction_deg": round(float(np.degrees(np.nanmedian(np.abs(dyaw)))) if np.isfinite(dyaw).any() else 0.0, 3),
        "max_abs_yaw_correction_deg": round(float(np.degrees(np.nanmax(np.abs(dyaw_s)))), 3),
        "floor_height_drift_range_m": round(float(np.nanmax(dz_s) - np.nanmin(dz_s)), 4),
        "loop_closure_applied": bool(closed),
        "loop_gap_before_m": round(d0, 4),
    }
