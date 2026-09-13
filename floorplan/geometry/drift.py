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

CARRY_MIN, CARRY_MAX = 0.9, 2.0      # plausible range for the phone above the floor, in metres
MAX_DATUM_RESIDUAL = 0.12            # a frame further than this from the running datum is an outlier


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


def _observe(frames, stride: int = 1, datum: str = "robust"):
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
            # The floor is the LOWEST well-supported horizontal surface a plausible carry height
            # below the camera, not the most populous one. In a furnished office the modal up-facing
            # surface is a desk at 0.75 m, and using it as the height datum swings the whole pose
            # by three quarters of a metre. See fixloop/DIFF.md, the second defect.
            cam_z = float(t[2])
            h, e = np.histogram(z, bins=300, range=(float(z.min()) - 0.01, float(z.max()) + 0.01))
            if datum == "legacy":
                pick = float(e[np.argmax(h)])      # pre-fix: the modal horizontal surface, desk included
            else:
                strong = np.where(h >= max(250, 0.25 * h.max()))[0]
                pick = None
                for b in strong:                   # ascending height: first plausible one wins
                    zc = float(e[b])
                    if CARRY_MIN <= cam_z - zc <= CARRY_MAX:
                        pick = zc
                        break
            if pick is not None:
                sel = np.abs(z - pick) < 0.05
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


def correct_drift(frames, window: int = 12, datum: str = "robust") -> dict:
    """Corrects poses in place. Returns the report that lands in plan.json under `drift`."""
    n = len(frames)
    normals, heights, az = _observe(frames, datum=datum)
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
    med_h = float(np.nanmedian(heights)) if np.isfinite(heights).any() else 0.0
    dz = heights - med_h
    # a frame whose floor estimate is far from the running datum measured something that is not the
    # floor; drop it rather than let the smoother drag the whole trajectory toward it
    rough = _running_median(dz, window)
    if datum != "legacy":
        dz = np.where(np.abs(dz - rough) > MAX_DATUM_RESIDUAL, np.nan, dz)
    n_datum_outliers = int(np.sum(np.isfinite(heights) & ~np.isfinite(dz)))
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
        "datum_mode": datum,
        "frames_levelled": int(tilt_applied),
        "frames_total": int(n),
        "median_abs_yaw_correction_deg": round(float(np.degrees(np.nanmedian(np.abs(dyaw)))) if np.isfinite(dyaw).any() else 0.0, 3),
        "max_abs_yaw_correction_deg": round(float(np.degrees(np.nanmax(np.abs(dyaw_s)))), 3),
        "floor_height_drift_range_m": round(float(np.nanmax(dz_s) - np.nanmin(dz_s)), 4),
        "frames_with_floor": int(np.isfinite(heights).sum()),
        "datum_outliers_rejected": n_datum_outliers,
        "loop_closure_applied": bool(closed),
        "loop_gap_before_m": round(d0, 4),
    }
