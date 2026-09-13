"""Reference ground truth for the supplied captures, measured from single depth frames.

WHY NOT TAPE. The brief asks for laser or tape ground truth. The captures we were given are of a
property we have no physical access to, so a tape measurement of them is not available to us at any
price. Rather than quietly skip the gates, we build a reference that is as independent of the
pipeline as the data allows, and we say exactly what it is and is not.

WHAT THIS IS. Every reference number here comes from ONE depth frame:

  * ceiling height  = distance between the floor plane and the ceiling plane fitted in that single
                      frame's own camera coordinates
  * wall-to-wall    = sum of the perpendicular distances to two opposite, parallel wall planes seen
                      in the same single frame
  * opening width   = the gap in one wall plane, measured across that plane in the same single frame

INDEPENDENCE. These numbers use only the depth image and the intrinsics. They do not use the poses,
the pose graph, the drift correction, the multi-frame fusion, the room segmentation, the grid, or
any pipeline module - this file implements its own plane fitting in about sixty lines and imports
nothing from floorplan except the loader. So it is a fair test of everything the pipeline does on
top of the sensor. It is NOT independent of the sensor itself: a systematic LiDAR range error would
move the reference and the pipeline together, and no experiment here can see that. Apple publishes
no accuracy figure for the sensor; independent measurements put it around 1 cm at these ranges.

UNCERTAINTY. Each reference value is the median over every qualifying frame, and its stated
uncertainty is the spread across those frames, which folds in sensor noise, incidence angle and the
plane-fit residual. Measurements supported by fewer than `--min-frames` frames are dropped.

    uv run python scripts/make_reference_gt.py Dataset/single_scan_with_ceiling \
        --out benchmark/ground_truth/flatA.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

from floorplan.io.stray import load_stray_capture


def backproject(depth, K):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = depth
    return np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], -1)


def grid_normals(P, k=3):
    dx = np.full_like(P, np.nan)
    dy = np.full_like(P, np.nan)
    dx[:, k:-k] = P[:, 2 * k:] - P[:, :-2 * k]
    dy[k:-k, :] = P[2 * k:, :] - P[:-2 * k, :]
    n = np.cross(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    flip = np.nansum(n * P, -1) > 0
    n[flip] *= -1
    return n


def fit(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    d = -float(n @ c)
    return n, d, float(np.sqrt(np.mean((pts @ n + d) ** 2)))


def plane_along(P, N, axis, pts_min=1500, tol=0.9, thick=0.04):
    """Strongest plane whose normal is within `tol` of `axis` (a unit vector in camera coords)."""
    m = np.isfinite(P).all(-1) & np.isfinite(N).all(-1) & ((N @ axis) > tol)
    if m.sum() < pts_min:
        return None
    p = P[m]
    s = p @ axis
    h, e = np.histogram(s, bins=400, range=(float(s.min()), float(s.max()) + 1e-3))
    if h.max() < pts_min // 2:
        return None
    peak = e[np.argmax(h)]
    sel = np.abs(s - peak) < thick
    if sel.sum() < pts_min // 2:
        return None
    n, d, rms = fit(p[sel].astype(np.float64))
    if n @ axis < 0:
        n, d = -n, -d
    return dict(n=n, d=float(d), rms=rms, n_pts=int(sel.sum()), pts=p[sel])


def dominant_wall_axis(P, N, up, bins=180):
    """The frame's own dominant wall normal, found without reference to any global frame."""
    m = np.isfinite(P).all(-1) & np.isfinite(N).all(-1) & (np.abs(N @ up) < 0.3)
    if m.sum() < 2500:
        return None
    e1 = np.cross(up, np.array([1.0, 0.0, 0.0]))
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, np.array([0.0, 1.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    n = N[m]
    az = np.arctan2(n @ e2, n @ e1)
    h, edges = np.histogram(az, bins=bins, range=(-np.pi, np.pi))
    h = np.convolve(np.r_[h[-3:], h, h[:3]], np.ones(7) / 7, "valid")
    a = edges[int(np.argmax(h))] + np.pi / bins
    return np.cos(a) * e1 + np.sin(a) * e2


def measure_frame(f, min_gap=0.55, max_gap=2.6):
    """Single-frame measurements, in that frame's own camera coordinates.

    The only thing taken from outside the frame is the direction of gravity, and only as a
    direction: either the frame's own floor plane normal, or the rotation part of the ARKit pose.
    No translation, no drift state, no other frame.
    """
    P = backproject(f.depth.astype(np.float64), f.K)
    N = grid_normals(P)
    ok = (f.depth > 0.25) & (f.depth < 5.0)
    if f.conf is not None:
        ok &= f.conf >= 2
    P = np.where(ok[..., None], P, np.nan)
    up = f.pose[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    floor = plane_along(P, N, up)
    if floor is not None:
        up = floor["n"] / np.linalg.norm(floor["n"])
        floor = plane_along(P, N, up)
    ceil = plane_along(P, N, -up)
    out = {}
    if floor and ceil:
        h = abs(floor["d"] + ceil["d"])
        if 1.9 < h < 4.2:
            out["ceiling_height"] = dict(value=float(h), source="floor+ceiling planes in one frame",
                                         n_pts=int(min(floor["n_pts"], ceil["n_pts"])))
    ax = dominant_wall_axis(P, N, up)
    if ax is None:
        return out
    lat = np.cross(up, ax)
    lat /= max(np.linalg.norm(lat), 1e-9)
    walls = {}
    for sgn in (1, -1):
        for a, tag in ((sgn * ax, f"a{sgn}"), (sgn * lat, f"b{sgn}")):
            w = plane_along(P, N, a, pts_min=2500)
            if w is not None:
                walls[tag] = w
    # a wall seen from its floor line to its ceiling line gives the room height inside one frame
    for tag, w in walls.items():
        z = w["pts"] @ up
        lo, hi = np.percentile(z, [0.5, 99.5])
        if 1.9 < hi - lo < 4.2 and w["n_pts"] > 6000:
            prev = out.get("ceiling_height")
            if prev is None or w["n_pts"] > prev.get("n_pts", 0):
                out["ceiling_height"] = dict(value=float(hi - lo), source="vertical extent of one wall plane",
                                             n_pts=int(w["n_pts"]))
    spans = []
    for p, q, nm in (("a1", "a-1", "a"), ("b1", "b-1", "b")):
        A, B = walls.get(p), walls.get(q)
        if A and B and abs(float(A["n"] @ B["n"])) > 0.99:
            s = abs(A["d"] + B["d"])
            if 1.2 < s < 9.0:
                spans.append(dict(axis=nm, value=float(s), n_pts=int(min(A["n_pts"], B["n_pts"]))))
    if spans:
        out["spans"] = spans
    ops = []
    for tag, w in walls.items():
        if w["n_pts"] < 4000:
            continue
        wl = np.cross(up, w["n"])
        nl = np.linalg.norm(wl)
        if nl < 1e-6:
            continue
        wl /= nl
        pts = w["pts"]
        u = pts @ wl
        z = pts @ up
        band = (z > z.min() + 0.35) & (z < z.min() + 1.85)
        if band.sum() < 1500:
            continue
        lo, hi = np.percentile(u[band], [1, 99])
        nb = int((hi - lo) / 0.02)
        if nb < 30:
            continue
        prof, _ = np.histogram(u[band], bins=nb, range=(lo, hi))
        prof = np.convolve(prof.astype(float), np.ones(5) / 5, "same")
        plateau = float(np.median(prof[prof > 0])) if (prof > 0).any() else 0.0
        if plateau <= 0:
            continue
        empty = prof < 0.18 * plateau
        k = 0
        while k < nb:
            if not empty[k]:
                k += 1
                continue
            j = k
            while j + 1 < nb and empty[j + 1]:
                j += 1
            if k > 2 and j < nb - 3:
                width = (j - k + 1) * 0.02
                if min_gap <= width <= max_gap:
                    ops.append(dict(value=float(width), n_pts=int(prof[max(0, k - 10):k].sum())))
            k = j + 1
    if ops:
        out["openings"] = ops
    return out


def cluster(values, tol=0.06, min_n=3):
    v = np.sort(np.asarray(values, float))
    if len(v) == 0:
        return []
    groups, cur = [], [v[0]]
    for x in v[1:]:
        if x - cur[-1] <= tol:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)
    out = []
    for gth in groups:
        if len(gth) < min_n:
            continue
        a = np.asarray(gth)
        out.append(dict(value=round(float(np.median(a)), 4), n_frames=len(a),
                        spread_m=round(float(a.max() - a.min()), 4),
                        sigma_m=round(float(max(0.004, np.std(a) / np.sqrt(len(a)))), 4)))
    return sorted(out, key=lambda d: -d["n_frames"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--min-frames", type=int, default=4)
    args = ap.parse_args()
    frames = load_stray_capture(args.capture, target_fps=args.fps)
    heights, spans, ops = [], [], []
    per_frame = []
    for f in frames:
        m = measure_frame(f)
        if not m:
            continue
        rec = {"frame": f.frame_index}
        if "ceiling_height" in m:
            heights.append(m["ceiling_height"]["value"])
            rec["ceiling_height"] = round(m["ceiling_height"]["value"], 4)
        for s in m.get("spans", []):
            spans.append(s["value"])
            rec.setdefault("spans", []).append(round(s["value"], 4))
        for o in m.get("openings", []):
            ops.append(o["value"])
            rec.setdefault("openings", []).append(round(o["value"], 4))
        if len(rec) > 1:
            per_frame.append(rec)
    gt = {
        "capture": os.path.basename(args.capture.rstrip("/")),
        "method": "single-frame reference; see the docstring of scripts/make_reference_gt.py",
        "independent_of": ["poses", "pose graph", "drift correction", "multi-frame fusion",
                           "room segmentation", "all floorplan geometry modules"],
        "not_independent_of": ["the LiDAR sensor's own range accuracy"],
        "frames_used": len(per_frame),
        "frames_scanned": len(frames),
        "ceiling_height": cluster(heights, tol=0.05, min_n=args.min_frames),
        "wall_to_wall": cluster(spans, tol=0.07, min_n=args.min_frames),
        "opening_width": cluster(ops, tol=0.05, min_n=args.min_frames),
        "per_frame": per_frame,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        yaml.safe_dump(gt, fh, sort_keys=False)
    print(f"{args.capture}: {len(per_frame)}/{len(frames)} frames contributed")
    for k in ("ceiling_height", "wall_to_wall", "opening_width"):
        print(f"  {k}:")
        for c in gt[k][:8]:
            print(f"    {c['value']:.3f} m  (n={c['n_frames']} frames, spread {c['spread_m']*100:.1f} cm, sigma {c['sigma_m']*100:.2f} cm)")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
