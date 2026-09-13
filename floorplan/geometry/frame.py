"""Per-frame geometry: back-projection, gravity alignment, Manhattan frame, wall distances, openings.

Everything here works on ONE frame in its own camera frame (no poses needed). The LiDAR path
reuses the same primitives on the merged point cloud.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------- basic point-cloud helpers ----------

def backproject(depth: np.ndarray, K: np.ndarray, stride: int = 1) -> np.ndarray:
    """H x W x 3 points in the camera frame (OpenCV: x right, y down, z forward). NaN where depth<=0."""
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], -1).astype(np.float32)
    pts[z <= 0] = np.nan
    return pts


def normals_from_grid(P: np.ndarray) -> np.ndarray:
    """Normals via central differences on the organised grid. Unit vectors, NaN where undefined."""
    dx = np.full_like(P, np.nan)
    dy = np.full_like(P, np.nan)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / nn
    return n


def fit_plane(pts: np.ndarray):
    """Least-squares plane through pts (N x 3). Returns unit normal, d (n·x + d = 0), rms."""
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    d = -float(n @ c)
    rms = float(np.sqrt(np.mean((pts @ n + d) ** 2)))
    return n, d, rms


def ransac_plane(pts: np.ndarray, normals: np.ndarray | None, axis: np.ndarray, max_angle_deg: float,
                 thresh: float, iters: int, rng: np.random.Generator, min_inliers: int = 200):
    """RANSAC for a plane whose normal is within max_angle_deg of `axis`. pts: N x 3 (finite)."""
    if len(pts) < min_inliers:
        return None
    if normals is not None:
        cosang = np.abs(normals @ axis)
        cand = pts[cosang > np.cos(np.radians(max_angle_deg))]
    else:
        cand = pts
    if len(cand) < min_inliers:
        return None
    best = None
    for _ in range(iters):
        i = rng.integers(0, len(cand), 3)
        p = cand[i]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if abs(n @ axis) < np.cos(np.radians(max_angle_deg)):
            continue
        d = -(n @ p[0])
        dist = np.abs(cand @ n + d)
        cnt = int((dist < thresh).sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d)
    if best is None or best[0] < min_inliers:
        return None
    _, n, d = best
    inl = cand[np.abs(cand @ n + d) < thresh]
    n, d, rms = fit_plane(inl)
    if n @ axis < 0:
        n, d = -n, -d
    return n, d, rms, inl


# ---------- gravity + Manhattan ----------

@dataclass
class FrameGeom:
    R: np.ndarray                     # 3x3 rotation camera → gravity-aligned frame (z up)
    cam_height: float                 # camera above floor (m), nan if floor not seen
    ceil_above: float                 # ceiling above camera (m), nan if not seen
    yaw: float                        # Manhattan yaw (rad) of the +x axis in the aligned frame
    walls: dict = field(default_factory=dict)  # dir ('+x','-x','+y','-y') -> dict(dist, rms, n, pts_uv)
    floor_rms: float = np.nan
    n_points: int = 0
    P: np.ndarray | None = None       # H x W x 3 aligned, Manhattan-rotated points (camera at origin)
    depth_rel_err: float = 0.04       # relative depth uncertainty of the source (monocular default)


def rotation_from_up(up: np.ndarray) -> np.ndarray:
    """Rotation mapping unit vector `up` to +z, minimal rotation."""
    up = up / np.linalg.norm(up)
    z = np.array([0, 0, 1.0])
    v = np.cross(up, z)
    s = np.linalg.norm(v)
    c = float(up @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def rotz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def analyse_frame(P_cam: np.ndarray, *, up_hint: np.ndarray, seed: int = 0, depth_rel_err: float = 0.04,
                  cam_height_range=(0.6, 2.2), wall_axis_deg: float = 25.0,
                  R_override: np.ndarray | None = None, yaw_override: float | None = None) -> FrameGeom | None:
    """P_cam: H x W x 3 organised points in camera frame. up_hint: approximate up in camera frame.
    R_override: trusted camera→z-up rotation (from a pose); yaw_override: trusted Manhattan yaw (rad)."""
    rng = np.random.default_rng(seed)
    N = normals_from_grid(P_cam)
    ok = np.isfinite(P_cam).all(-1) & np.isfinite(N).all(-1)
    pts, nrm = P_cam[ok], N[ok]
    if len(pts) < 500:
        return None
    up_hint = up_hint / np.linalg.norm(up_hint)
    # floor: normal ~ up, plane below the camera
    thresh = 0.03
    floor = None
    for ang in (30.0, 45.0):
        r = ransac_plane(pts, nrm, up_hint, ang, thresh, 300, rng, min_inliers=300)
        if r is not None:
            n, d, rms, inl = r
            h = d  # signed distance of camera origin: n·0 + d = d ; camera above floor → d>0 when n points up
            if cam_height_range[0] <= h <= cam_height_range[1]:
                floor = (n, d, rms, len(inl))
                break
    ceil = None
    r = ransac_plane(pts, nrm, -up_hint, 30.0, thresh, 300, rng, min_inliers=300)
    if r is not None:
        n, d, rms, inl = r  # n points down; camera below ceiling → distance = d with n·x+d=0 → d>0
        if 0.5 <= d <= 2.5:
            ceil = (-n, -d, rms, len(inl))  # store with normal up: n_up·x + d_up = 0, d_up = -d
    if floor is None and ceil is None:
        return None
    if floor is not None and ceil is not None:
        up = floor[0] + ceil[0]
    else:
        up = (floor or ceil)[0]
    R = rotation_from_up(up) if R_override is None else R_override
    cam_h = float(floor[1]) if floor else np.nan
    ceil_above = float(-ceil[1]) if ceil else np.nan
    # aligned points (camera at origin, z up)
    Pa = P_cam @ R.T
    Na = N @ R.T
    okw = ok & (np.abs(Na[..., 2]) < np.sin(np.radians(wall_axis_deg)))  # near-vertical surfaces
    W = Pa[okw]
    NW = Na[okw]
    if len(W) < 300:
        yaw0 = float(yaw_override) if yaw_override is not None else 0.0
        Rm0 = rotz(-yaw0)
        g = FrameGeom(R=Rm0 @ R, cam_height=cam_h, ceil_above=ceil_above, yaw=yaw0,
                      floor_rms=floor[2] if floor else np.nan, n_points=len(pts), P=Pa @ Rm0.T,
                      depth_rel_err=depth_rel_err)
        return g
    # Manhattan yaw from azimuth histogram of wall normals (mod 90°)
    az = np.arctan2(NW[:, 1], NW[:, 0])
    a4 = np.mod(az, np.pi / 2)
    bins = 90
    hist, edges = np.histogram(a4, bins=bins, range=(0, np.pi / 2))
    hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5) / 5, mode="valid")
    k = int(np.argmax(hist))
    yaw = float((k + 0.5) * (np.pi / 2) / bins) if yaw_override is None else float(yaw_override)
    Rm = rotz(-yaw)
    Pm = Pa @ Rm.T
    Wm = W @ Rm.T
    NWm = NW @ Rm.T
    walls = {}
    for name, axis in (("+x", np.array([1.0, 0, 0])), ("-x", np.array([-1.0, 0, 0])),
                       ("+y", np.array([0, 1.0, 0])), ("-y", np.array([0, -1.0, 0]))):
        # wall in direction +axis has normal pointing back toward the camera: normal ≈ -axis
        sel = (NWm @ (-axis)) > np.cos(np.radians(20.0))
        if sel.sum() < 150:
            continue
        q = Wm[sel]
        proj = q @ axis  # distance along the axis (positive = in front along that direction)
        proj = proj[proj > 0.3]
        if len(proj) < 150:
            continue
        # farthest strongly supported peak = the room wall (nearer peaks are furniture)
        h, e = np.histogram(proj, bins=int(np.ceil(proj.max() / 0.05)) + 1, range=(0, proj.max() + 0.05))
        h = np.convolve(h, np.ones(3) / 3, mode="same")
        peak_min = max(60, 0.15 * h.max())
        idx = np.where(h >= peak_min)[0]
        if len(idx) == 0:
            continue
        # group contiguous bins, pick the farthest group
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        far = groups[-1]
        lo, hi = e[far[0]], e[far[-1] + 1]
        inl = q[(proj_all := (q @ axis)) >= lo - 0.05]
        inl = inl[(inl @ axis) <= hi + 0.05]
        if len(inl) < 100:
            continue
        n, d, rms = fit_plane(inl)
        if n @ (-axis) < 0:
            n, d = -n, -d
        # enforce axis-aligned normal (Manhattan): distance = mean projection of inliers
        dist = float(np.median(inl @ axis))
        # lateral extent of the observed wall patch (for coverage / corner visibility)
        lat_axis = np.array([0, 1.0, 0]) if abs(axis[0]) > 0.5 else np.array([1.0, 0, 0])
        lat = inl @ lat_axis
        walls[name] = dict(dist=dist, rms=float(rms), n=int(len(inl)), lat_min=float(np.percentile(lat, 2)),
                           lat_max=float(np.percentile(lat, 98)), z_min=float(np.percentile(inl[:, 2], 2)),
                           z_max=float(np.percentile(inl[:, 2], 98)))
    g = FrameGeom(R=Rm @ R, cam_height=cam_h, ceil_above=ceil_above, yaw=yaw, walls=walls,
                  floor_rms=floor[2] if floor else np.nan, n_points=len(pts), P=Pm, depth_rel_err=depth_rel_err)
    return g


# ---------- openings on a wall from one frame ----------

DIRS = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
        "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0])}


def view_dir_xy(g: "FrameGeom") -> np.ndarray:
    """Camera optical axis projected on the floor, in the frame's Manhattan frame (unit 2-vector)."""
    v = g.R @ np.array([0, 0, 1.0])
    v = v[:2]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, 1.0])


def rotate_geom(g: "FrameGeom", k: int) -> "FrameGeom":
    """Rotate the frame's Manhattan frame by k*90° (counter-clockwise, seen from above)."""
    k %= 4
    if k == 0:
        return g
    Rk = rotz(k * np.pi / 2)
    names = ["+x", "+y", "-x", "-y"]  # CCW order
    walls = {}
    for d, w in g.walls.items():
        nd = names[(names.index(d) + k) % 4]
        walls[nd] = dict(w)
    return FrameGeom(R=Rk @ g.R, cam_height=g.cam_height, ceil_above=g.ceil_above, yaw=g.yaw, walls=walls,
                     floor_rms=g.floor_rms, n_points=g.n_points, P=(g.P @ Rk.T) if g.P is not None else None,
                     depth_rel_err=g.depth_rel_err)


def best_rotation(g: "FrameGeom", expected_view_xy: np.ndarray) -> int:
    """k in 0..3 such that rotating the frame by k*90° makes its view direction closest to expected."""
    v = view_dir_xy(g)
    e = np.asarray(expected_view_xy, float); e = e / np.linalg.norm(e)
    best, bk = -2.0, 0
    for k in range(4):
        c, s_ = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        vr = np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])
        if vr @ e > best:
            best, bk = vr @ e, k
    return bk


def wall_uz(P: np.ndarray, direction: str, dist: float, cam_h: float):
    """Split a frame's points w.r.t. the wall plane at `dist` along `direction`.

    Returns (on_u, on_z, cross_u, cross_z, lo, hi): on-wall points and, for points BEHIND the wall,
    the (u, z) where the camera ray crosses the wall plane. u is the lateral coordinate in the frame's
    Manhattan frame (x for y-walls, y for x-walls), z is height above the floor.
    """
    axis = DIRS[direction]
    lat_axis = np.array([0, 1.0, 0]) if abs(axis[0]) > 0.5 else np.array([1.0, 0, 0])
    ok = np.isfinite(P).all(-1)
    Q = P[ok]
    a = Q @ axis
    sel = a > 0.3
    Q, a = Q[sel], a[sel]
    ch = cam_h if np.isfinite(cam_h) else 1.4
    on = np.abs(a - dist) <= 0.12
    behind = a > dist + 0.25
    on_u, on_z = Q[on] @ lat_axis, Q[on][:, 2] + ch
    t = dist / a[behind]
    cross = Q[behind] * t[:, None]
    cross_u, cross_z = cross @ lat_axis, cross[:, 2] + ch
    return on_u, on_z, cross_u, cross_z


def openings_from_uz(on_u, on_z, cross_u, cross_z, *, bin_m: float = 0.05, min_w: float = 0.55,
                     max_w: float = 1.4, min_support: int = 6):
    """Holes in a wall from on-wall (u,z) samples and ray crossings (u,z) of points behind the wall."""
    if len(on_u) < 100:
        return []
    lo, hi = np.percentile(on_u, 1), np.percentile(on_u, 99)
    nb = int(np.ceil((hi - lo) / bin_m)) + 1
    if nb < 4:
        return []
    zb = 0.10
    nz = int(np.ceil(3.2 / zb))
    rng = [[lo, lo + nb * bin_m], [0, nz * zb]]
    Go, _, _ = np.histogram2d(on_u, on_z, bins=[nb, nz], range=rng)
    Gb, _, _ = np.histogram2d(cross_u, cross_z, bins=[nb, nz], range=rng)
    out = []

    def runs(mask):
        i = 0
        while i < nb:
            if not mask[i]:
                i += 1; continue
            j = i
            while j + 1 < nb and mask[j + 1]:
                j += 1
            yield i, j
            i = j + 1

    def edge_refine(i, j):
        # fractional edges from the balance of behind vs on-wall in the boundary bins
        u0 = lo + i * bin_m; u1 = lo + (j + 1) * bin_m
        return u0, u1

    zsel = slice(int(0.3 / zb), int(1.6 / zb))
    b = Gb[:, zsel].sum(1); o = Go[:, zsel].sum(1)
    hole = (b >= min_support) & (b > 2.0 * o)
    for i, j in runs(hole):
        width = (j - i + 1) * bin_m
        u0, u1 = edge_refine(i, j)
        if min_w <= width <= max_w and (Go[:i, zsel].sum() + Go[j + 1:, zsel].sum()) > 20:
            col = Gb[i:j + 1].sum(0)
            zs = np.where(col >= 2)[0]
            top = float((zs.max() + 1) * zb) if len(zs) else np.nan
            bottom = float(zs.min() * zb) if len(zs) else np.nan
            typ = "door" if (np.isfinite(bottom) and bottom <= 0.35) else "window"
            out.append(dict(type=typ, u0=float(u0), u1=float(u1), width=float(width), top=top, bottom=bottom,
                            support=int(b[i:j + 1].sum())))
    zsel2 = slice(int(0.9 / zb), int(1.9 / zb))
    b2 = Gb[:, zsel2].sum(1); o2 = Go[:, zsel2].sum(1)
    below = Go[:, :int(0.7 / zb)].sum(1)
    hole2 = (b2 >= min_support) & (b2 > 2.0 * o2) & (below >= 2)
    for i, j in runs(hole2):
        width = (j - i + 1) * bin_m
        u0, u1 = edge_refine(i, j)
        if 0.4 <= width <= 3.0 and not any(abs(0.5 * (u0 + u1) - 0.5 * (q["u0"] + q["u1"])) < 0.5 for q in out):
            col = Gb[i:j + 1].sum(0)
            zs = np.where(col >= 2)[0]
            top = float((zs.max() + 1) * zb) if len(zs) else np.nan
            bottom = float(zs.min() * zb) if len(zs) else np.nan
            out.append(dict(type="window", u0=float(u0), u1=float(u1), width=float(width), top=top, bottom=bottom,
                            support=int(b2[i:j + 1].sum())))
    return out


def wall_openings(P: np.ndarray, direction: str, dist: float, cam_h: float, **kw):
    on_u, on_z, cross_u, cross_z = wall_uz(P, direction, dist, cam_h)
    return openings_from_uz(on_u, on_z, cross_u, cross_z, **kw)
