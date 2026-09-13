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


def normals_from_grid(P: np.ndarray, k: int = 4) -> np.ndarray:
    """Normals via central differences with a ±k pixel stencil (noise-robust). Unit vectors, NaN where undefined."""
    dx = np.full_like(P, np.nan)
    dy = np.full_like(P, np.nan)
    dx[:, k:-k] = P[:, 2 * k:] - P[:, :-2 * k]
    dy[k:-k, :] = P[2 * k:, :] - P[:-2 * k, :]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / nn
    # orient every normal toward the camera (origin): the visible side satisfies n·p < 0
    flip = np.sum(n * P, axis=-1) > 0
    n[flip] *= -1
    return n


def fit_plane(pts: np.ndarray):
    """Least-squares plane through pts (N x 3). Returns unit normal, d (n·x + d = 0), rms."""
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    d = -float(n @ c)
    rms = float(np.sqrt(np.mean((pts @ n + d) ** 2)))
    return n, d, rms


MAX_RANSAC_PTS = 30000


def ransac_plane(pts: np.ndarray, normals: np.ndarray | None, axis: np.ndarray, max_angle_deg: float,
                 thresh: float, iters: int, rng: np.random.Generator, min_inliers: int = 200):
    """RANSAC for a plane whose normal is within max_angle_deg of `axis`. pts: N x 3 (finite).

    The candidate set is capped: past about thirty thousand points the inlier fraction is already
    estimated to well under a millimetre and every extra point is pure cost.
    """
    if len(pts) < min_inliers:
        return None
    if normals is not None:
        cosang = normals @ axis          # normals are oriented toward the camera, so the sign matters
        cand = pts[cosang > np.cos(np.radians(max_angle_deg))]
    else:
        cand = pts
    if len(cand) < min_inliers:
        return None
    if len(cand) > MAX_RANSAC_PTS:
        cand = cand[rng.choice(len(cand), MAX_RANSAC_PTS, replace=False)]
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
        peak_min = max(60, 0.4 * h.max(), 0.02 * len(pts))
        idx = np.where(h >= peak_min)[0]
        if len(idx) == 0:
            continue
        # group contiguous bins, pick the farthest group
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        far = groups[-1]
        lo, hi = e[far[0]], e[far[-1] + 1]
        inl = q[(_proj_all := (q @ axis)) >= lo - 0.05]
        inl = inl[(inl @ axis) <= hi + 0.05]
        if len(inl) < max(100, 0.02 * len(pts)):
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


def view_dir_xy(g: FrameGeom) -> np.ndarray:
    """Camera optical axis projected on the floor, in the frame's Manhattan frame (unit 2-vector)."""
    v = g.R @ np.array([0, 0, 1.0])
    v = v[:2]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, 1.0])


def rotate_geom(g: FrameGeom, k: int) -> FrameGeom:
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


def best_rotation(g: FrameGeom, expected_view_xy: np.ndarray) -> int:
    """k in 0..3 such that rotating the frame by k*90° makes its view direction closest to expected."""
    v = view_dir_xy(g)
    e = np.asarray(expected_view_xy, float)
    e = e / np.linalg.norm(e)
    best, bk = -2.0, 0
    for k in range(4):
        c, s_ = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        vr = np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])
        if vr @ e > best:
            best, bk = vr @ e, k
    return bk


def wall_uz(P: np.ndarray, direction: str, dist: float, cam_h: float):
    """Split a frame's points w.r.t. the wall plane at `dist` along `direction`.

    Returns dict(on_u, on_z, cross_u, cross_z, near_u, near_z): on-wall points, ray crossings (u, z) of
    points BEHIND the wall plane, and NEAR points (occluders in front of the wall) projected along
    their ray onto the wall plane. u is the lateral coordinate in the frame's Manhattan frame
    (x for y-walls, y for x-walls) relative to the camera; z is height above the floor.
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
    near = (a < dist - 0.25) & (a > 0.3)
    out = {"on_u": Q[on] @ lat_axis, "on_z": Q[on][:, 2] + ch}
    t = dist / a[behind]
    cross = Q[behind] * t[:, None]
    out["cross_u"], out["cross_z"] = cross @ lat_axis, cross[:, 2] + ch
    t = dist / a[near]
    npr = Q[near] * t[:, None]
    out["near_u"], out["near_z"] = npr @ lat_axis, npr[:, 2] + ch
    return out


def openings_from_uz(uz: dict, *, bin_m: float = 0.05, min_w: float = 0.55, max_w: float = 1.4,
                     min_support: int = 6, u_range: tuple | None = None):
    """Holes in a wall from (u, z) samples: on-wall points, ray crossings of points behind the wall,
    and near occluders projected onto the wall. Doors: floor-touching gaps; windows: elevated gaps,
    detected from crossings (monocular/LiDAR through glass) or from a no-return hole (LiDAR glass)."""
    on_u, on_z = uz["on_u"], uz["on_z"]
    if len(on_u) < 100:
        return []
    if u_range is None:
        lo, hi = np.percentile(on_u, 1), np.percentile(on_u, 99)
    else:
        lo, hi = u_range
    nb = int(np.ceil((hi - lo) / bin_m)) + 1
    if nb < 4:
        return []
    zb = 0.10
    nz = int(np.ceil(3.2 / zb))
    rng = [[lo, lo + nb * bin_m], [0, nz * zb]]
    Go, _, _ = np.histogram2d(on_u, on_z, bins=[nb, nz], range=rng)
    Gb, _, _ = np.histogram2d(uz["cross_u"], uz["cross_z"], bins=[nb, nz], range=rng)
    Gn, _, _ = np.histogram2d(uz["near_u"], uz["near_z"], bins=[nb, nz], range=rng)
    col_ref = np.median(Go.sum(1)[Go.sum(1) > 0]) if (Go.sum(1) > 0).any() else 1.0   # typical on-wall column
    out = []

    def runs(mask):
        i = 0
        while i < nb:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < nb and mask[j + 1]:
                j += 1
            yield i, j
            i = j + 1

    def halfmax_edges(i, j, o):
        """Sub-bin edges from where the on-wall column profile crosses half of its plateau."""
        left = o[max(0, i - 14):max(0, i - 3)]
        right = o[j + 4:j + 15]
        pl = np.median(left) if len(left) else np.nan
        pr = np.median(right) if len(right) else np.nan
        u0 = lo + i * bin_m
        u1 = lo + (j + 1) * bin_m
        if np.isfinite(pl) and pl > 0:
            k = i
            while k - 1 >= 0 and o[k - 1] < 0.5 * pl:
                k -= 1
            if k - 1 >= 0 and o[k - 1] != o[k]:
                frac = (0.5 * pl - o[k]) / (o[k - 1] - o[k])      # 0 at column k, 1 at column k-1
                u0 = lo + k * bin_m + bin_m * (0.5 - frac)
            else:
                u0 = lo + k * bin_m
        if np.isfinite(pr) and pr > 0:
            k = j
            while k + 1 < nb and o[k + 1] < 0.5 * pr:
                k += 1
            if k + 1 < nb and o[k + 1] != o[k]:
                frac = (0.5 * pr - o[k]) / (o[k + 1] - o[k])
                u1 = lo + (k + 1) * bin_m - bin_m * (0.5 - frac)
            else:
                u1 = lo + (k + 1) * bin_m
        return u0, u1

    def refine(i, j, b, o):
        return halfmax_edges(i, j, o)

    def emit(i, j, b, o, band_lo, force_type=None):
        u0, u1 = refine(i, j, b, o)
        width = u1 - u0
        col = Gb[i:j + 1].sum(0)
        zs = np.where(col >= 2)[0]
        top = float((zs.max() + 1) * zb) if len(zs) else np.nan
        bottom = float(zs.min() * zb) if len(zs) else np.nan
        below_on = Go[i:j + 1, :int(0.6 / zb)].sum() / max(j - i + 1, 1)
        typ = force_type or ("window" if below_on > 0.25 * col_ref * (0.6 / 3.2) * 4 else "door")
        if typ == "door":
            bottom = 0.0
        return dict(type=typ, u0=float(u0), u1=float(u1), width=float(width), top=top, bottom=bottom,
                    support=int(b[i:j + 1].sum()))

    # 1) floor-touching gaps with crossings: doors (or passages)
    zsel = slice(int(0.3 / zb), int(1.6 / zb))
    b = Gb[:, zsel].sum(1)
    o = Go[:, zsel].sum(1)
    n = Gn[:, zsel].sum(1)
    hole = (b >= min_support) & (b > 2.0 * o)
    for i, j in runs(hole):
        d = emit(i, j, b, o, 0.3)
        if min_w <= d["width"] <= max_w * 1.6 and (Go[:i, zsel].sum() + Go[j + 1:, zsel].sum()) > 20:
            out.append(d)
    # 2) elevated gaps: windows from crossings
    zsel2 = slice(int(0.9 / zb), int(1.9 / zb))
    b2 = Gb[:, zsel2].sum(1)
    o2 = Go[:, zsel2].sum(1)
    n2 = Gn[:, zsel2].sum(1)
    below = Go[:, :int(0.7 / zb)].sum(1)
    hole2 = (b2 >= min_support) & (b2 > 2.0 * o2) & (below >= 2)
    for i, j in runs(hole2):
        d = emit(i, j, b2, o2, 0.9, force_type="window")
        if 0.4 <= d["width"] <= 3.0 and not any(abs(0.5 * (d["u0"] + d["u1"]) - 0.5 * (q["u0"] + q["u1"])) < 0.5 for q in out):
            out.append(d)
    # 3) elevated no-return gaps (LiDAR through glass): no on-wall, no near occluder, wall below and beside
    quiet = (o2 <= 0.05 * col_ref) & (n2 <= 1) & (below >= 0.1 * col_ref)
    for i, j in runs(quiet):
        width = (j - i + 1) * bin_m
        if not (0.4 <= width <= 3.0):
            continue
        side_ok = (i > 0 and o2[i - 1] > 0.1 * col_ref) or (j + 1 < nb and o2[j + 1] > 0.1 * col_ref)
        if not side_ok:
            continue
        u0, u1 = lo + i * bin_m, lo + (j + 1) * bin_m
        if any(abs(0.5 * (u0 + u1) - 0.5 * (q["u0"] + q["u1"])) < 0.5 for q in out):
            continue
        colz = Go[i:j + 1].sum(0)
        zs = np.where(colz <= 0.05 * col_ref / nz * 4)[0]
        zs = zs[(zs >= int(0.5 / zb)) & (zs < int(2.4 / zb))]
        bottom = float(zs.min() * zb) if len(zs) else 0.9
        top = float((zs.max() + 1) * zb) if len(zs) else 2.0
        out.append(dict(type="window", u0=float(u0), u1=float(u1), width=float(width), top=top, bottom=bottom,
                        support=int(below[i:j + 1].sum()), no_return=True))
    # 4) floor-touching no-return gaps (door to the outside / unscanned space): no wall, no occluder,
    #    wall present on both sides and above the door head
    quiet_d = (o <= 0.05 * col_ref) & (n <= 1) & (b <= 1)
    for i, j in runs(quiet_d):
        width = (j - i + 1) * bin_m
        if not (min_w <= width <= max_w):
            continue
        side_l = o[max(0, i - 6):i].max() if i > 0 else 0
        side_r = o[j + 1:j + 7].max() if j + 1 < nb else 0
        if not (side_l > 0.12 * col_ref and side_r > 0.12 * col_ref):
            continue
        above = Go[i:j + 1, int(2.2 / zb):int(3.0 / zb)].sum() / max(j - i + 1, 1)
        u0, u1 = halfmax_edges(i, j, o)
        width = u1 - u0
        if not (min_w <= width <= max_w):
            continue
        if any(abs(0.5 * (u0 + u1) - 0.5 * (q["u0"] + q["u1"])) < 0.5 for q in out):
            continue
        colz = Go[i:j + 1].sum(0)
        zs = np.where(colz > 0.05 * col_ref / nz * 4)[0]
        zs = zs[zs >= int(1.6 / zb)]
        top = float(zs.min() * zb) if len(zs) else 2.0
        out.append(dict(type="door", u0=float(u0), u1=float(u1), width=float(width), top=top, bottom=0.0,
                        support=int(o[i - 1] + o[j + 1]), no_return=True, weak=(above < 0.05 * col_ref)))
    if u_range is not None:
        out = [d for d in out if d["u1"] > lo + 0.05 and d["u0"] < hi - 0.05]
    return out


def wall_openings(P: np.ndarray, direction: str, dist: float, cam_h: float, **kw):
    return openings_from_uz(wall_uz(P, direction, dist, cam_h), **kw)
