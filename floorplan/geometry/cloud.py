"""Global point-cloud geometry for the LiDAR tier.

Per-frame plane fitting is fragile in furnished rooms: a wardrobe face outvotes the wall behind it,
and a single frame rarely sees two opposite walls. With poses we can do better - accumulate every
frame into one gravity-aligned cloud and solve the layout once, globally.

Pipeline: build -> level (floor plane) -> Manhattan yaw -> floor/ceiling -> wall lines ->
occupancy -> room segmentation (watershed on free space) -> polygons -> openings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from floorplan.core.types import Frame
from floorplan.geometry.frame import backproject, fit_plane, normals_from_grid, rotation_from_up, rotz

MAX_RANGE = 5.0      # LiDAR beyond this is noisy and mostly through-window returns
MIN_RANGE = 0.25


@dataclass
class Cloud:
    P: np.ndarray                  # N x 3, world, z up, yaw-aligned
    N: np.ndarray                  # N x 3 unit normals, oriented toward the observing camera
    fi: np.ndarray                 # N int32, index into `cams`
    cams: np.ndarray               # F x 3 camera positions in the same frame
    yaw: float = 0.0               # Manhattan yaw removed from the world (rad)
    tilt: np.ndarray = field(default_factory=lambda: np.eye(3))
    floor_z: float = 0.0
    ceil_z: float = float("nan")
    floor_sigma: float = 0.0
    ceil_sigma: float = float("nan")
    ceil_support: int = 0

    @property
    def vertical(self) -> np.ndarray:
        return np.abs(self.N[:, 2]) > 0.9

    @property
    def wallish(self) -> np.ndarray:
        return np.abs(self.N[:, 2]) < 0.35


def build_cloud(frames: list[Frame], stride: int = 2, conf_min: int = 2) -> Cloud:
    P, N, FI, cams = [], [], [], []
    for i, f in enumerate(frames):
        cams.append(f.pose[:3, 3])
        p = backproject(f.depth, f.K, stride=stride)
        n = normals_from_grid(p, k=3)
        c = f.conf[::stride, ::stride] if f.conf is not None else None
        d = f.depth[::stride, ::stride]
        ok = np.isfinite(p).all(-1) & np.isfinite(n).all(-1) & (d > MIN_RANGE) & (d < MAX_RANGE)
        if c is not None:
            ok &= c >= conf_min
        if not ok.any():
            continue
        R = f.pose[:3, :3]
        P.append((p[ok] @ R.T + f.pose[:3, 3]).astype(np.float32))
        N.append((n[ok] @ R.T).astype(np.float32))
        FI.append(np.full(int(ok.sum()), i, np.int32))
    if not P:
        raise ValueError("no usable depth in this capture")
    cloud = Cloud(P=np.concatenate(P), N=np.concatenate(N), fi=np.concatenate(FI),
                  cams=np.asarray(cams, np.float64))
    _level(cloud)
    _manhattan(cloud)
    _floor_ceiling(cloud)
    return cloud


def _level(cloud: Cloud, band: float = 0.06) -> None:
    """Remove residual gravity tilt by fitting the dominant horizontal plane (the floor)."""
    v = cloud.vertical & (cloud.N[:, 2] > 0.9)
    if v.sum() < 2000:
        return
    z = cloud.P[v, 2]
    h, e = np.histogram(z, bins=600, range=(float(z.min()), float(z.max()) + 1e-3))
    zf = e[np.argmax(h)]
    sel = np.where(v)[0][np.abs(z - zf) < band]
    if len(sel) < 2000:
        return
    idx = sel if len(sel) < 200000 else np.random.default_rng(0).choice(sel, 200000, replace=False)
    n, d, rms = fit_plane(cloud.P[idx].astype(np.float64))
    if n[2] < 0:
        n, d = -n, -d
    if abs(n[2]) < 0.95:          # implausible: leave the ARKit gravity alone
        return
    R = rotation_from_up(n)
    cloud.tilt = R
    cloud.P = (cloud.P @ R.T).astype(np.float32)
    cloud.N = (cloud.N @ R.T).astype(np.float32)
    cloud.cams = cloud.cams @ R.T


def _manhattan(cloud: Cloud, bins: int = 360) -> None:
    w = cloud.wallish
    if w.sum() < 1000:
        return
    az = np.arctan2(cloud.N[w, 1], cloud.N[w, 0])
    a4 = np.mod(az, np.pi / 2)
    h, _ = np.histogram(a4, bins=bins, range=(0, np.pi / 2))
    h = np.convolve(np.r_[h[-4:], h, h[:4]], np.ones(9) / 9, mode="valid")
    yaw = float((np.argmax(h) + 0.5) * (np.pi / 2) / bins)
    cloud.yaw = yaw
    R = rotz(-yaw)
    cloud.P = (cloud.P @ R.T).astype(np.float32)
    cloud.N = (cloud.N @ R.T).astype(np.float32)
    cloud.cams = cloud.cams @ R.T


def _peak(z: np.ndarray, lo: float, hi: float, res: float = 0.01, band: float = 0.04):
    if len(z) < 200:
        return None
    h, e = np.histogram(z, bins=int((hi - lo) / res) + 1, range=(lo, hi))
    if h.max() < 200:
        return None
    m = e[np.argmax(h)] + res / 2
    sel = np.abs(z - m) < band
    if sel.sum() < 200:
        return None
    v = z[sel]
    return float(np.median(v)), int(sel.sum()), float(np.std(v))


def _floor_ceiling(cloud: Cloud, min_height: float = 1.9, max_height: float = 4.2) -> None:
    up = cloud.vertical & (cloud.N[:, 2] > 0.9)
    dn = cloud.vertical & (cloud.N[:, 2] < -0.9)
    zs = cloud.P[:, 2]
    lo, hi = float(zs.min()) - 0.1, float(zs.max()) + 0.1
    f = _peak(cloud.P[up, 2], lo, hi)
    if f is None:                       # no floor: fall back to the lowest camera minus a carry height
        cloud.floor_z, cloud.floor_sigma = float(cloud.cams[:, 2].min() - 1.45), 0.15
    else:
        cloud.floor_z, cloud.floor_sigma = f[0], max(0.003, f[2] / np.sqrt(f[1]))
    zc = cloud.P[dn, 2]
    zc = zc[zc > cloud.floor_z + min_height]
    zc = zc[zc < cloud.floor_z + max_height]
    c = _peak(zc, cloud.floor_z + min_height, cloud.floor_z + max_height)
    if c is not None:
        cloud.ceil_z, cloud.ceil_sigma, cloud.ceil_support = c[0], max(0.004, c[2] / np.sqrt(c[1])), c[1]
    else:
        cloud.ceil_z, cloud.ceil_sigma, cloud.ceil_support = float("nan"), float("nan"), 0


# --------------------------------------------------------------------------- occupancy & rooms

@dataclass
class Grid:
    res: float
    x0: float
    y0: float
    wall: np.ndarray      # counts of wall-ish points in the waist band
    zspan: np.ndarray     # vertical extent (m) of those points: furniture edges are short, walls tall
    floor: np.ndarray     # counts of floor points
    ceil: np.ndarray      # counts of ceiling points
    free: np.ndarray      # bool: carved free space
    shape: tuple = ()

    def to_world(self, i, j):
        return self.x0 + (np.asarray(i) + 0.5) * self.res, self.y0 + (np.asarray(j) + 0.5) * self.res

    def to_cell(self, x, y):
        return (np.floor((np.asarray(x) - self.x0) / self.res).astype(int),
                np.floor((np.asarray(y) - self.y0) / self.res).astype(int))


def _carve(cloud: Cloud, g_shape, x0, y0, res, waist, n_az: int = 288, step: float = 0.06) -> np.ndarray:
    """Free space by 2-D ray carving: every LiDAR return says the space in front of it was empty.

    Per frame the returns are binned by azimuth; the nearest return in each bin bounds the free wedge.
    Bins with no return in the waist band are carved to the sensor range, which is what opens a
    doorway the walker never stepped through.
    """
    nx, ny = g_shape
    acc = np.zeros((nx, ny), np.float32)
    fz = cloud.floor_z
    P, cams = cloud.P, cloud.cams
    band = (P[:, 2] > fz + waist[0]) & (P[:, 2] < fz + waist[1])
    order = np.argsort(cloud.fi[band], kind="stable")
    fis = cloud.fi[band][order]
    XY = P[band][:, :2][order]
    bounds = np.searchsorted(fis, np.arange(len(cams) + 1))
    two_pi = 2 * np.pi
    for k in range(len(cams)):
        a, b = bounds[k], bounds[k + 1]
        if b - a < 50:
            continue
        c = cams[k, :2]
        d = XY[a:b] - c
        r = np.hypot(d[:, 0], d[:, 1])
        az = ((np.arctan2(d[:, 1], d[:, 0]) + two_pi) % two_pi)
        bin_i = (az / two_pi * n_az).astype(np.int32).clip(0, n_az - 1)
        rmin = np.full(n_az, np.inf, np.float32)
        np.minimum.at(rmin, bin_i, r.astype(np.float32))
        seen = np.isfinite(rmin)
        if seen.sum() < 8:
            continue
        # unobserved wedges inherit the neighbouring range so we never carve through a wall we simply
        # did not sample; observed wedges are carved to 12 cm short of the return.
        reach = np.where(seen, rmin - 0.12, 0.0)
        reach = np.clip(reach, 0.0, MAX_RANGE)
        ang = (np.arange(n_az) + 0.5) / n_az * two_pi
        nstep = int(np.ceil(reach.max() / step)) + 1
        ts = np.arange(1, nstep + 1) * step
        keep = ts[None, :] <= reach[:, None]
        if not keep.any():
            continue
        px = c[0] + np.cos(ang)[:, None] * ts[None, :]
        py = c[1] + np.sin(ang)[:, None] * ts[None, :]
        ix = np.floor((px[keep] - x0) / res).astype(np.int32)
        iy = np.floor((py[keep] - y0) / res).astype(np.int32)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.add.at(acc, (ix[ok], iy[ok]), 1.0)
    return acc


def build_grid(cloud: Cloud, res: float = 0.05, waist=(0.35, 1.9), carve: bool = True) -> Grid:
    fz = cloud.floor_z
    P = cloud.P
    wall_sel = cloud.wallish & (P[:, 2] > fz + waist[0]) & (P[:, 2] < fz + waist[1])
    floor_sel = cloud.vertical & (cloud.N[:, 2] > 0.9) & (np.abs(P[:, 2] - fz) < 0.08)
    ceil_sel = np.zeros(len(P), bool)
    if np.isfinite(cloud.ceil_z):
        ceil_sel = cloud.vertical & (cloud.N[:, 2] < -0.9) & (np.abs(P[:, 2] - cloud.ceil_z) < 0.10)
    x0 = float(P[:, 0].min()) - 0.6
    y0 = float(P[:, 1].min()) - 0.6
    nx = int((P[:, 0].max() + 0.6 - x0) / res) + 2
    ny = int((P[:, 1].max() + 0.6 - y0) / res) + 2

    def cells(sel):
        ix = np.floor((P[sel, 0] - x0) / res).astype(np.int32).clip(0, nx - 1)
        iy = np.floor((P[sel, 1] - y0) / res).astype(np.int32).clip(0, ny - 1)
        return ix, iy

    def hist(sel):
        g = np.zeros((nx, ny), np.float32)
        if sel.any():
            ix, iy = cells(sel)
            np.add.at(g, (ix, iy), 1.0)
        return g

    wall = hist(wall_sel)
    zspan = np.zeros((nx, ny), np.float32)
    if wall_sel.any():
        ix, iy = cells(wall_sel)
        z = P[wall_sel, 2]
        zmax = np.full((nx, ny), -1e9, np.float32)
        zmin = np.full((nx, ny), 1e9, np.float32)
        np.maximum.at(zmax, (ix, iy), z)
        np.minimum.at(zmin, (ix, iy), z)
        m = zmax > -1e8
        zspan[m] = zmax[m] - zmin[m]
    g = Grid(res=res, x0=x0, y0=y0, wall=wall, zspan=zspan, floor=hist(floor_sel), ceil=hist(ceil_sel),
             free=np.zeros((nx, ny), bool))
    g.shape = (nx, ny)

    free = np.zeros((nx, ny), bool)
    if carve:
        acc = _carve(cloud, g.shape, x0, y0, res, waist)
        free |= acc >= 3
    if (g.floor > 0).any():
        free |= g.floor >= max(3.0, float(np.quantile(g.floor[g.floor > 0], 0.25)))
    if (g.ceil > 0).any():
        free |= g.ceil >= max(3.0, float(np.quantile(g.ceil[g.ceil > 0], 0.25)))
    ci, cj = g.to_cell(cloud.cams[:, 0], cloud.cams[:, 1])
    ok = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
    path = np.zeros((nx, ny), bool)
    path[ci[ok], cj[ok]] = True
    free |= ndimage.binary_dilation(path, np.ones((3, 3)), iterations=max(1, int(0.2 / res)))
    free = ndimage.binary_closing(free, np.ones((3, 3)))
    g.free = ndimage.binary_opening(free, np.ones((3, 3)))
    return g


def wall_mask(g: Grid, min_zspan: float = 0.55, quantile: float = 0.55, min_count: float = 6.0) -> np.ndarray:
    """A wall cell carries a tall column of near-vertical returns. Furniture edges are short."""
    nz = g.wall[g.wall > 0]
    thr = max(min_count, float(np.quantile(nz, quantile)) if len(nz) else min_count)
    return (g.wall >= thr) & (g.zspan >= min_zspan)


def _split_large(lab: np.ndarray, D: np.ndarray, free: np.ndarray, res: float, max_area: float) -> np.ndarray:
    """A room bigger than `max_area` is usually two spaces joined through a wide opening. Re-flood it
    from tighter cores so the watershed gets another chance to find the neck."""
    out = lab.copy()
    nxt = int(out.max()) + 1
    for r in range(1, int(lab.max()) + 1):
        m = lab == r
        if m.sum() * res ** 2 <= max_area:
            continue
        Dm = D * m
        for core_r in (1.45, 1.25, 1.1):
            cores, n = ndimage.label(Dm > core_r)
            sizes = np.asarray(ndimage.sum(np.ones_like(cores), cores, range(1, n + 1))) if n else np.array([])
            keep = [i + 1 for i, s in enumerate(sizes) if s * res ** 2 > 0.6]
            if len(keep) >= 2:
                relab = np.zeros(n + 1, np.int32)
                for k, l in enumerate(keep, start=1):
                    relab[l] = k
                sub = _priority_flood(relab[cores].astype(np.int32), Dm, m)
                for k in range(2, int(sub.max()) + 1):
                    out[(sub == k)] = nxt
                    nxt += 1
                break
    return out


def segment_rooms(g: Grid, min_area_m2: float = 1.5, max_area_m2: float = 26.0) -> np.ndarray:
    """Watershed the carved free space; doorway necks become the borders between rooms."""
    walls = wall_mask(g)
    free = g.free & ~walls
    free = ndimage.binary_opening(free, np.ones((3, 3)))
    if not free.any():
        return np.zeros(g.shape, np.int32)
    D = ndimage.distance_transform_edt(free) * g.res
    for core_r in (1.1, 0.95, 0.8, 0.65, 0.5, 0.4):
        cores, n = ndimage.label(D > core_r)
        if n == 0:
            continue
        sizes = np.asarray(ndimage.sum(np.ones_like(cores), cores, range(1, n + 1)))
        keep = [i + 1 for i, s in enumerate(sizes) if s * g.res ** 2 > 0.3]
        if len(keep) >= 2 or core_r <= 0.4:
            relab = np.zeros(n + 1, np.int32)
            for k, lab in enumerate(keep, start=1):
                relab[lab] = k
            cores = relab[cores]
            if cores.max() > 0:
                break
    if cores.max() == 0:
        cores, _ = ndimage.label(free)
    labels = _priority_flood(cores.astype(np.int32), D, free)
    labels = _split_large(labels, D, free, g.res, max_area_m2)
    out = labels.copy()
    for _ in range(3):
        changed = False
        for lab in range(1, int(out.max()) + 1):
            m = out == lab
            if not m.any() or m.sum() * g.res ** 2 >= min_area_m2:
                continue
            nb = ndimage.binary_dilation(m, np.ones((3, 3))) & ~m & (out > 0)
            vals, counts = np.unique(out[nb], return_counts=True)
            sel = vals != lab
            vals, counts = vals[sel], counts[sel]
            out[m] = vals[np.argmax(counts)] if len(vals) else 0
            changed = True
        if not changed:
            break
    return _compact_labels(out)


def _priority_flood(markers: np.ndarray, D: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Watershed by flooding from the widest free space downhill; doorway necks become borders."""
    import heapq
    lab = markers.copy().astype(np.int32)
    h = []
    nx, ny = lab.shape
    seeded = lab > 0
    bd = ndimage.binary_dilation(seeded, np.ones((3, 3))) & mask & ~seeded
    for i, j in zip(*np.where(bd)):
        heapq.heappush(h, (-float(D[i, j]), int(i), int(j)))
    visited = seeded.copy()
    while h:
        _, i, j = heapq.heappop(h)
        if visited[i, j]:
            continue
        best = 0
        bestd = -1.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and lab[a, b] > 0 and D[a, b] > bestd:
                best, bestd = int(lab[a, b]), float(D[a, b])
        if best == 0:
            continue
        lab[i, j] = best
        visited[i, j] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and mask[a, b] and not visited[a, b]:
                heapq.heappush(h, (-float(D[a, b]), a, b))
    return lab


def _compact_labels(lab: np.ndarray) -> np.ndarray:
    vals = [v for v in np.unique(lab) if v > 0]
    out = np.zeros_like(lab)
    for k, v in enumerate(vals, start=1):
        out[lab == v] = k
    return out
