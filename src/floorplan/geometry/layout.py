"""From a segmented room mask to a dimensioned polygon with openings.

The mask gives topology, the point cloud gives metric edges. We take the mask contour, force it
rectilinear in the Manhattan frame, then move every edge onto the wall plane the LiDAR actually
measured, which is what turns a 5 cm grid into a centimetre-level wall length.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from floorplan.geometry.cloud import Cloud, Grid


@dataclass
class Edge:
    axis: int              # 0: edge runs along x (constant y), 1: runs along y (constant x)
    coord: float           # the constant coordinate
    u0: float              # start along the running axis
    u1: float              # end
    outward: int           # +1 or -1: which way is outside the room
    sigma: float = 0.02
    n_support: int = 0
    refined: bool = False

    @property
    def length(self) -> float:
        return abs(self.u1 - self.u0)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.asarray(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))
    return lab == (int(np.argmax(sizes)) + 1)


def room_contour(g: Grid, mask: np.ndarray, simplify_m: float = 0.12) -> np.ndarray | None:
    m = _largest_component(ndimage.binary_fill_holes(mask))
    if m.sum() < 4:
        return None
    img = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    img[1:-1, 1:-1] = m.astype(np.uint8)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, simplify_m / g.res, True).reshape(-1, 2)
    # contour x is the column index (j), y is the row index (i)
    j = c[:, 0] - 1
    i = c[:, 1] - 1
    x, y = g.to_world(i, j)
    return np.stack([x, y], 1)


def rectilinear(poly: np.ndarray, min_edge: float = 0.25) -> list[Edge]:
    """Force the contour onto the Manhattan axes and absorb edges shorter than `min_edge`.

    A short edge is a 5 cm grid artefact, not a wall. Absorbing it means merging its two neighbours
    (which are parallel) into one wall at their length-weighted coordinate, so the polygon stays
    closed and the wall count stays at what a homeowner would count.
    """
    n = len(poly)
    if n < 4:
        return []
    edges = []
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        d = b - a
        if abs(d[0]) >= abs(d[1]):
            edges.append(Edge(axis=0, coord=float((a[1] + b[1]) / 2), u0=float(a[0]), u1=float(b[0]), outward=0))
        else:
            edges.append(Edge(axis=1, coord=float((a[0] + b[0]) / 2), u0=float(a[1]), u1=float(b[1]), outward=0))
    # merge consecutive edges on the same axis with a similar coordinate
    merged: list[Edge] = []
    for e in edges:
        if merged and merged[-1].axis == e.axis and abs(merged[-1].coord - e.coord) < 0.18:
            p = merged[-1]
            p.coord = (p.coord * p.length + e.coord * e.length) / max(p.length + e.length, 1e-6)
            p.u1 = e.u1
        else:
            merged.append(e)
    if len(merged) > 2 and merged[0].axis == merged[-1].axis and abs(merged[0].coord - merged[-1].coord) < 0.18:
        merged[0].u0 = merged[-1].u0
        merged.pop()
    # absorb short edges: drop the stub and merge the parallel neighbours either side of it
    for _ in range(200):
        if len(merged) <= 4:
            break
        n = len(merged)
        cands = sorted(range(n), key=lambda i: merged[i].length)
        done = True
        for k in cands:
            if merged[k].length >= min_edge:
                break
            a, b = merged[(k - 1) % n], merged[(k + 1) % n]
            if a.axis != b.axis or a is b:
                continue
            wa, wb = max(a.length, 1e-6), max(b.length, 1e-6)
            a.coord = (a.coord * wa + b.coord * wb) / (wa + wb)
            a.u1 = b.u1
            for idx in sorted({k, (k + 1) % n}, reverse=True):
                merged.pop(idx)
            done = False
            break
        if done:
            break
    return merged


def close_polygon(edges: list[Edge]) -> np.ndarray:
    """Corners are the intersections of consecutive perpendicular edges."""
    pts = []
    n = len(edges)
    for k in range(n):
        a, b = edges[k], edges[(k + 1) % n]
        if a.axis == b.axis:
            pts.append((a.u1, a.coord) if a.axis == 0 else (a.coord, a.u1))
            continue
        pts.append((b.coord, a.coord) if a.axis == 0 else (a.coord, b.coord))
    return np.asarray(pts, float)


def orient_edges(edges: list[Edge], centroid: np.ndarray) -> None:
    for e in edges:
        e.outward = 1 if e.coord > (centroid[1] if e.axis == 0 else centroid[0]) else -1


def refine_edges(edges: list[Edge], cloud: Cloud, band: float = 0.28, min_pts: int = 250) -> None:
    """Snap each edge onto the wall plane the sensor measured, and take its uncertainty from the fit."""
    fz = cloud.floor_z
    P = cloud.P
    sel = cloud.wallish & (P[:, 2] > fz + 0.3) & (P[:, 2] < fz + 2.0)
    XY = P[sel][:, :2]
    NX = cloud.N[sel]
    for e in edges:
        a, b = (0, 1) if e.axis == 1 else (1, 0)   # a = axis of the constant coord, b = running axis
        lo, hi = min(e.u0, e.u1), max(e.u0, e.u1)
        m = (np.abs(XY[:, a] - e.coord) < band) & (XY[:, b] > lo + 0.05) & (XY[:, b] < hi - 0.05)
        # the wall normal must face into the room
        m &= (NX[:, a] * (-e.outward)) > 0.6
        if m.sum() < min_pts:
            e.n_support = int(m.sum())
            continue
        v = XY[m, a]
        h, ed = np.histogram(v, bins=int(2 * band / 0.01) + 1, range=(e.coord - band, e.coord + band))
        c0 = ed[np.argmax(h)] + 0.005
        keep = np.abs(v - c0) < 0.04
        if keep.sum() < min_pts:
            e.n_support = int(keep.sum())
            continue
        e.coord = float(np.median(v[keep]))
        e.sigma = float(max(0.004, np.std(v[keep]) / np.sqrt(keep.sum())))
        e.n_support = int(keep.sum())
        e.refined = True


def trim_edges(edges: list[Edge]) -> None:
    """After snapping, re-cut every edge at its neighbours so the polygon stays closed."""
    n = len(edges)
    for k in range(n):
        a, b = edges[k], edges[(k + 1) % n]
        if a.axis == b.axis:
            continue
        a.u1 = b.coord
        b.u0 = a.coord


# --------------------------------------------------------------------------- openings

def wall_openings(e: Edge, cloud: Cloud, g: Grid, free: np.ndarray, *, bin_m: float = 0.02,
                  door_band=(0.25, 1.85), win_band=(1.05, 1.75), low_band=(0.15, 0.55),
                  min_door=0.55, max_door=2.6, min_win=0.35) -> list[dict]:
    """Gaps in a wall's material, classified by what is behind them and what is below them."""
    fz = cloud.floor_z
    P = cloud.P
    a, b = (0, 1) if e.axis == 1 else (1, 0)
    lo, hi = min(e.u0, e.u1), max(e.u0, e.u1)
    if hi - lo < 0.4:
        return []
    near = (np.abs(P[:, a] - e.coord) < 0.12) & (P[:, b] > lo) & (P[:, b] < hi)
    if near.sum() < 200:
        return []
    u = P[near][:, b]
    z = P[near][:, 2] - fz
    nb = int(np.ceil((hi - lo) / bin_m))
    rngu = (lo, lo + nb * bin_m)

    def prof(z0, z1):
        s = (z > z0) & (z < z1)
        h, _ = np.histogram(u[s], bins=nb, range=rngu)
        return h.astype(float)

    door = prof(*door_band)
    win = prof(*win_band)
    low = prof(*low_band)
    if door.sum() < 200:
        return []
    sm = np.convolve(door, np.ones(5) / 5, mode="same")
    plateau = float(np.median(sm[sm > 0])) if (sm > 0).any() else 0.0
    if plateau <= 0:
        return []
    empty = sm < 0.18 * plateau
    # free space on both sides of the wall => a real passage rather than an unscanned patch
    ii = np.arange(nb)
    uc = lo + (ii + 0.5) * bin_m
    off = 0.22
    if e.axis == 1:
        p1 = g.to_cell(np.full(nb, e.coord + off), uc)
        p2 = g.to_cell(np.full(nb, e.coord - off), uc)
    else:
        p1 = g.to_cell(uc, np.full(nb, e.coord + off))
        p2 = g.to_cell(uc, np.full(nb, e.coord - off))
    def sample(pc):
        i, j = pc
        ok = (i >= 0) & (i < free.shape[0]) & (j >= 0) & (j < free.shape[1])
        out = np.zeros(nb, bool)
        out[ok] = free[i[ok], j[ok]]
        return out
    through = sample(p1) & sample(p2)

    out = []
    k = 0
    while k < nb:
        if not empty[k]:
            k += 1
            continue
        j = k
        while j + 1 < nb and empty[j + 1]:
            j += 1
        run = (j - k + 1) * bin_m
        if k > 0 and j < nb - 1 and run >= min_win:
            u0, u1 = _halfmax(sm, k, j, lo, bin_m, plateau)
            w = u1 - u0
            frac_through = float(through[k:j + 1].mean())
            below = float(low[k:j + 1].mean())
            side = min(float(np.mean(low[max(0, k - 12):k])), float(np.mean(low[j + 1:j + 13]))) if k >= 1 else 0.0
            has_below = below > 0.35 * max(side, 1e-6)
            zs = z[(u >= u0) & (u <= u1)]
            top = float(np.percentile(zs, 97)) if len(zs) > 20 else float("nan")
            if frac_through > 0.5 and not has_below and min_door <= w <= max_door:
                out.append(dict(type="door", u0=u0, u1=u1, width=w, top=top, sill=0.0,
                                support=int(sm[max(0, k - 10):k].sum() + sm[j + 1:j + 11].sum()), evidence="free-both-sides"))
            elif has_below and win.sum() < 0.2 * door.sum() * (win_band[1] - win_band[0]) / (door_band[1] - door_band[0]) + 1e9:
                sill = float(np.percentile(zs[zs < 1.2], 95)) if (len(zs) and (zs < 1.2).any()) else 0.9
                if w >= min_win and w <= 3.2:
                    out.append(dict(type="window", u0=u0, u1=u1, width=w, top=top, sill=sill,
                                    support=int(low[k:j + 1].sum()), evidence="material-below-gap"))
        k = j + 1
    return out


def _halfmax(prof: np.ndarray, k: int, j: int, lo: float, bin_m: float, plateau: float):
    """Sub-bin edges where the wall-material profile crosses half its plateau."""
    n = len(prof)
    a = k
    while a - 1 >= 0 and prof[a - 1] < 0.5 * plateau:
        a -= 1
    b = j
    while b + 1 < n and prof[b + 1] < 0.5 * plateau:
        b += 1
    u0 = lo + a * bin_m
    u1 = lo + (b + 1) * bin_m
    if a - 1 >= 0 and prof[a - 1] > prof[a]:
        f = (0.5 * plateau - prof[a]) / (prof[a - 1] - prof[a])
        u0 = lo + (a + 0.5 - f) * bin_m
    if b + 1 < n and prof[b + 1] > prof[b]:
        f = (0.5 * plateau - prof[b]) / (prof[b + 1] - prof[b])
        u1 = lo + (b + 0.5 + f) * bin_m
    return float(u0), float(u1)


def room_ceiling(cloud: Cloud, g: Grid, mask: np.ndarray, min_pts: int = 400):
    """Ceiling height for one room, from ceiling points inside that room's footprint."""
    P = cloud.P
    dn = cloud.vertical & (cloud.N[:, 2] < -0.9) & (P[:, 2] > cloud.floor_z + 1.9)
    if dn.sum() < min_pts:
        return None
    i, j = g.to_cell(P[dn, 0], P[dn, 1])
    ok = (i >= 0) & (i < mask.shape[0]) & (j >= 0) & (j < mask.shape[1])
    inside = np.zeros(dn.sum(), bool)
    inside[ok] = mask[i[ok], j[ok]]
    z = P[dn][inside, 2]
    if len(z) < min_pts:
        return None
    h, e = np.histogram(z, bins=400, range=(cloud.floor_z + 1.9, cloud.floor_z + 4.2))
    if h.max() < min_pts // 2:
        return None
    m = e[np.argmax(h)] + 0.003
    sel = np.abs(z - m) < 0.05
    v = z[sel]
    zc = float(np.median(v))
    return zc - cloud.floor_z, float(max(0.004, np.std(v) / np.sqrt(len(v)))), int(len(v)), float(np.std(v))
