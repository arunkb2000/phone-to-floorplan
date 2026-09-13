"""Textured photo-tier and video-tier synthetic captures of the same flats as the LiDAR tier.

Reuses the box-room geometry, wall conventions and `Surface`/`build_surfaces` from
`floorplan.io.synthetic` (never modified here) and adds

* a textured shader: painted walls (per-room tint, skirting, cornice, door/window frames),
  wood / grey tile / terracotta floors, white ceilings, sky through windows, box furniture,
  a point light per room, vignetting and sensor noise;
* staged damage in the first bedroom: a ceiling water stain and a wall crack, whose exact
  extents are returned by `damage_ground_truth`;
* the photo protocol (docs/capture/protocol.md section 2): 6 portrait stills per room with
  EXIF FocalLengthIn35mmFilm = 26;
* the video protocol (section 3): one continuous portrait clip with 2 s palm covers at
  every doorway transit.

Conventions (same as the LiDAR generator): world z-up, metres; wall A max-y, B max-x,
C min-y, D min-x; wall UV = (offset along the wall from its start, height above the
floor); ceiling UV = (x, y) from the room origin.  Camera poses are z-up world / OpenCV
camera 4x4 (`synthetic.pose_from_euler`).

Walls, floors and ceilings are rendered one-sided (front face = the side facing into
their own room), so through a door or window hole the ray continues into the room
beyond; with nothing beyond, windows show a sky gradient and doors a dim grey.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml
from PIL import Image

from floorplan.io import synthetic as syn
from floorplan.io.photos import intrinsics_from_f35
from floorplan.io.synthetic import FlatSpec, RoomSpec, Surface
from floorplan.io.video import DEFAULT_VIDEO_F35

# --------------------------------------------------------------------------- constants
PHOTO_W, PHOTO_H = 1512, 2016            # portrait 3:4
PHOTO_F35 = 26.0                         # 35 mm-equivalent focal length written to EXIF
EXIF_FOCAL_MM = 6.86                     # physical focal length written to EXIF
VIDEO_W, VIDEO_H = 1080, 1920
VIDEO_F35 = float(DEFAULT_VIDEO_F35)     # what floorplan.io.video assumes for iPhone clips (28)
VIDEO_FPS = 10.0
CAM_H = 1.45
COVER_S = 2.0                            # palm cover per doorway transit
MIN_SEGMENT_S = 4.5                      # loader drops segments < 3 s; keep a margin
_TURN_RATE = math.radians(90.0)
_PAUSE_S = 2.0
_TILT = math.radians(25.0)
_OTHER = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
_WALL_YAW = {"A": math.pi / 2, "B": 0.0, "C": -math.pi / 2, "D": math.pi}   # facing that wall


def photo_K(w: int = PHOTO_W, h: int = PHOTO_H) -> np.ndarray:
    return intrinsics_from_f35(PHOTO_F35, w, h)


def video_K(w: int = VIDEO_W, h: int = VIDEO_H) -> np.ndarray:
    return intrinsics_from_f35(VIDEO_F35, w, h)


# --------------------------------------------------------------------------- furniture
@dataclass
class Box:
    room: str
    name: str
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    material: str                        # bed | wardrobe | counter
    against: str                         # wall letter it stands against


def default_furniture(flat: FlatSpec) -> list[Box]:
    """Bed + wardrobe in the room labelled 'room', counters along wall A of the kitchen."""
    boxes = []
    for r in flat.rooms:
        ox, oy = r.origin
        sx, sy = r.size
        if r.label in ("room", "bedroom"):
            boxes.append(Box(r.key, "bed", (ox, oy + 1.0, 0.0), (ox + 1.5, oy + 2.9, 0.5), "bed", "D"))
            boxes.append(Box(r.key, "wardrobe", (ox + sx - 0.6, oy + 0.25, 0.0),
                             (ox + sx, oy + 1.45, 2.0), "wardrobe", "B"))
        elif r.label == "kitchen":
            segs = [(ox, ox + sx)]
            for op in r.openings:
                if op["wall"] != "A" or op["type"] != "door":
                    continue
                lo, hi = syn.opening_interval(r, op)
                nxt = []
                for a, b in segs:
                    if hi + 0.1 <= a or lo - 0.1 >= b:
                        nxt.append((a, b))
                        continue
                    if lo - 0.1 > a:
                        nxt.append((a, lo - 0.1))
                    if hi + 0.1 < b:
                        nxt.append((hi + 0.1, b))
                segs = nxt
            for k, (a, b) in enumerate(segs):
                if b - a >= 0.5:
                    boxes.append(Box(r.key, f"counter{k}", (a, oy + sy - 0.6, 0.0),
                                     (b, oy + sy, 0.9), "counter", "A"))
    return boxes


def _box_surfaces(b: Box) -> list[Surface]:
    (x0, y0, z0), (x1, y1, z1) = b.lo, b.hi
    e = np.zeros((0, 5))
    k = "furniture"
    return [Surface(2, z1, np.array([x0, y0]), np.array([x1, y1]), e, k, b.room, "top", 0.6),
            Surface(0, x0, np.array([y0, z0]), np.array([y1, z1]), e, k, b.room, "x0", 0.6),
            Surface(0, x1, np.array([y0, z0]), np.array([y1, z1]), e, k, b.room, "x1", 0.6),
            Surface(1, y0, np.array([x0, z0]), np.array([x1, z1]), e, k, b.room, "y0", 0.6),
            Surface(1, y1, np.array([x0, z0]), np.array([x1, z1]), e, k, b.room, "y1", 0.6)]


# --------------------------------------------------------------------------- damage
def _crack_polyline(room: RoomSpec) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Main crack polyline on wall A (u, v) plus two branches; widths in metres per segment end."""
    win = [op for op in room.openings if op["wall"] == "A" and op["type"] == "window"]
    u_edge = win[0]["offset"] if win else room.size[0] / 2          # window's left edge (u)
    start = np.array([u_edge - 0.4, 2.1])
    drop = 0.6
    run = math.sqrt(0.9 ** 2 - drop ** 2)
    end = np.array([start[0] - run, start[1] - drop])                   # down-left, clear of the window
    axis = (end - start) / np.linalg.norm(end - start)
    perp = np.array([-axis[1], axis[0]])
    jag = [0.0, 0.02, -0.015, 0.03, -0.02, 0.012, -0.025, 0.015, 0.0]
    main = [start + (k / 8) * (end - start) + jag[k] * perp for k in range(9)]
    segs, widths = [], []
    for k in range(8):
        segs.append(np.array([main[k], main[k + 1]]))
        widths.append((0.005 - 0.002 * k / 8, 0.005 - 0.002 * (k + 1) / 8))
    for k0, ang, length in ((3, math.radians(55), 0.12), (6, math.radians(-50), 0.10)):
        c, s = math.cos(ang), math.sin(ang)
        d = np.array([c * axis[0] - s * axis[1], s * axis[0] + c * axis[1]])
        p0 = main[k0]
        p1 = p0 + d * length * 0.55 + perp * 0.008
        p2 = p0 + d * length
        segs += [np.array([p0, p1]), np.array([p1, p2])]
        widths += [(0.003, 0.0025), (0.0025, 0.002)]
    return segs, widths


def _stain_params(room: RoomSpec) -> dict:
    return {"cu": 1.2, "cv": room.size[1] - 1.0, "a": 0.225, "b": 0.15}


def _stain_rho(u, v, sp):
    du, dv = u - sp["cu"], v - sp["cv"]
    th = np.arctan2(dv / sp["b"], du / sp["a"])
    mod = 1 + 0.12 * np.sin(3 * th + 0.7) + 0.08 * np.sin(5 * th + 2.1) + 0.05 * np.sin(7 * th)
    return np.sqrt((du / sp["a"]) ** 2 + (dv / sp["b"]) ** 2) / mod


def _stain_alpha(u, v, sp):
    rho = _stain_rho(u, v, sp)
    a = np.clip((1.05 - rho) / 0.25, 0.0, 1.0)
    a = a * a * (3 - 2 * a)
    rim = np.exp(-((rho - 0.93) / 0.07) ** 2)
    return a, rim


def _crack_alpha(u, v, segs, widths):
    alpha = np.zeros_like(u)
    for (p0, p1), (w0, w1) in zip(segs, widths):
        e = p1 - p0
        L2 = float(e @ e)
        s = np.clip(((u - p0[0]) * e[0] + (v - p0[1]) * e[1]) / L2, 0.0, 1.0)
        dist = np.hypot(u - (p0[0] + s * e[0]), v - (p0[1] + s * e[1]))
        w = w0 + s * (w1 - w0)
        alpha = np.maximum(alpha, np.clip((w / 2 + 0.0012 - dist) / 0.0012, 0.0, 1.0))
    return alpha


def damage_ground_truth(flat: FlatSpec) -> list[dict]:
    """Exact extents of the staged damage (UV conventions in the module docstring)."""
    out = []
    for r in flat.rooms:
        if r.label not in ("room", "bedroom"):
            continue
        sp = _stain_params(r)
        g = 0.002
        uu, vv = np.meshgrid(np.arange(sp["cu"] - 0.4, sp["cu"] + 0.4, g),
                             np.arange(sp["cv"] - 0.3, sp["cv"] + 0.3, g))
        a, _ = _stain_alpha(uu, vv, sp)
        inside = a >= 0.5
        out.append({"room": r.key, "surface": "ceiling", "class": "water_stain",
                    "bbox_uv_m": [round(float(uu[inside].min()), 4), round(float(vv[inside].min()), 4),
                                  round(float(uu[inside].max()), 4), round(float(vv[inside].max()), 4)],
                    "centre_uv_m": [sp["cu"], round(sp["cv"], 4)],
                    "area_m2": round(float(inside.sum() * g * g), 4), "length_m": 0.0,
                    "note": "brown blotch, soft edge (alpha>=0.5 counted), ceiling u=x v=y from origin"})
        segs, widths = _crack_polyline(r)
        main = segs[:8]
        length = float(sum(np.linalg.norm(s[1] - s[0]) for s in main))
        pts = np.concatenate([s for s in segs])
        wmax = max(max(w) for w in widths) / 2
        out.append({"room": r.key, "surface": "wall:A", "class": "crack",
                    "bbox_uv_m": [round(float(pts[:, 0].min() - wmax), 4), round(float(pts[:, 1].min() - wmax), 4),
                                  round(float(pts[:, 0].max() + wmax), 4), round(float(pts[:, 1].max() + wmax), 4)],
                    "centre_uv_m": [round(float(main[4][0][0]), 4), round(float(main[4][0][1]), 4)],
                    "area_m2": round(length * 0.004, 5), "length_m": round(length, 4),
                    "width_mm": [3.0, 5.0],
                    "polyline_uv_m": [[round(float(p[0]), 4), round(float(p[1]), 4)]
                                      for p in [main[0][0]] + [s[1] for s in main]],
                    "branches": 2,
                    "note": "starts 0.4 m LEFT of the window's left edge at 2.1 m, runs down-left to "
                            "1.5 m (the right side of the window is taken by the balcony door)"})
    return out


# --------------------------------------------------------------------------- scene
@dataclass
class Scene:
    flat: FlatSpec
    surfaces: list[Surface]
    boxes: list                           # parallel to surfaces: Box or None
    front: np.ndarray                     # per surface: +1/-1 accepted sign of d[axis], 0 = two-sided
    lights: dict
    damage: list[dict] = field(default_factory=list)
    damage_on: bool = True
    _wf: dict = field(default_factory=dict)

    def wall_frame(self, s: Surface):
        key = (s.room, s.wall)
        if key not in self._wf:
            self._wf[key] = syn.wall_frame(self.flat.room(s.room), s.wall)
        return self._wf[key]


def build_scene(flat: FlatSpec, furniture: bool = True, damage: bool = True) -> Scene:
    surfaces = syn.build_surfaces(flat)
    boxes: list = [None] * len(surfaces)
    front = []
    for s in surfaces:
        if s.kind == "wall":
            front.append(-float(syn.wall_frame(flat.room(s.room), s.wall).n_in[s.axis]))
        elif s.kind == "floor":
            front.append(-1.0)
        else:
            front.append(1.0)
    if furniture:
        for b in default_furniture(flat):
            for s in _box_surfaces(b):
                surfaces.append(s)
                boxes.append(b)
                front.append(0.0)
    lights = {r.key: np.array([r.origin[0] + r.size[0] / 2, r.origin[1] + r.size[1] / 2,
                               r.height - 0.3]) for r in flat.rooms}
    return Scene(flat, surfaces, boxes, np.array(front), lights,
                 damage_ground_truth(flat) if damage else [], damage)


# --------------------------------------------------------------------------- ray casting
def _visible(s: Surface, K, W, H, R, p) -> bool:
    """Conservative frustum cull: all 4 corners outside one frustum half-space -> skip."""
    a1, a2 = _OTHER[s.axis]
    c = np.zeros((4, 3))
    c[:, s.axis] = s.coord
    c[:, a1] = [s.lo[0], s.hi[0], s.lo[0], s.hi[0]]
    c[:, a2] = [s.lo[1], s.lo[1], s.hi[1], s.hi[1]]
    cc = (c - p) @ R
    x, y, z = cc[:, 0], cc[:, 1], cc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if np.all(z <= 1e-6):
        return False
    if np.all(fx * x - (W - cx) * z > 0) or np.all(fx * x + cx * z < 0):
        return False
    if np.all(fy * y - (H - cy) * z > 0) or np.all(fy * y + cy * z < 0):
        return False
    return True


def cast_scene(scene: Scene, K, W, H, R, p):
    """Like synthetic.cast but one-sided walls/floors/ceilings and frustum culling."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    d_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=0).reshape(3, -1)
    d = (R @ d_cam).T
    n = d.shape[0]
    best_t = np.full(n, np.inf)
    best_id = np.full(n, -1, dtype=np.int32)
    t_door = np.full(n, np.inf)
    t_win = np.full(n, np.inf)
    for sid, s in enumerate(scene.surfaces):
        if not _visible(s, K, W, H, R, p):
            continue
        den = d[:, s.axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (s.coord - p[s.axis]) / den
        ok = (t > 1e-6) & (t < best_t)
        fs = scene.front[sid]
        if fs != 0.0:
            ok &= den * fs > 0
        if not ok.any():
            continue
        a1, a2 = _OTHER[s.axis]
        pa = p[a1] + t * d[:, a1]
        pb = p[a2] + t * d[:, a2]
        ok &= (pa >= s.lo[0]) & (pa <= s.hi[0]) & (pb >= s.lo[1]) & (pb <= s.hi[1])
        for h in s.holes:
            inh = ok & (pa >= h[0]) & (pa <= h[1]) & (pb >= h[2]) & (pb <= h[3])
            if inh.any():
                tgt = t_door if h[4] > 0.5 else t_win
                np.minimum(tgt, np.where(inh, t, np.inf), out=tgt)
                ok &= ~inh
        best_t = np.where(ok, t, best_t)
        best_id = np.where(ok, sid, best_id)
    return best_t, best_id, t_door, t_win, d


# --------------------------------------------------------------------------- textures
def _hash2(ix, iy, seed=0.0):
    h = np.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453
    return h - np.floor(h)


def _vnoise(x, y, seed=0.0):
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a, b = _hash2(ix, iy, seed), _hash2(ix + 1, iy, seed)
    c, dd = _hash2(ix, iy + 1, seed), _hash2(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + dd * fx) * fy


_TINTS = {"room": (0.94, 0.90, 0.83), "bedroom": (0.94, 0.90, 0.83), "kitchen": (0.90, 0.92, 0.87),
          "washroom": (0.88, 0.92, 0.95), "balcony": (0.93, 0.91, 0.85)}
_TINT_CYCLE = [(0.94, 0.90, 0.83), (0.90, 0.92, 0.87), (0.88, 0.92, 0.95), (0.93, 0.91, 0.85)]
_DOOR_FRAME = np.array([0.32, 0.20, 0.12])
_WIN_FRAME = np.array([0.97, 0.97, 0.96])
_CRACK = np.array([0.12, 0.10, 0.09])
_STAIN = np.array([0.55, 0.42, 0.28])


def _tint(flat: FlatSpec, room: RoomSpec) -> np.ndarray:
    if room.label in _TINTS:
        return np.array(_TINTS[room.label])
    return np.array(_TINT_CYCLE[flat.rooms.index(room) % len(_TINT_CYCLE)])


def _wall_albedo(scene: Scene, s: Surface, u, v):
    room = scene.flat.room(s.room)
    h = room.height
    wf = scene.wall_frame(s)
    n = _vnoise(u / 0.6, v / 0.6, 1.0) + 0.5 * _vnoise(u / 0.15, v / 0.15, 2.0)
    col = _tint(scene.flat, room)[None, :] * (1.0 + 0.05 * (n - 0.75))[:, None]
    sk = v < 0.10
    col[sk] = np.array([0.36, 0.30, 0.26])[None, :] * \
        (1 + 0.05 * (_vnoise(u[sk] / 0.1, v[sk] / 0.1, 3.0) - 0.5))[:, None]
    col[(v > h - 0.05) & (v <= h - 0.035)] *= 0.80
    col[v > h - 0.035] = (0.98, 0.98, 0.97)
    f = 0.06
    for hole in s.holes:
        ua = (hole[0] - wf.start[wf.along]) * wf.dir[wf.along]
        ub = (hole[1] - wf.start[wf.along]) * wf.dir[wf.along]
        lo, hi = min(ua, ub), max(ua, ub)
        m = (u >= lo - f) & (u <= hi + f) & (v >= max(hole[2] - f, 0.0)) & (v <= hole[3] + f)
        m &= ~((u >= lo) & (u <= hi) & (v >= hole[2]) & (v <= hole[3]))
        col[m] = _DOOR_FRAME if hole[4] > 0.5 else _WIN_FRAME
    if scene.damage_on:
        for dmg in scene.damage:
            if dmg["room"] == s.room and dmg["surface"] == f"wall:{s.wall}":
                b = dmg["bbox_uv_m"]
                m = (u >= b[0] - 0.01) & (u <= b[2] + 0.01) & (v >= b[1] - 0.01) & (v <= b[3] + 0.01)
                if m.any():
                    segs, widths = _crack_polyline(room)
                    a = _crack_alpha(u[m], v[m], segs, widths)[:, None]
                    col[m] = col[m] * (1 - a) + _CRACK[None, :] * a
    return col


def _wood(u, v):
    row = np.floor(v / 0.12)
    uu = u + _hash2(row, np.zeros_like(row), 5.0) * 1.2
    board = np.floor(uu / 1.2)
    tone = 1 + 0.04 * (2 * (row % 2) - 1) + 0.14 * (_hash2(row, board, 6.0) - 0.5)
    grain = 0.05 * (_vnoise(u / 0.35, v / 0.012, 7.0) - 0.5)
    col = np.array([0.60, 0.43, 0.28])[None, :] * (tone + grain)[:, None]
    gap = ((v % 0.12) < 0.003) | ((uu % 1.2) < 0.004)
    col[gap] *= 0.5
    return col


def _tiles(u, v, size, base, grout):
    tx, ty = np.floor(u / size), np.floor(v / size)
    tone = 1 + 0.08 * (_hash2(tx, ty, 8.0) - 0.5) + 0.04 * (_vnoise(u / 0.05, v / 0.05, 9.0) - 0.5)
    col = np.array(base)[None, :] * tone[:, None]
    col[((u % size) < 0.005) | ((v % size) < 0.005)] = grout
    return col


def _floor_albedo(scene: Scene, s: Surface, u, v):
    room = scene.flat.room(s.room)
    if room.label in ("room", "bedroom"):
        return _wood(u, v)
    if room.label == "balcony":
        return _tiles(u, v, 0.3, (0.72, 0.42, 0.28), (0.55, 0.40, 0.32))
    return _tiles(u, v, 0.3, (0.62, 0.62, 0.60), (0.45, 0.45, 0.44))


def _ceiling_albedo(scene: Scene, s: Surface, u, v):
    room = scene.flat.room(s.room)
    col = np.array([0.97, 0.97, 0.955])[None, :] * (1 - 0.05 * u / room.size[0])[:, None]
    if scene.damage_on:
        for dmg in scene.damage:
            if dmg["room"] == s.room and dmg["surface"] == "ceiling":
                sp = _stain_params(room)
                b = dmg["bbox_uv_m"]
                m = (u >= b[0] - 0.05) & (u <= b[2] + 0.05) & (v >= b[1] - 0.05) & (v <= b[3] + 0.05)
                if m.any():
                    a, rim = _stain_alpha(u[m], v[m], sp)
                    tex = 1 + 0.15 * (_vnoise(u[m] / 0.04, v[m] / 0.04, 11.0) - 0.5)
                    stain = _STAIN[None, :] * tex[:, None]
                    mix = (0.65 * a)[:, None]
                    col[m] = (col[m] * (1 - mix) + stain * mix) * (1 - 0.30 * rim * a)[:, None]
    return col


def _furniture_albedo(box: Box, s: Surface, pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    m = len(x)
    if box.material == "bed":
        col = np.tile((0.13, 0.18, 0.40), (m, 1)) * \
            (1 + 0.08 * (_vnoise(x / 0.05, (y + z) / 0.05, 10.0) - 0.5))[:, None]
        if s.wall == "top":
            pillow = (x > box.lo[0] + 0.1) & (x < box.hi[0] - 0.1) & (y > box.hi[1] - 0.55) & (y < box.hi[1] - 0.1)
            col[pillow] = (0.92, 0.92, 0.90)
        else:
            col[z < 0.15] = (0.42, 0.30, 0.20)
        return col
    if box.material == "wardrobe":
        a = y if s.axis == 0 else x
        col = np.tile((0.70, 0.56, 0.38), (m, 1)) * \
            (1 + 0.10 * (_vnoise(a / 0.015, z / 0.6, 12.0) - 0.5))[:, None]
        mid = (box.lo[1] + box.hi[1]) / 2
        col[(np.abs(y - mid) < 0.003) & (s.wall != "top")] = (0.25, 0.18, 0.12)
        col[(z > box.hi[2] - 0.02) | (s.wall == "top")] *= 0.9
        return col
    col = np.tile((0.50, 0.51, 0.53), (m, 1)) * \
        (1 + 0.04 * (_vnoise(x / 0.03, (y + z) / 0.03, 13.0) - 0.5))[:, None]
    if s.wall == "top":
        col[:] = (0.72, 0.72, 0.70)
    else:
        col[z > box.hi[2] - 0.04] = (0.35, 0.35, 0.36)
    return col


# --------------------------------------------------------------------------- rendering
def render_view(scene: Scene, K, W: int, H: int, T: np.ndarray, rng: np.random.Generator,
                noise_sigma: float = 2.0 / 255, vignette: float = 0.22) -> np.ndarray:
    """Shaded RGB uint8 (H, W, 3) from a z-up/OpenCV camera-to-world pose."""
    R, p = T[:3, :3], T[:3, 3]
    t, sid, t_door, t_win, d = cast_scene(scene, K, W, H, R, p)
    n = W * H
    hit = np.isfinite(t)
    rgb = np.tile(np.array([0.28, 0.27, 0.26]), (n, 1))
    dn = np.linalg.norm(d, axis=1)
    el = np.arcsin(np.clip(d[:, 2] / dn, -1.0, 1.0))
    mix = np.clip((el + 0.05) / 0.45, 0.0, 1.0)[:, None]
    sky = (1 - mix) * np.array([0.95, 0.96, 0.98]) + mix * np.array([0.58, 0.76, 0.96])
    wmask = ~hit & np.isfinite(t_win)
    rgb[wmask] = sky[wmask]
    pts = p[None, :] + np.where(hit, t, 0.0)[:, None] * d
    for s_id in np.unique(sid[hit]):
        m = sid == s_id
        s = scene.surfaces[s_id]
        P = pts[m]
        box = scene.boxes[s_id]
        if box is not None:
            alb = _furniture_albedo(box, s, P)
        elif s.kind == "wall":
            wf = scene.wall_frame(s)
            u = (P[:, wf.along] - wf.start[wf.along]) * wf.dir[wf.along]
            alb = _wall_albedo(scene, s, u, P[:, 2])
        else:
            room = scene.flat.room(s.room)
            u, v = P[:, 0] - room.origin[0], P[:, 1] - room.origin[1]
            alb = _floor_albedo(scene, s, u, v) if s.kind == "floor" else _ceiling_albedo(scene, s, u, v)
        L = scene.lights[s.room]
        lv = L[None, :] - P
        r = np.linalg.norm(lv, axis=1) + 1e-9
        sign = -np.sign(d[m, s.axis])
        ndotl = np.maximum(0.0, sign * lv[:, s.axis] / r)
        diff = ndotl / (1 + 0.06 * r * r)
        rgb[m] = alb * (0.55 + 0.55 * diff)[:, None]
    img = rgb.reshape(H, W, 3)
    if vignette > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = K[0, 2], K[1, 2]
        r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (cx ** 2 + cy ** 2)
        img = img * (1 - vignette * r2)[:, :, None]
    if noise_sigma > 0:
        img = img + rng.normal(0.0, noise_sigma, img.shape)
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def save_photo(path: str, rgb: np.ndarray, quality: int = 92) -> None:
    """JPEG with EXIF FocalLengthIn35mmFilm (0xA405) and FocalLength (0x920A) in the Exif IFD."""
    exif = Image.Exif()
    exif[0x010F] = "floorplan"
    exif[0x0110] = "synthetic pinhole"
    ifd = exif.get_ifd(0x8769)
    ifd[0xA405] = int(PHOTO_F35)
    ifd[0x920A] = EXIF_FOCAL_MM
    Image.fromarray(rgb).save(path, quality=quality, exif=exif.tobytes())


# --------------------------------------------------------------------------- photo protocol
def _facing_basis(room: RoomSpec):
    f = syn.wall_frame(room, room.entry_wall).n_in
    return f, np.array([f[1], -f[0]])            # forward, right (as seen facing in)


def _corner(room: RoomSpec, near: bool, left: bool, inset: float) -> np.ndarray:
    f, r = _facing_basis(room)
    c = np.array(room.origin) + np.array(room.size) / 2
    hf = abs(float(np.array(room.size) @ f)) / 2 - inset
    hr = abs(float(np.array(room.size) @ r)) / 2 - inset
    return c + f * (-hf if near else hf) + r * (-hr if left else hr)


def _push_out_of_furniture(pos: np.ndarray, room: RoomSpec, boxes: list[Box], clear: float = 0.3):
    moved = False
    for b in boxes:
        if b.room != room.key:
            continue
        if b.lo[0] - clear < pos[0] < b.hi[0] + clear and b.lo[1] - clear < pos[1] < b.hi[1] + clear:
            n_in = syn.wall_frame(room, b.against).n_in
            ax = 0 if n_in[0] != 0 else 1
            pos[ax] = (b.hi[ax] + clear) if n_in[ax] > 0 else (b.lo[ax] - clear)
            moved = True
    return pos, moved


def _next_room_door(flat: FlatSpec, room: RoomSpec) -> dict:
    idx = flat.rooms.index(room)
    if idx + 1 < len(flat.rooms):
        nxt = flat.rooms[idx + 1].key
        for op in room.openings:
            if op["type"] == "door" and op.get("to") == nxt:
                return op
    return syn.entrance_door(room)


def photo_plan(flat: FlatSpec, boxes: list[Box], inset: float = 0.45) -> dict[str, list[dict]]:
    """Nominal (un-jittered) photo poses per room, protocol order 01..06."""
    plan = {}
    for room in flat.rooms:
        f, _ = _facing_basis(room)
        ent = syn.entrance_door(room)
        shots = []
        pos = syn.opening_centre(room, ent) + f * 0.15
        shots.append(("01_doorway", pos, math.atan2(f[1], f[0])))
        corners = {"near_left": _corner(room, True, True, inset), "far_left": _corner(room, False, True, inset),
                   "far_right": _corner(room, False, False, inset), "near_right": _corner(room, True, False, inset)}
        order = [("02_near_left", "near_left", "far_right"), ("03_far_left", "far_left", "near_right"),
                 ("04_far_right", "far_right", "near_left"), ("05_near_right", "near_right", "far_left")]
        for name, here, aim in order:
            p0 = corners[here].copy()
            p1, moved = _push_out_of_furniture(p0, room, boxes)
            tgt = corners[aim]
            shots.append((name + ("_shifted" if moved else ""), p1, math.atan2(tgt[1] - p1[1], tgt[0] - p1[0])))
        door = _next_room_door(flat, room)
        wf = syn.wall_frame(room, door["wall"])
        depth = room.size[0 if wf.n_in[0] != 0 else 1]
        pos = syn.opening_centre(room, door) + wf.n_in * min(1.5, depth - 0.3)
        pos, moved = _push_out_of_furniture(pos, room, boxes)
        role = "06_exit_door" if door is not syn.entrance_door(room) or door.get("to") else "06_entrance_door"
        shots.append((role + ("_shifted" if moved else ""), pos, math.atan2(-wf.n_in[1], -wf.n_in[0])))
        plan[room.key] = [{"name": n, "xy": p, "yaw": y} for n, p, y in shots]
    return plan


def render_photo_set(flat: FlatSpec, out_dir: str, *, seed: int = 0, jitter_pos: float = 0.02,
                     jitter_yaw_deg: float = 2.0, jitter_pitch_deg: float = 2.0, furniture: bool = True,
                     damage: bool = True, verbose: bool = True) -> dict:
    """Six protocol photos per room -> out_dir/<room>/01.jpg..06.jpg plus poses.yaml."""
    scene = build_scene(flat, furniture, damage)
    boxes = [b for b in dict.fromkeys(b for b in scene.boxes if b is not None)]
    plan = photo_plan(flat, boxes)
    rng = np.random.default_rng([seed, 7])
    K = photo_K()
    poses = {"intrinsics": {"width": PHOTO_W, "height": PHOTO_H, "f35_mm": PHOTO_F35,
                            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])},
             "camera_height_m": CAM_H, "seed": seed,
             "jitter": {"pos_m": jitter_pos, "yaw_deg": jitter_yaw_deg, "pitch_deg": jitter_pitch_deg},
             "convention": "z-up world metres; yaw from +x about +z (deg); pitch +up; roll 0; "
                           "poses are camera centres, camera looks along yaw/pitch; debug reference only",
             "rooms": {}}
    for room in flat.rooms:
        rd = os.path.join(out_dir, room.key)
        os.makedirs(rd, exist_ok=True)
        entries = []
        for i, shot in enumerate(plan[room.key]):
            xy = shot["xy"] + rng.uniform(-jitter_pos, jitter_pos, 2)
            z = CAM_H + rng.uniform(-jitter_pos, jitter_pos)
            yaw = shot["yaw"] + math.radians(rng.uniform(-jitter_yaw_deg, jitter_yaw_deg))
            pitch = math.radians(rng.uniform(-jitter_pitch_deg, jitter_pitch_deg))
            T = syn.pose_from_euler(xy[0], xy[1], z, yaw, pitch, 0.0)
            rgb = render_view(scene, K, PHOTO_W, PHOTO_H, T, rng)
            fn = f"{i + 1:02d}.jpg"
            save_photo(os.path.join(rd, fn), rgb)
            entries.append({"file": fn, "shot": shot["name"],
                            "nominal": {"x": round(float(shot["xy"][0]), 4), "y": round(float(shot["xy"][1]), 4),
                                        "z": CAM_H, "yaw_deg": round(math.degrees(shot["yaw"]), 3), "pitch_deg": 0.0},
                            "actual": {"x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4), "z": round(float(z), 4),
                                       "yaw_deg": round(math.degrees(yaw), 3), "pitch_deg": round(math.degrees(pitch), 3)}})
            if verbose:
                print(f"  [{os.path.basename(out_dir)}] {room.key}/{fn} {shot['name']}", flush=True)
        poses["rooms"][room.key] = entries
    with open(os.path.join(out_dir, "poses.yaml"), "w") as f:
        yaml.safe_dump(poses, f, sort_keys=False)
    return poses


# --------------------------------------------------------------------------- video protocol
class _VPath:
    """Events: (kind, dur, xy0, xy1, yaw0, yaw1, cover, room). Yaw kept continuous."""

    def __init__(self, xy, yaw, room):
        self.ev: list[tuple] = []
        self.xy = np.asarray(xy, dtype=float)
        self.yaw = float(yaw)
        self.room = room
        self.seg_start = 0.0
        self.now = 0.0

    def _add(self, kind, dur, xy1, yaw1, cover=False):
        self.ev.append((kind, dur, self.xy.copy(), np.asarray(xy1, float), self.yaw, float(yaw1), cover, self.room))
        self.xy = np.asarray(xy1, float)
        self.yaw = float(yaw1)
        self.now += dur

    def hold(self, dur):
        self._add("hold", dur, self.xy, self.yaw)

    def pause_tilt(self):
        self._add("tilt", _PAUSE_S, self.xy, self.yaw)

    def turn_to(self, yaw):
        tgt = self.yaw + syn._wrap(yaw - self.yaw)
        if abs(tgt - self.yaw) > 1e-3:
            self._add("turn", abs(tgt - self.yaw) / _TURN_RATE, self.xy, tgt)

    def move(self, xy, speed, face_walk=True):
        xy = np.asarray(xy, float)
        dist = float(np.linalg.norm(xy - self.xy))
        if dist < 1e-6:
            return
        if face_walk:
            self.turn_to(math.atan2(xy[1] - self.xy[1], xy[0] - self.xy[0]))
        self._add("move", dist / speed, xy, self.yaw)

    def cover_to(self, xy, new_room):
        if self.now - self.seg_start < MIN_SEGMENT_S:
            self.hold(MIN_SEGMENT_S - (self.now - self.seg_start))
        self._add("cover", COVER_S, xy, self.yaw, cover=True)
        self.room = new_room
        self.seg_start = self.now

    def sample(self, fps):
        durs = np.array([e[1] for e in self.ev])
        starts = np.concatenate([[0.0], np.cumsum(durs)])
        n = math.floor(starts[-1] * fps) + 1
        t = np.arange(n) / fps
        idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(self.ev) - 1)
        xy = np.zeros((n, 2))
        yaw = np.zeros(n)
        pitch = np.zeros(n)
        cover = np.zeros(n, bool)
        rooms = []
        for i in range(n):
            kind, dur, p0, p1, y0, y1, cv, room = self.ev[idx[i]]
            s = min(1.0, (t[i] - starts[idx[i]]) / dur) if dur > 0 else 1.0
            xy[i] = p0 + s * (p1 - p0)
            yaw[i] = y0 + s * (y1 - y0)
            if kind == "tilt":
                pitch[i] = _TILT * math.sin(2 * math.pi * s)
            cover[i] = cv
            rooms.append(room)
        return t, xy, yaw, pitch, cover, rooms


def _loop_rect(room: RoomSpec, boxes: list[Box], clear: float = 0.3):
    ox, oy = room.origin
    sx, sy = room.size
    m = min(1.0, 0.4 * min(sx, sy))
    x0, x1, y0, y1 = ox + m, ox + sx - m, oy + m, oy + sy - m
    for b in boxes:
        if b.room != room.key:
            continue
        if b.against == "D":
            x0 = max(x0, b.hi[0] + clear)
        elif b.against == "B":
            x1 = min(x1, b.lo[0] - clear)
        elif b.against == "A":
            y1 = min(y1, b.lo[1] - clear)
        elif b.against == "C":
            y0 = max(y0, b.hi[1] + clear)
    return x0, x1, y0, y1


def _door_graph(flat: FlatSpec):
    g: dict[str, list] = {r.key: [] for r in flat.rooms}
    for a, oa, b, ob in syn.check_shared_doors(flat):
        g[a].append((b, oa, ob))
        g[b].append((a, ob, oa))
    return g


def _bfs(graph, src, dst):
    prev = {src: None}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        if cur == dst:
            break
        for nxt, oa, ob in graph[cur]:
            if nxt not in prev:
                prev[nxt] = (cur, oa, ob)
                queue.append(nxt)
    if dst not in prev:
        raise ValueError(f"no door path {src} -> {dst}")
    hops, k = [], dst
    while prev[k] is not None:
        c, oa, ob = prev[k]
        hops.append((k, oa, ob))
        k = c
    return hops[::-1]


def video_visits(flat: FlatSpec) -> list[tuple[str, str]]:
    """(room, mode) sequence: loop every room in walk order, transit through rooms in between,
    then return to the first room."""
    graph = _door_graph(flat)
    visits = [(flat.rooms[0].key, "loop")]
    cur = flat.rooms[0].key
    for r in flat.rooms[1:]:
        hops = _bfs(graph, cur, r.key)
        for k, (nxt, _, _) in enumerate(hops):
            visits.append((nxt, "loop" if k == len(hops) - 1 else "transit"))
        cur = r.key
    if cur != flat.rooms[0].key:
        hops = _bfs(graph, cur, flat.rooms[0].key)
        for k, (nxt, _, _) in enumerate(hops):
            visits.append((nxt, "final" if k == len(hops) - 1 else "transit"))
    return visits


def _inside_pt(room: RoomSpec, door: dict, dist: float = 0.5) -> np.ndarray:
    return syn.opening_centre(room, door) + syn.wall_frame(room, door["wall"]).n_in * dist


def video_path(flat: FlatSpec, boxes: list[Box], walk_speed: float = 0.5) -> _VPath:
    graph = _door_graph(flat)
    visits = video_visits(flat)
    first = flat.rooms[0]
    ent = syn.entrance_door(first)
    f = syn.wall_frame(first, ent["wall"]).n_in
    start_xy = syn.opening_centre(first, ent) + f * 0.15
    path = _VPath(start_xy, math.atan2(f[1], f[0]), first.key)
    path.hold(1.5)
    for vi, (key, mode) in enumerate(visits):
        room = flat.room(key)
        if mode in ("loop",):
            x0, x1, y0, y1 = _loop_rect(room, boxes)
            corners = [np.array([x0, y1]), np.array([x1, y1]), np.array([x1, y0]), np.array([x0, y0])]
            faces = ["A", "B", "C", "D"]                       # wall on the left of each clockwise edge
            k0 = int(np.argmin([np.linalg.norm(c - path.xy) for c in corners]))
            corners = corners[k0:] + corners[:k0]
            faces = faces[k0:] + faces[:k0]
            path.move(corners[0], walk_speed)
            for j in range(4):
                a, b = corners[j], corners[(j + 1) % 4]
                path.turn_to(_WALL_YAW[faces[j]])
                path.move((a + b) / 2, walk_speed, face_walk=False)
                path.pause_tilt()
                path.move(b, walk_speed, face_walk=False)
        if mode == "final":
            path.move(start_xy, walk_speed)
            path.turn_to(math.atan2(f[1], f[0]))
            path.hold(2.0)
            break
        nxt_key = visits[vi + 1][0]
        hop = [h for h in graph[key] if h[0] == nxt_key][0]
        _, door_here, door_there = hop
        path.move(_inside_pt(room, door_here), walk_speed)
        wf = syn.wall_frame(room, door_here["wall"])
        path.turn_to(math.atan2(-wf.n_in[1], -wf.n_in[0]))
        path.cover_to(_inside_pt(flat.room(nxt_key), door_there), nxt_key)
        path.hold(1.0)
    return path


def render_video(flat: FlatSpec, out_dir: str, *, seed: int = 0, fps: float = VIDEO_FPS,
                 walk_speed: float = 0.5, render_scale: float = 0.5, furniture: bool = True,
                 damage: bool = True, out_size: tuple[int, int] = (VIDEO_W, VIDEO_H),
                 verbose: bool = True) -> dict:
    """walk.mp4 (+ rooms.txt, meta.yaml, poses_gt.csv) for the section-3 protocol walk."""
    scene = build_scene(flat, furniture, damage)
    boxes = [b for b in dict.fromkeys(b for b in scene.boxes if b is not None)]
    path = video_path(flat, boxes, walk_speed)
    t, xy, yaw, pitch, cover, rooms = path.sample(fps)
    n = len(t)
    yaw = yaw + math.radians(3.0) * np.sin(2 * math.pi * t / 5.3)
    pitch = pitch + math.radians(2.0) * np.sin(2 * math.pi * t / 3.7)
    z = CAM_H + 0.02 * np.sin(2 * math.pi * 1.8 * t)
    W, H = out_size
    rw, rh = int(round(W * render_scale)), int(round(H * render_scale))
    K = intrinsics_from_f35(VIDEO_F35, rw, rh)
    rng = np.random.default_rng([seed, 9])
    os.makedirs(out_dir, exist_ok=True)
    vpath = os.path.join(out_dir, "walk.mp4")
    writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {vpath}")
    for i in range(n):
        if cover[i]:
            fr = np.clip(rng.normal(3.0, 1.5, (H, W, 3)), 0, 255).astype(np.uint8)
        else:
            T = syn.pose_from_euler(xy[i, 0], xy[i, 1], z[i], yaw[i], pitch[i], 0.0)
            rgb = render_view(scene, K, rw, rh, T, rng)
            if (rw, rh) != (W, H):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            fr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(fr)
        if verbose and (i % 200 == 0 or i == n - 1):
            print(f"  [{os.path.basename(out_dir)}] frame {i + 1}/{n}", flush=True)
    writer.release()
    # segments / covers for meta
    segments, covers = [], []
    cur, s0 = None, 0
    for i in range(n + 1):
        state = None if i == n else ("cover" if cover[i] else rooms[i])
        if state != cur:
            if cur is not None:
                (covers if cur == "cover" else segments).append(
                    {"room": None if cur == "cover" else cur, "t_start": round(float(t[s0]), 3),
                     "t_end": round(float(t[i - 1]), 3), "frames": int(i - s0)})
            cur, s0 = state, i
    with open(os.path.join(out_dir, "rooms.txt"), "w") as f:
        f.write("\n".join(s["room"] for s in segments) + "\n")
    with open(os.path.join(out_dir, "poses_gt.csv"), "w") as f:
        f.write("frame, t, x, y, z, yaw_deg, pitch_deg, cover, room\n")
        for i in range(n):
            f.write(f"{i}, {t[i]:.3f}, {xy[i, 0]:.4f}, {xy[i, 1]:.4f}, {z[i]:.4f}, "
                    f"{math.degrees(yaw[i]):.3f}, {math.degrees(pitch[i]):.3f}, {int(cover[i])}, {rooms[i]}\n")
    meta = {"fps": float(fps), "width": W, "height": H, "render_size": [rw, rh],
            "f35_mm": VIDEO_F35, "camera_height_m": CAM_H, "walk_speed": walk_speed, "seed": seed,
            "n_frames": int(n), "duration_s": round(float(t[-1]), 3), "cover_s": COVER_S,
            "segments": [{k: v for k, v in s.items() if k != "room"} | {"room": s["room"]} for s in segments],
            "covers": [{k: v for k, v in c.items() if k != "room"} for c in covers],
            "note": "poses_gt.csv is a debug reference (z-up world, yaw from +x, pitch +up); "
                    "the pipeline must not read it"}
    with open(os.path.join(out_dir, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return meta


def write_damage_yaml(flat: FlatSpec, path: str) -> list[dict]:
    dmg = damage_ground_truth(flat)
    doc = {"conventions": {"wall_uv": "u = offset along the wall from its start (A: from x_min, B: from y_max, "
                                      "C: from x_max, D: from y_min), v = height above the floor, metres",
                           "ceiling_uv": "u = x - origin_x, v = y - origin_y, metres",
                           "bbox_uv_m": "[u0, v0, u1, v1]"},
           "damage": dmg}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return dmg
