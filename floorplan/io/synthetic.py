"""Synthetic Stray Scanner captures of axis-aligned box rooms with exact ground truth.

We have no LiDAR iPhone, so the LiDAR tier is validated on data that mimics the
Stray Scanner iOS export byte-for-byte in layout:

    <out_dir>/
      camera_matrix.csv      3 lines, 3x3 intrinsics for the RGB resolution (640x480 here)
      odometry.csv           "timestamp, frame, x, y, z, qx, qy, qz, qw", camera-to-world,
                             ARKit world (gravity aligned, y up), OpenGL camera (looks down -z)
      depth/NNNNNN.png       16-bit, 192 rows x 256 cols, millimetres, 0 = invalid
      confidence/NNNNNN.png  8-bit, same size, 0/1/2 (2 = high)
      rgb/NNNNNN.jpg         640x480 shaded gray frames (the real app writes rgb.mp4 at
                             1920x1440; our loader accepts either)
      traj_gt.csv            TRUE poses, same format and convention as odometry.csv
      ground_truth.yaml      room dimensions, openings, areas, placement, drift realised

Internal geometry is **z-up**, metres.  Rooms are axis-aligned rectangles with `origin`
= min corner and `size` = (x extent, y extent).  Wall naming (pipeline convention):
standing in the entrance door facing in, A is ahead (max-y wall, along x), B is on the
right (max-x wall), C is behind (min-y wall), D is on the left (min-x wall).  Offsets
along a wall run clockwise as seen from above, i.e. from the wall's left corner as seen
from inside the room:

    A: from (x_min, y_max) towards +x      B: from (x_max, y_max) towards -y
    C: from (x_max, y_min) towards -x      D: from (x_min, y_min) towards +y

Walls are infinitely thin.  Doors are holes from z=0 to `height`, windows from `sill`
to `sill+height`; rays go through holes into the neighbouring room (or to nothing).

Pose conversion (documented here, implemented in `zup_pose_to_stray` /
`stray_pose_to_zup`, which are exact inverses):

    world  z-up  -> ARKit y-up :  X_ar = x,  Y_ar = z,  Z_ar = -y        (matrix W)
    camera OpenCV (x right, y down, z fwd) -> OpenGL (x right, y up, z back):  C = diag(1,-1,-1)

    T_stray = [W 0; 0 1] @ T_zup_cv @ [C 0; 0 1]
    i.e.  R_gl = W R_cv C ,  t_ar = W t_zup      and back:  R_cv = W^T R_gl C,  t_zup = W^T t_ar

Quaternions are (qx, qy, qz, qw), Hamilton convention, camera-to-world.
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

# --------------------------------------------------------------------------- constants
DEPTH_W, DEPTH_H = 256, 192
DEPTH_F = 212.0                      # ~ real Stray Scanner depth intrinsics
K_DEPTH = np.array([[DEPTH_F, 0.0, 128.0], [0.0, DEPTH_F, 96.0], [0.0, 0.0, 1.0]])
RGB_W, RGB_H = 640, 480
_RGB_SCALE = RGB_W / DEPTH_W         # 2.5
K_RGB = np.array([[DEPTH_F * _RGB_SCALE, 0.0, RGB_W / 2], [0.0, DEPTH_F * _RGB_SCALE, RGB_H / 2],
                  [0.0, 0.0, 1.0]])
MAX_DEPTH = 6.0                      # metres; farther hits are invalid (0)
CAM_HEIGHT = 1.4
WALLS = ("A", "B", "C", "D")
CSV_HEADER = "timestamp, frame, x, y, z, qx, qy, qz, qw"

# world z-up -> ARKit y-up ; OpenCV camera <-> OpenGL camera
_W = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_C = np.diag([1.0, -1.0, -1.0])
# level camera looking along +x: columns = (right, down, forward) in world coords
_R0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


# --------------------------------------------------------------------------- specs
@dataclass
class RoomSpec:
    key: str
    label: str
    origin: tuple[float, float]          # min corner (x, y), metres
    size: tuple[float, float]            # (x extent, y extent), metres
    height: float
    openings: list[dict] = field(default_factory=list)
    # each: {"wall": "A|B|C|D", "type": "door|window", "offset": m, "width": m,
    #        "height": m, "sill": m (windows), "to": "<room key>" (doors, optional)}
    entry_wall: str = "C"                # wall holding the entrance door

    @property
    def area(self) -> float:
        return self.size[0] * self.size[1]


@dataclass
class FlatSpec:
    rooms: list[RoomSpec]

    def room(self, key: str) -> RoomSpec:
        for r in self.rooms:
            if r.key == key:
                return r
        raise KeyError(key)


# --------------------------------------------------------------------------- wall geometry
@dataclass
class WallFrame:
    start: np.ndarray        # (x, y) of the wall's start corner
    dir: np.ndarray          # unit (x, y) along the wall, clockwise from above
    n_in: np.ndarray         # unit (x, y) pointing into the room
    length: float
    axis: int                # world axis normal to the wall (0 = x, 1 = y)
    coord: float             # plane coordinate along that axis
    along: int               # world axis the wall runs along


def wall_frame(room: RoomSpec, wall: str) -> WallFrame:
    ox, oy = room.origin
    sx, sy = room.size
    if wall == "A":
        return WallFrame(np.array([ox, oy + sy]), np.array([1.0, 0.0]), np.array([0.0, -1.0]),
                         sx, 1, oy + sy, 0)
    if wall == "B":
        return WallFrame(np.array([ox + sx, oy + sy]), np.array([0.0, -1.0]), np.array([-1.0, 0.0]),
                         sy, 0, ox + sx, 1)
    if wall == "C":
        return WallFrame(np.array([ox + sx, oy]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]),
                         sx, 1, oy, 0)
    if wall == "D":
        return WallFrame(np.array([ox, oy]), np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                         sy, 0, ox, 1)
    raise ValueError(f"unknown wall {wall!r}")


def opening_interval(room: RoomSpec, op: dict) -> tuple[float, float]:
    """World-coordinate interval of an opening along the wall's `along` axis."""
    wf = wall_frame(room, op["wall"])
    s0 = wf.start[wf.along] + wf.dir[wf.along] * op["offset"]
    s1 = s0 + wf.dir[wf.along] * op["width"]
    return (min(s0, s1), max(s0, s1))


def opening_z(op: dict) -> tuple[float, float]:
    if op["type"] == "door":
        return (0.0, float(op["height"]))
    sill = float(op.get("sill", 0.0))
    return (sill, sill + float(op["height"]))


def opening_centre(room: RoomSpec, op: dict) -> np.ndarray:
    wf = wall_frame(room, op["wall"])
    return wf.start + wf.dir * (op["offset"] + op["width"] / 2.0)


def mirror_offset(room_a: RoomSpec, wall_a: str, offset: float, width: float,
                  room_b: RoomSpec, wall_b: str) -> float:
    """Offset on `room_b.wall_b` that makes an opening coincide with the one on `room_a.wall_a`."""
    fa, fb = wall_frame(room_a, wall_a), wall_frame(room_b, wall_b)
    if fa.axis != fb.axis or abs(fa.coord - fb.coord) > 1e-6:
        raise ValueError(f"{room_a.key}.{wall_a} and {room_b.key}.{wall_b} are not coplanar")
    lo, hi = opening_interval(room_a, {"wall": wall_a, "offset": offset, "width": width})
    e1 = (lo - fb.start[fb.along]) * fb.dir[fb.along]
    e2 = (hi - fb.start[fb.along]) * fb.dir[fb.along]
    off = min(e1, e2)
    if off < -1e-6 or off + width > fb.length + 1e-6:
        raise ValueError(f"opening does not fit on {room_b.key}.{wall_b}")
    return float(off)


def shared_doors(flat: FlatSpec, tol: float = 1e-3) -> list[tuple[str, dict, str, dict]]:
    """Pairs of doors on coincident wall planes with the same hole (within `tol`)."""
    pairs = []
    rooms = flat.rooms
    for i, ra in enumerate(rooms):
        for oa in ra.openings:
            if oa["type"] != "door":
                continue
            fa = wall_frame(ra, oa["wall"])
            ia = opening_interval(ra, oa)
            for rb in rooms[i + 1:]:
                for ob in rb.openings:
                    if ob["type"] != "door":
                        continue
                    fb = wall_frame(rb, ob["wall"])
                    if fa.axis != fb.axis or abs(fa.coord - fb.coord) > tol:
                        continue
                    ib = opening_interval(rb, ob)
                    if abs(ia[0] - ib[0]) <= tol and abs(ia[1] - ib[1]) <= tol \
                            and abs(oa["height"] - ob["height"]) <= tol:
                        pairs.append((ra.key, oa, rb.key, ob))
    return pairs


def check_shared_doors(flat: FlatSpec) -> list[tuple[str, dict, str, dict]]:
    """Assert every door carrying "to" has an exactly coincident partner door. Returns pairs."""
    pairs = shared_doors(flat)
    for r in flat.rooms:
        for op in r.openings:
            if op["type"] == "door" and op.get("to"):
                ok = any((a == r.key and oa is op and b == op["to"]) or
                         (b == r.key and ob is op and a == op["to"]) for a, oa, b, ob in pairs)
                if not ok:
                    raise ValueError(f"door on {r.key}.{op['wall']} (offset {op['offset']}) has no "
                                     f"coincident partner door in {op['to']}")
    return pairs


def entrance_door(room: RoomSpec) -> dict:
    for op in room.openings:
        if op["type"] == "door" and op["wall"] == room.entry_wall:
            return op
    raise ValueError(f"{room.key}: no door on entry wall {room.entry_wall}")


# --------------------------------------------------------------------------- surfaces
@dataclass
class Surface:
    axis: int                 # normal axis
    coord: float
    lo: np.ndarray            # bounds on the two other axes (ascending axis order)
    hi: np.ndarray
    holes: np.ndarray         # (M, 5): lo1, hi1, lo2, hi2, is_door
    kind: str                 # wall | floor | ceiling
    room: str
    wall: str
    shade: float


_WALL_SHADE = {"A": 0.85, "B": 0.72, "C": 0.80, "D": 0.66}


def build_surfaces(flat: FlatSpec) -> list[Surface]:
    surfaces = []
    for r in flat.rooms:
        ox, oy = r.origin
        sx, sy = r.size
        for w in WALLS:
            wf = wall_frame(r, w)
            a1 = wf.along
            lo_a = min(wf.start[a1], wf.start[a1] + wf.dir[a1] * wf.length)
            hi_a = lo_a + wf.length
            holes = []
            for op in r.openings:
                if op["wall"] != w:
                    continue
                i0, i1 = opening_interval(r, op)
                z0, z1 = opening_z(op)
                holes.append([i0, i1, z0, z1, 1.0 if op["type"] == "door" else 0.0])
            surfaces.append(Surface(wf.axis, wf.coord, np.array([lo_a, 0.0]),
                                    np.array([hi_a, r.height]),
                                    np.array(holes, dtype=float).reshape(-1, 5),
                                    "wall", r.key, w, _WALL_SHADE[w]))
        empty = np.zeros((0, 5))
        surfaces.append(Surface(2, 0.0, np.array([ox, oy]), np.array([ox + sx, oy + sy]),
                                empty, "floor", r.key, "", 0.5))
        surfaces.append(Surface(2, r.height, np.array([ox, oy]), np.array([ox + sx, oy + sy]),
                                empty, "ceiling", r.key, "", 0.95))
    return surfaces


# --------------------------------------------------------------------------- ray casting
_OTHER = {0: (1, 2), 1: (0, 2), 2: (0, 1)}


def cast(surfaces: list[Surface], K: np.ndarray, W: int, H: int, R: np.ndarray, p: np.ndarray):
    """Cast one ray per pixel.  Returns (t, sid, t_door, t_win, d) flattened, row-major.

    `t` is the z-depth (rays have unit z in camera coords), inf where nothing is hit.
    `t_door`/`t_win` is the nearest door/window hole crossed by the ray (inf if none).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    d_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=0).reshape(3, -1)
    d = (R @ d_cam).T                                     # (N, 3) world directions
    n = d.shape[0]
    best_t = np.full(n, np.inf)
    best_id = np.full(n, -1, dtype=np.int32)
    t_door = np.full(n, np.inf)
    t_win = np.full(n, np.inf)
    for sid, s in enumerate(surfaces):
        den = d[:, s.axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (s.coord - p[s.axis]) / den
        ok = (t > 1e-6) & (t < best_t)
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


def render_depth(surfaces, R, p, rng, noise_mm: float):
    """Depth (uint16 mm) and confidence (uint8) images plus the clean float depth."""
    t, sid, _, _, d = cast(surfaces, K_DEPTH, DEPTH_W, DEPTH_H, R, p)
    axes = np.array([s.axis for s in surfaces] + [0])
    valid = np.isfinite(t) & (t <= MAX_DEPTH)
    depth = np.where(valid, t, 0.0)
    dn = np.linalg.norm(d, axis=1)
    cosang = np.abs(d[np.arange(len(t)), axes[sid]]) / dn
    sigma = noise_mm / 1000.0 * (depth / 2.0) ** 2
    noisy = depth + rng.normal(size=depth.shape) * sigma
    noisy[~valid] = 0.0
    mm = np.clip(np.rint(noisy * 1000.0), 0, 65535).astype(np.uint16)
    conf = np.where(depth < 3.0, 2, np.where(depth < 5.0, 1, 0)).astype(np.uint8)
    conf[(cosang < 0.2) | ~valid] = 0
    return (mm.reshape(DEPTH_H, DEPTH_W), conf.reshape(DEPTH_H, DEPTH_W),
            depth.reshape(DEPTH_H, DEPTH_W))


_LIGHT = np.array([0.4, 0.25, 0.88])
_LIGHT /= np.linalg.norm(_LIGHT)


def render_rgb(surfaces, R, p, rng) -> np.ndarray:
    """Lambertian gray shading, per-wall base shade, door passages darker, cheap texture."""
    t, sid, t_door, t_win, d = cast(surfaces, K_RGB, RGB_W, RGB_H, R, p)
    hit = np.isfinite(t)
    axes = np.array([s.axis for s in surfaces] + [0])
    shades = np.array([s.shade for s in surfaces] + [0.0])
    ax = axes[sid]
    sign = -np.sign(d[np.arange(len(t)), ax])            # normal faces the camera
    ndotl = sign * _LIGHT[ax]
    inten = shades[sid] * (0.35 + 0.65 * np.clip(ndotl, 0.0, 1.0))
    tt = np.where(hit, t, 0.0)
    pts = p[None, :] + tt[:, None] * d
    cell = np.floor(pts / 0.05)
    hsh = np.sin(cell @ np.array([12.9898, 78.233, 37.719])) * 43758.5453
    hsh = hsh - np.floor(hsh)
    inten = inten * (1.0 + 0.08 * (hsh - 0.5))
    inten[~hit] = 0.5
    inten[~hit & np.isfinite(t_win)] = 0.92
    inten[t_door < t] *= 0.8
    inten += rng.normal(0.0, 0.01, size=inten.shape)
    return np.clip(inten * 255.0, 0, 255).astype(np.uint8).reshape(RGB_H, RGB_W)


# --------------------------------------------------------------------------- rotations / poses
def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def pose_from_euler(x, y, z, yaw, pitch, roll) -> np.ndarray:
    """4x4 camera-to-world (z-up world, OpenCV camera). yaw from +x about +z, pitch +up, roll."""
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll) @ _R0
    T[:3, 3] = (x, y, z)
    return T


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """(qx, qy, qz, qw) with qw >= 0 (Shepperd's method)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


def zup_pose_to_stray(T: np.ndarray) -> tuple[float, ...]:
    """z-up world / OpenCV camera 4x4 -> (x, y, z, qx, qy, qz, qw) in ARKit y-up / OpenGL camera."""
    R_gl = _W @ T[:3, :3] @ _C
    t = _W @ T[:3, 3]
    return (float(t[0]), float(t[1]), float(t[2])) + rot_to_quat(R_gl)


def stray_pose_to_zup(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """Stray Scanner odometry row (ARKit y-up world, OpenGL camera) -> 4x4 z-up / OpenCV camera.

    Exact inverse of `zup_pose_to_stray`:  R_cv = W^T R_gl C,  t_zup = W^T t_ar  with
    W = [[1,0,0],[0,0,1],[0,-1,0]]  (X_ar = x, Y_ar = z, Z_ar = -y)  and  C = diag(1,-1,-1).
    """
    T = np.eye(4)
    T[:3, :3] = _W.T @ quat_to_rot(qx, qy, qz, qw) @ _C
    T[:3, 3] = _W.T @ np.array([x, y, z], dtype=float)
    return T


def write_stray_csv(path: str, timestamps: np.ndarray, poses: np.ndarray) -> None:
    """Write poses (N,4,4 z-up/OpenCV) as a Stray Scanner odometry.csv."""
    with open(path, "w") as f:
        f.write(CSV_HEADER + "\n")
        for i, (ts, T) in enumerate(zip(timestamps, poses)):
            vals = zup_pose_to_stray(T)
            f.write(f"{ts:.6f}, {i}, " + ", ".join(f"{v:.6f}" for v in vals) + "\n")


def load_stray_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read odometry.csv / traj_gt.csv -> (timestamps, frame ids, poses (N,4,4) z-up/OpenCV)."""
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    poses = np.stack([stray_pose_to_zup(*row[2:9]) for row in arr])
    return arr[:, 0], arr[:, 1].astype(int), poses


# --------------------------------------------------------------------------- trajectory
_TURN_RATE = math.radians(90.0)      # rad/s when turning on the spot
_PAUSE_S = 2.5                       # pitch up 25 deg then down -25 deg
_PITCH_AMP = math.radians(25.0)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class _Path:
    """Sequence of (kind, duration, xy0, xy1, yaw0, yaw1) events; yaw is kept continuous."""

    def __init__(self, xy: np.ndarray, yaw: float):
        self.ev: list[tuple] = []
        self.xy = np.asarray(xy, dtype=float)
        self.yaw = yaw

    def hold(self, dur: float):
        self.ev.append(("hold", dur, self.xy, self.xy, self.yaw, self.yaw))

    def pause(self):
        self.ev.append(("pause", _PAUSE_S, self.xy, self.xy, self.yaw, self.yaw))

    def goto(self, xy, speed: float):
        xy = np.asarray(xy, dtype=float)
        dist = float(np.linalg.norm(xy - self.xy))
        if dist < 1e-6:
            return
        yaw_t = self.yaw + _wrap(math.atan2(xy[1] - self.xy[1], xy[0] - self.xy[0]) - self.yaw)
        if abs(yaw_t - self.yaw) > 1e-3:
            self.ev.append(("turn", abs(yaw_t - self.yaw) / _TURN_RATE, self.xy, self.xy,
                            self.yaw, yaw_t))
            self.yaw = yaw_t
        self.ev.append(("move", dist / speed, self.xy, xy, self.yaw, self.yaw))
        self.xy = xy

    def sample(self, fps: float):
        durs = np.array([e[1] for e in self.ev])
        starts = np.concatenate([[0.0], np.cumsum(durs)])
        total = starts[-1]
        n = math.floor(total * fps) + 1
        t = np.arange(n) / fps
        idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(self.ev) - 1)
        xy = np.zeros((n, 2))
        yaw = np.zeros(n)
        pitch = np.zeros(n)
        for i in range(n):
            kind, dur, p0, p1, y0, y1 = self.ev[idx[i]]
            s = min(1.0, (t[i] - starts[idx[i]]) / dur) if dur > 0 else 1.0
            xy[i] = p0 + s * (p1 - p0)
            yaw[i] = y0 + s * (y1 - y0)
            if kind == "pause":
                pitch[i] = _PITCH_AMP * math.sin(2 * math.pi * s)
        return t, xy, yaw, pitch


def _inset(room: RoomSpec) -> float:
    return min(1.0, 0.4 * min(room.size))


def _door_inside_point(room: RoomSpec, op: dict) -> np.ndarray:
    wf = wall_frame(room, op["wall"])
    return opening_centre(room, op) + wf.n_in * min(0.5, 0.4 * min(room.size))


def _room_graph(flat: FlatSpec) -> dict[str, list[tuple[str, dict, dict]]]:
    g: dict[str, list] = {r.key: [] for r in flat.rooms}
    for a, oa, b, ob in check_shared_doors(flat):
        g[a].append((b, oa, ob))
        g[b].append((a, ob, oa))
    return g


def _route(graph, src: str, dst: str) -> list[tuple[str, dict, dict]]:
    """BFS over rooms; returns hops (next_room, door_in_current_room, door_in_next_room)."""
    if src == dst:
        return []
    prev: dict[str, tuple | None] = {src: None}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for nxt, oa, ob in graph[cur]:
            if nxt not in prev:
                prev[nxt] = (cur, oa, ob)
                queue.append(nxt)
                if nxt == dst:
                    hops = []
                    k = dst
                    while prev[k] is not None:
                        c, oa, ob = prev[k]
                        hops.append((k, oa, ob))
                        k = c
                    return hops[::-1]
    raise ValueError(f"no door path from {src} to {dst}")


def build_trajectory(flat: FlatSpec, fps: float, walk_speed: float, rng: np.random.Generator):
    """Hand-held walk through the rooms in order; returns (t, poses (N,4,4) true, info dict)."""
    graph = _room_graph(flat)
    first = flat.rooms[0]
    ent = entrance_door(first)
    start_xy = opening_centre(first, ent)
    n_in = wall_frame(first, ent["wall"]).n_in
    path = _Path(start_xy, math.atan2(n_in[1], n_in[0]))
    path.hold(1.0)
    room_of_frame_events: list[tuple[int, str]] = []
    cur_key = first.key
    for i, room in enumerate(flat.rooms):
        room_of_frame_events.append((len(path.ev), room.key))
        ox, oy = room.origin
        sx, sy = room.size
        m = _inset(room)
        corners = [np.array([ox + m, oy + sy - m]), np.array([ox + sx - m, oy + sy - m]),
                   np.array([ox + sx - m, oy + m]), np.array([ox + m, oy + m])]   # clockwise
        k0 = int(np.argmin([np.linalg.norm(c - path.xy) for c in corners]))
        corners = corners[k0:] + corners[:k0]
        path.goto(corners[0], walk_speed)
        for j in range(4):
            a, b = corners[j], corners[(j + 1) % 4]
            path.goto((a + b) / 2, walk_speed)
            path.pause()
            path.goto(b, walk_speed)
        target = flat.rooms[i + 1].key if i + 1 < len(flat.rooms) else first.key
        for nxt, door_here, door_there in _route(graph, cur_key, target):
            here = flat.room(cur_key)
            there = flat.room(nxt)
            path.goto(_door_inside_point(here, door_here), walk_speed)
            path.goto(opening_centre(here, door_here), walk_speed)
            path.goto(_door_inside_point(there, door_there), walk_speed)
            cur_key = nxt
    path.goto(start_xy, walk_speed)
    path.hold(1.0)

    t, xy, yaw, pitch = path.sample(fps)
    n = len(t)
    yaw = yaw + math.radians(10.0) * np.sin(2 * math.pi * t / 7.0)          # slow sway
    z = CAM_HEIGHT + 0.03 * np.sin(2 * math.pi * 1.8 * t)                   # bob
    roll = np.zeros(n)
    for i in range(1, n):                                                   # AR(1) roll noise
        roll[i] = 0.9 * roll[i - 1] + rng.normal(0.0, math.radians(0.3))
    poses = np.stack([pose_from_euler(xy[i, 0], xy[i, 1], z[i], yaw[i], pitch[i], roll[i])
                      for i in range(n)])
    path_len = float(sum(np.linalg.norm(e[3] - e[2]) for e in path.ev))
    return t, poses, {"path_length_m": path_len, "duration_s": float(t[-1])}


# --------------------------------------------------------------------------- drift
def apply_drift(poses: np.ndarray, rng: np.random.Generator, trans_sigma_mm: float = 1.5,
                yaw_sigma_deg: float = 0.02, yaw_bias_deg: float = 0.001) -> np.ndarray:
    """Odometry = true pose composed with an accumulated error.

    Per frame the yaw error random-walks (sigma `yaw_sigma_deg`) with a constant bias
    `yaw_bias_deg`; the reported position integrates the true displacement rotated by the
    accumulated yaw error plus a translation random walk (sigma `trans_sigma_mm` per axis).
    """
    out = poses.copy()
    yaw_err = 0.0
    p = poses[0, :3, 3].copy()
    for k in range(1, len(poses)):
        yaw_err += math.radians(yaw_bias_deg) + rng.normal(0.0, math.radians(yaw_sigma_deg))
        Rz = rot_z(yaw_err)
        p = p + Rz @ (poses[k, :3, 3] - poses[k - 1, :3, 3]) + rng.normal(0.0, trans_sigma_mm / 1000.0, 3)
        out[k, :3, :3] = Rz @ poses[k, :3, :3]
        out[k, :3, 3] = p
    return out


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """(translation error m, yaw error deg) between two z-up poses."""
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    fa, fb = T_a[:3, 2].copy(), T_b[:3, 2].copy()
    fa[2] = fb[2] = 0.0
    ang = math.degrees(math.atan2(fa[0] * fb[1] - fa[1] * fb[0], float(fa[:2] @ fb[:2])))
    return dt, abs(ang)


# --------------------------------------------------------------------------- ground truth
def _r(v: float) -> float:
    return round(float(v), 4)


def ground_truth(flat: FlatSpec) -> dict:
    rooms = []
    for r in flat.rooms:
        ops = []
        for op in r.openings:
            ops.append({"wall": op["wall"], "type": op["type"], "offset": _r(op["offset"]),
                        "width": _r(op["width"]), "height": _r(op["height"]),
                        "sill": _r(op.get("sill", 0.0)), "to": op.get("to", "")})
        rooms.append({
            "key": r.key, "label": r.label,
            "origin": [_r(r.origin[0]), _r(r.origin[1])],
            "size": [_r(r.size[0]), _r(r.size[1])],
            "walls": {"A": _r(r.size[0]), "B": _r(r.size[1]), "C": _r(r.size[0]), "D": _r(r.size[1])},
            "ceiling_height": _r(r.height),
            "floor_area": _r(r.area),
            "entry_wall": r.entry_wall,
            "openings": ops,
        })
    xs0 = [r.origin[0] for r in flat.rooms]
    ys0 = [r.origin[1] for r in flat.rooms]
    xs1 = [r.origin[0] + r.size[0] for r in flat.rooms]
    ys1 = [r.origin[1] + r.size[1] for r in flat.rooms]
    return {
        "rooms": rooms,
        "property": {
            "n_rooms": len(flat.rooms),
            "footprint_area": _r(sum(r.area for r in flat.rooms)),
            "bbox": {"x_min": _r(min(xs0)), "y_min": _r(min(ys0)),
                     "x_max": _r(max(xs1)), "y_max": _r(max(ys1))},
            "extent": [_r(max(xs1) - min(xs0)), _r(max(ys1) - min(ys0))],
            "walk_order": [r.key for r in flat.rooms],
        },
    }


# --------------------------------------------------------------------------- generate
def generate(flat: FlatSpec, out_dir: str, *, seed: int = 0, drift: bool = True,
             noise_mm: float = 8.0, fps: float = 10.0, walk_speed: float = 0.5,
             drift_trans_sigma_mm: float = 1.5, drift_yaw_sigma_deg: float = 0.02,
             drift_yaw_bias_deg: float = 0.001, reuse_frames_from: str | None = None,
             verbose: bool = True) -> dict:
    """Render a Stray Scanner style dataset into `out_dir`; returns the ground truth dict.

    `reuse_frames_from`: hard-link depth/confidence/rgb from a dataset generated with the
    same flat, seed, fps, walk_speed and noise (only odometry differs, e.g. drift ablation).
    """
    check_shared_doors(flat)
    rng_traj = np.random.default_rng([seed, 0])
    rng_render = np.random.default_rng([seed, 1])
    rng_drift = np.random.default_rng([seed, 2])

    t, poses_true, info = build_trajectory(flat, fps, walk_speed, rng_traj)
    n = len(t)
    poses_odom = poses_true
    if drift:
        poses_odom = apply_drift(poses_true, rng_drift, drift_trans_sigma_mm, drift_yaw_sigma_deg,
                                 drift_yaw_bias_deg)

    os.makedirs(out_dir, exist_ok=True)
    for sub in ("depth", "confidence", "rgb"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    with open(os.path.join(out_dir, "camera_matrix.csv"), "w") as f:
        f.writelines(", ".join(f"{v:.6f}" for v in row) + "\n" for row in K_RGB)
    write_stray_csv(os.path.join(out_dir, "odometry.csv"), t, poses_odom)
    write_stray_csv(os.path.join(out_dir, "traj_gt.csv"), t, poses_true)

    if reuse_frames_from:
        for sub, ext in (("depth", "png"), ("confidence", "png"), ("rgb", "jpg")):
            src_dir = os.path.join(reuse_frames_from, sub)
            if len(os.listdir(src_dir)) != n:
                raise ValueError(f"{src_dir}: frame count differs from {n}")
            for i in range(n):
                src = os.path.join(src_dir, f"{i:06d}.{ext}")
                dst = os.path.join(out_dir, sub, f"{i:06d}.{ext}")
                if os.path.exists(dst):
                    os.remove(dst)
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
    else:
        surfaces = build_surfaces(flat)
        for i in range(n):
            R, p = poses_true[i, :3, :3], poses_true[i, :3, 3]
            mm, conf, _ = render_depth(surfaces, R, p, rng_render, noise_mm)
            gray = render_rgb(surfaces, R, p, rng_render)
            cv2.imwrite(os.path.join(out_dir, "depth", f"{i:06d}.png"), mm)
            cv2.imwrite(os.path.join(out_dir, "confidence", f"{i:06d}.png"), conf)
            cv2.imwrite(os.path.join(out_dir, "rgb", f"{i:06d}.jpg"), gray,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            if verbose and (i % 200 == 0 or i == n - 1):
                print(f"  [{os.path.basename(out_dir)}] frame {i + 1}/{n}", flush=True)

    end_dt, end_dyaw = pose_error(poses_odom[-1], poses_true[-1])
    gt = ground_truth(flat)
    gt.update({
        "generator": "floorplan.io.synthetic",
        "seed": int(seed),
        "fps": float(fps),
        "n_frames": int(n),
        "duration_s": _r(info["duration_s"]),
        "path_length_m": _r(info["path_length_m"]),
        "walk_speed": float(walk_speed),
        "noise_mm": float(noise_mm),
        "camera_height_m": CAM_HEIGHT,
        "max_depth_m": MAX_DEPTH,
        "depth_intrinsics": {"width": DEPTH_W, "height": DEPTH_H, "fx": DEPTH_F, "fy": DEPTH_F,
                             "cx": 128.0, "cy": 96.0},
        "rgb_intrinsics": {"width": RGB_W, "height": RGB_H, "fx": float(K_RGB[0, 0]),
                           "fy": float(K_RGB[1, 1]), "cx": float(K_RGB[0, 2]),
                           "cy": float(K_RGB[1, 2])},
        "drift": {"enabled": bool(drift), "trans_sigma_mm_per_frame": float(drift_trans_sigma_mm),
                  "yaw_sigma_deg_per_frame": float(drift_yaw_sigma_deg),
                  "yaw_bias_deg_per_frame": float(drift_yaw_bias_deg),
                  "end_translation_error_m": _r(end_dt), "end_yaw_error_deg": _r(end_dyaw)},
        "conventions": {
            "world": "ground_truth.yaml geometry is z-up metres; rooms are axis-aligned boxes",
            "odometry_csv": "Stray Scanner: ARKit y-up world, OpenGL camera (x right, y up, "
                            "looks down -z), camera-to-world, quaternion qx qy qz qw",
            "traj_gt_csv": "TRUE poses, same format/convention as odometry.csv; convert with "
                           "floorplan.io.synthetic.stray_pose_to_zup",
            "zup_from_arkit": "x = X, y = -Z, z = Y ; R_cv = W^T R_gl diag(1,-1,-1)",
            "walls": "A max-y, B max-x, C min-y, D min-x; offsets clockwise from above "
                     "(from the wall's left corner seen from inside)",
        },
    })
    with open(os.path.join(out_dir, "ground_truth.yaml"), "w") as f:
        yaml.safe_dump(gt, f, sort_keys=False)
    return gt
