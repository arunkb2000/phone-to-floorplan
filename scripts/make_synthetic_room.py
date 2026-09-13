"""Render a synthetic room in Stray Scanner format, with exactly known dimensions and staged damage.

Three requirements could not be met on the supplied captures because they need physical access to
the property: tape ground truth, a furnished room with staged damage spanning two classes, and the
head-to-head. This closes the first two on data where the truth is not measured but *defined*, which
is stronger than a tape and weaker than a real room, and is labelled as such everywhere it appears.

What it proves: that the geometry recovers dimensions it was never told, that the wall-length gates
score, and that the damage path detects two classes and measures their extent against known values.
What it cannot prove: anything about real sensor noise, real surfaces, or real clutter. The supplied
captures carry that half.

    uv run python scripts/make_synthetic_room.py --out data/benchmark/captures/synth_room
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image

DEPTH_W, DEPTH_H = 256, 192
RGB_W, RGB_H = 960, 720
FX_D = 212.0
CAM_H = 1.45

# --- the room, defined exactly. Every number below is ground truth. -----------------------------
ROOM = dict(
    lx=4.260,          # wall A and C, along x
    ly=3.180,          # wall B and D, along y
    h=2.640,           # floor to ceiling
    door=dict(wall="C", offset=0.680, width=0.915, height=2.045),      # C is the y=0 wall
    window=dict(wall="A", offset=1.520, width=1.240, height=1.180, sill=0.900),
)
# two damage classes, positioned and sized exactly
DAMAGE = [
    dict(cls="water_stain", surface="ceiling", u=1.60, v=1.15, w=0.520, hgt=0.380),
    dict(cls="crack", surface="wall:B", u=1.05, v=1.35, w=0.040, hgt=0.870),
]

_W = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # z-up -> ARKit y-up is _W.T


def rot_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return x, y, z, w


def surfaces(r):
    """(axis, coord, sign, rect_bounds, holes, name). Normal points into the room."""
    lx, ly, h = r["lx"], r["ly"], r["h"]
    d, w = r["door"], r["window"]
    holes_C = [(d["offset"], d["offset"] + d["width"], 0.0, d["height"])]
    holes_A = [(w["offset"], w["offset"] + w["width"], w["sill"], w["sill"] + w["height"])]
    return [
        (1, 0.0, +1, (0, lx, 0, h), holes_C, "C"),      # y = 0
        (1, ly, -1, (0, lx, 0, h), holes_A, "A"),       # y = ly
        (0, 0.0, +1, (0, ly, 0, h), [], "D"),           # x = 0
        (0, lx, -1, (0, ly, 0, h), [], "B"),            # x = lx
        (2, 0.0, +1, (0, lx, 0, ly), [], "floor"),
        (2, h, -1, (0, lx, 0, ly), [], "ceiling"),
    ]


def cast(surfs, origin, dirs):
    """Nearest hit per ray. Returns (t, surface index, u, v on that surface)."""
    n = dirs.shape[0]
    best_t = np.full(n, np.inf)
    best_s = np.full(n, -1, np.int32)
    best_u = np.zeros(n)
    best_v = np.zeros(n)
    for si, (ax, coord, _sgn, (a0, a1, b0, b1), holes, _nm) in enumerate(surfs):
        den = dirs[:, ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (coord - origin[ax]) / den
        ok = np.isfinite(t) & (t > 0.05) & (t < best_t)
        if not ok.any():
            continue
        hit = origin + np.where(np.isfinite(t), t, 0.0)[:, None] * dirs
        if ax == 2:
            u, v = hit[:, 0], hit[:, 1]
        elif ax == 1:
            u, v = hit[:, 0], hit[:, 2]
        else:
            u, v = hit[:, 1], hit[:, 2]
        ok &= (u >= a0) & (u <= a1) & (v >= b0) & (v <= b1)
        for hu0, hu1, hv0, hv1 in holes:
            ok &= ~((u >= hu0) & (u <= hu1) & (v >= hv0) & (v <= hv1))
        best_t = np.where(ok, t, best_t)
        best_s = np.where(ok, si, best_s)
        best_u = np.where(ok, u, best_u)
        best_v = np.where(ok, v, best_v)
    return best_t, best_s, best_u, best_v


BASE_RGB = {"C": (231, 227, 217), "A": (234, 230, 220), "D": (226, 222, 212),
            "B": (236, 232, 222), "floor": (139, 118, 96), "ceiling": (243, 241, 236)}
SKIRTING = (222, 219, 212)


def _wobble(u, v, k1=7.0, k2=11.0):
    """Deterministic irregularity, so a stain has a ragged edge rather than a drawn one."""
    return (0.18 * np.sin(k1 * u + 1.3) + 0.14 * np.sin(k2 * v - 0.7)
            + 0.09 * np.sin(3.1 * u + 5.0 * v) + 0.06 * np.sin(17.0 * v + 2.2))


def shade(surfs, sid, u, v, rng):
    """Colour appearance in each surface's own uv frame, with the damage decals composited in.

    A flat grey blob is not a water stain to any detector, so the stain gets an irregular edge and a
    darker core, and the crack gets a jagged path with a branch. Nothing here is photoreal; it is
    enough that the classes look like what they are called.
    """
    img = np.zeros(sid.shape + (3,), np.float32)
    for si, (_ax, _c, _s, _r, _h, nm) in enumerate(surfs):
        m = sid == si
        if not m.any():
            continue
        img[m] = BASE_RGB[nm]
        if nm in ("A", "B", "C", "D"):
            img[m & (v < 0.115)] = SKIRTING
            img[m & (v > 0.0) & (v < 0.012)] = (120, 116, 110)

    for d in DAMAGE:
        nm = d["surface"].split(":")[-1] if ":" in d["surface"] else d["surface"]
        si = next(i for i, srf in enumerate(surfs) if srf[5] == nm)
        m = sid == si
        if not m.any():
            continue
        if d["cls"] == "water_stain":
            du = (u - d["u"]) / max(d["w"] / 2, 1e-6)
            dv = (v - d["v"]) / max(d["hgt"] / 2, 1e-6)
            rad = np.sqrt(du ** 2 + dv ** 2) - _wobble(u * 6, v * 6)
            edge = np.clip(1.15 - rad, 0.0, 1.0)
            inside = m & (rad < 1.0)
            a = (edge ** 1.6)[inside][:, None]
            stain = np.array([150.0, 106.0, 58.0])                  # ochre brown
            core = np.array([112.0, 74.0, 38.0])
            tone = stain + (core - stain) * np.clip(1.0 - rad[inside], 0, 1)[:, None]
            img[inside] = img[inside] * (1 - 0.88 * a) + tone * (0.88 * a)
        else:
            # a jagged crack: the path wanders in u as it runs down v, and throws one branch
            path = d["u"] + 0.035 * np.sin(9.0 * v) + 0.018 * np.sin(23.0 * v + 1.1)
            band = (np.abs(u - path) < d["w"] / 2) & (np.abs(v - d["v"]) < d["hgt"] / 2)
            branch_v = d["v"] + d["hgt"] * 0.18
            bpath = path + (v - branch_v) * 0.55
            branch = (np.abs(u - bpath) < d["w"] / 2.4) & (v > branch_v) & (v < branch_v + d["hgt"] * 0.3)
            inside = m & (band | branch)
            img[inside] = np.array([38.0, 34.0, 31.0])
            halo = m & (np.abs(u - path) < d["w"] * 1.6) & (np.abs(v - d["v"]) < d["hgt"] / 2) & ~inside
            img[halo] = img[halo] * 0.86

    # a little shading so surfaces are not perfectly flat, then sensor noise
    img *= (0.88 + 0.12 * np.clip(1.0 - np.abs(v - 1.3) / 3.0, 0, 1))[..., None]
    img[sid < 0] = (18, 18, 20)
    return np.clip(img + rng.normal(0, 2.2, img.shape), 0, 255).astype(np.uint8)


def trajectory(r, per_edge=14, spin=20):
    """The capture protocol, walked: the perimeter one metre in, plus a full sweep at each corner.

    Facing only the wall you are walking along leaves the opposite wall unseen, which is exactly the
    mistake the protocol tells a person not to make, so the corner sweeps are here for the same
    reason they are on the page.
    """
    lx, ly = r["lx"], r["ly"]
    inset = min(1.0, min(lx, ly) / 2 - 0.35)
    corners = [(inset, inset), (lx - inset, inset), (lx - inset, ly - inset), (inset, ly - inset)]
    poses = []
    for i in range(4):
        a, b = np.array(corners[i]), np.array(corners[(i + 1) % 4])
        # sweep the whole room from this corner, tilting up and down as the protocol asks
        for k in range(spin):
            f = k / spin
            yaw = 2 * np.pi * f
            pitch = np.radians(32.0 * np.sin(4 * np.pi * f))
            poses.append((np.array([a[0], a[1], CAM_H + 0.015 * np.sin(7 * np.pi * f)]), yaw, pitch))
        # then walk to the next corner, looking ahead
        travel = np.arctan2(b[1] - a[1], b[0] - a[0])
        for k in range(1, per_edge):
            f = k / per_edge
            p = a + (b - a) * f
            yaw = travel + np.radians(28.0 * np.sin(2 * np.pi * f))
            pitch = np.radians(20.0 * np.sin(2 * np.pi * f))
            poses.append((np.array([p[0], p[1], CAM_H + 0.02 * np.sin(9 * np.pi * f)]), yaw, pitch))
    return poses


def pose_matrix(yaw, pitch):
    """Camera-to-world, OpenCV axes (x right, y down, z forward), z-up world."""
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/benchmark/captures/synth_room")
    ap.add_argument("--gt", default="data/benchmark/ground_truth/synth_room.yaml")
    ap.add_argument("--noise-mm", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    r = ROOM
    surfs = surfaces(r)
    os.makedirs(f"{args.out}/depth", exist_ok=True)
    os.makedirs(f"{args.out}/confidence", exist_ok=True)
    os.makedirs(f"{args.out}/rgb", exist_ok=True)

    Kd = np.array([[FX_D, 0, DEPTH_W / 2], [0, FX_D, DEPTH_H / 2], [0, 0, 1]])
    scale = RGB_W / DEPTH_W
    Krgb = np.array([[FX_D * scale, 0, RGB_W / 2], [0, FX_D * scale, RGB_H / 2], [0, 0, 1]])
    np.savetxt(f"{args.out}/camera_matrix.csv", Krgb, delimiter=", ", fmt="%.4f")

    ys, xs = np.mgrid[0:DEPTH_H, 0:DEPTH_W]
    d_cam = np.stack([(xs - Kd[0, 2]) / FX_D, (ys - Kd[1, 2]) / FX_D, np.ones_like(xs, float)], -1).reshape(-1, 3)
    ysr, xsr = np.mgrid[0:RGB_H, 0:RGB_W]
    r_cam = np.stack([(xsr - Krgb[0, 2]) / Krgb[0, 0], (ysr - Krgb[1, 2]) / Krgb[1, 1],
                      np.ones_like(xsr, float)], -1).reshape(-1, 3)

    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw"]
    for i, (pos, yaw, pitch) in enumerate(trajectory(r)):
        R = pose_matrix(yaw, pitch)
        t, sid, _u, _v = cast(surfs, pos, d_cam @ R.T)
        depth = np.where(np.isfinite(t), t, 0.0).reshape(DEPTH_H, DEPTH_W)
        depth = depth + rng.normal(0, args.noise_mm / 1000.0, depth.shape) * np.clip(depth / 2.0, 0.3, 3.0)
        depth = np.clip(depth, 0, 8.0)
        conf = np.where(depth > 0, np.where(depth < 3.0, 2, 1), 0).astype(np.uint8)
        Image.fromarray((depth * 1000).astype(np.uint16)).save(f"{args.out}/depth/{i:06d}.png")
        Image.fromarray(conf).save(f"{args.out}/confidence/{i:06d}.png")

        tr, sr, ur, vr = cast(surfs, pos, r_cam @ R.T)
        rgb = shade(surfs, sr, ur, vr, rng).reshape(RGB_H, RGB_W, 3)
        cv2.imwrite(f"{args.out}/rgb/{i:06d}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        R_ar = _W.T @ R
        q = rot_to_quat(R_ar)
        p_ar = _W.T @ pos
        rows.append(f"{i * 0.1:.6f}, {i:06d}, {p_ar[0]:.8f}, {p_ar[1]:.8f}, {p_ar[2]:.8f}, "
                    f"{q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}")
    open(f"{args.out}/odometry.csv", "w").write("\n".join(rows) + "\n")

    gt = [
        "# Ground truth for the synthetic room. These numbers are DEFINED, not measured: the renderer",
        "# was given them and the pipeline was not. See scripts/make_synthetic_room.py.",
        "capture: synth_room",
        "method: exact, by construction (synthetic render)",
        "ceiling_height:",
        f"  - {{value: {r['h']:.3f}, n_frames: 1, spread_m: 0.0, sigma_m: 0.0}}",
        "wall_to_wall:",
        f"  - {{value: {r['lx']:.3f}, n_frames: 1, spread_m: 0.0, sigma_m: 0.0}}   # walls A and C",
        f"  - {{value: {r['ly']:.3f}, n_frames: 1, spread_m: 0.0, sigma_m: 0.0}}   # walls B and D",
        "opening_width:",
        f"  - {{value: {r['door']['width']:.3f}, n_frames: 1, spread_m: 0.0, sigma_m: 0.0}}   # door on wall C",
        f"  - {{value: {r['window']['width']:.3f}, n_frames: 1, spread_m: 0.0, sigma_m: 0.0}}   # window on wall A",
        "damage:",
    ]
    for d in DAMAGE:
        gt.append(f"  - {{class: {d['cls']}, surface: '{d['surface']}', u: {d['u']:.3f}, v: {d['v']:.3f}, "
                  f"width_m: {d['w']:.3f}, height_m: {d['hgt']:.3f}, area_m2: {d['w'] * d['hgt']:.4f}}}")
    open(args.gt, "w").write("\n".join(gt) + "\n")

    print(f"{len(rows) - 1} frames -> {args.out}")
    print(f"ground truth -> {args.gt}")
    print(f"  room {r['lx']} x {r['ly']} x {r['h']} m, door {r['door']['width']} m, window {r['window']['width']} m")
    print(f"  damage: {', '.join(d['cls'] + ' ' + format(d['w'] * d['hgt'], '.3f') + ' m2' for d in DAMAGE)}")


if __name__ == "__main__":
    main()
