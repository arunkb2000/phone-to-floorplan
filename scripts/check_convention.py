"""Evidence for the pose convention used by the LiDAR loader.

Stray Scanner's odometry.csv documents an ARKit camera-to-world transform. ARKit's own camera axes
are OpenGL-style (x right, y up, -z forward), so the textbook conversion to OpenCV axes is
diag(1, -1, -1). On the sample captures that conversion is wrong: the exported rotation is already in
OpenCV axes. This script scores each candidate convention by how sharply the vertical-normal points
collapse onto a floor plane, and prints the table quoted in the technical report.

    uv run python scripts/check_convention.py Dataset/single_room
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from PIL import Image

from floorplan.geometry.frame import backproject, normals_from_grid


def quat_R(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main(d: str, stride: int = 40):
    K = np.loadtxt(os.path.join(d, "camera_matrix.csv"), delimiter=",").reshape(3, 3)
    Kd = K.copy()
    Kd[0] *= 256 / (2 * K[0, 2])
    Kd[1] *= 192 / (2 * K[1, 2])
    rows = list(csv.reader(open(os.path.join(d, "odometry.csv"))))[1:][::stride]
    data = []
    for r in rows:
        fi = int(float(r[1]))
        dp = os.path.join(d, "depth", f"{fi:06d}.png")
        cp = os.path.join(d, "confidence", f"{fi:06d}.png")
        if not os.path.exists(dp):
            continue
        data.append((np.asarray(Image.open(dp), np.float32) / 1000, np.asarray(Image.open(cp)),
                     np.array([float(r[2]), float(r[3]), float(r[4])]), quat_R(*[float(v) for v in r[5:9]])))
    W_yup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    print(f"{len(data)} frames from {d}\n")
    print("| world map | camera axes | vertical-normal points | sharpest 3 cm height bin | floor z |")
    print("|---|---|---|---|---|")
    for wn, W in (("ARKit y-up -> z-up", W_yup), ("identity", np.eye(3))):
        for cn, C in (("OpenGL -> OpenCV", np.diag([1., -1, -1])), ("as exported", np.eye(3))):
            P, N = [], []
            for depth, conf, t, Rg in data:
                p = backproject(depth, Kd)
                n = normals_from_grid(p, k=3)
                R = W @ Rg @ C
                ok = np.isfinite(p).all(-1) & np.isfinite(n).all(-1) & (conf >= 2) & (depth < 4.5) & (depth > 0.3)
                P.append(p[ok] @ R.T + W @ t)
                N.append(n[ok] @ R.T)
            P, N = np.concatenate(P), np.concatenate(N)
            vert = np.abs(N[:, 2]) > 0.9
            h, e = np.histogram(P[vert, 2], bins=200, range=(-3, 3))
            print(f"| {wn} | {cn} | {vert.mean():.1%} | {h.max() / max(vert.sum(), 1):.3f} | {e[np.argmax(h)]:+.2f} m |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Dataset/single_room")
