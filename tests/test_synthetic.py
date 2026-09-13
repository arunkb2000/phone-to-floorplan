"""Synthetic Stray Scanner generator: file layout, depth accuracy, pose conventions."""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
import pytest
import yaml

from floorplan.io import synthetic as syn

ROOM = syn.RoomSpec("01_room", "room", (0.0, 0.0), (3.2, 3.5), 2.5, [
    {"wall": "C", "type": "door", "offset": 0.5, "width": 0.9, "height": 2.05},
    {"wall": "A", "type": "window", "offset": 0.2, "width": 0.8, "height": 1.0, "sill": 1.0},
], entry_wall="C")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("syn") / "synthetic_test")
    gt = syn.generate(syn.FlatSpec([ROOM]), out, seed=3, drift=True, noise_mm=2.0, fps=2.0,
                      walk_speed=1.0, verbose=False)
    return out, gt


def _read_depth(out, i):
    d = cv2.imread(os.path.join(out, "depth", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
    assert d is not None
    return d


def test_layout_and_dtypes(dataset):
    out, gt = dataset
    n = gt["n_frames"]
    assert n > 10
    for sub in ("depth", "confidence", "rgb"):
        assert len(os.listdir(os.path.join(out, sub))) == n
    d = _read_depth(out, 0)
    assert d.shape == (192, 256) and d.dtype == np.uint16
    c = cv2.imread(os.path.join(out, "confidence", "000000.png"), cv2.IMREAD_UNCHANGED)
    assert c.shape == (192, 256) and c.dtype == np.uint8 and set(np.unique(c)) <= {0, 1, 2}
    rgb = cv2.imread(os.path.join(out, "rgb", "000000.jpg"))
    assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
    K = np.loadtxt(os.path.join(out, "camera_matrix.csv"), delimiter=",")
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(212 * 640 / 256) and K[0, 2] == 320 and K[1, 2] == 240
    with open(os.path.join(out, "odometry.csv")) as f:
        assert f.readline().strip() == "timestamp, frame, x, y, z, qx, qy, qz, qw"
    ts, frames, poses = syn.load_stray_csv(os.path.join(out, "odometry.csv"))
    assert poses.shape == (n, 4, 4) and frames[-1] == n - 1
    assert ts[1] - ts[0] == pytest.approx(0.5)
    with open(os.path.join(out, "ground_truth.yaml")) as f:
        y = yaml.safe_load(f)
    r = y["rooms"][0]
    assert r["walls"] == {"A": 3.2, "B": 3.5, "C": 3.2, "D": 3.5}
    assert r["ceiling_height"] == 2.5 and r["floor_area"] == pytest.approx(11.2)
    assert y["property"]["footprint_area"] == pytest.approx(11.2)
    assert y["property"]["bbox"] == {"x_min": 0.0, "y_min": 0.0, "x_max": 3.2, "y_max": 3.5}


def test_centre_depth_matches_wall_ahead(dataset):
    out, _ = dataset
    _, _, poses = syn.load_stray_csv(os.path.join(out, "traj_gt.csv"))
    T = poses[0]
    fwd, cam = T[:3, 2], T[:3, 3]
    # first frame: in the entrance door (wall C) looking in; the optical axis hits wall A (y = 3.5)
    expected = (ROOM.origin[1] + ROOM.size[1] - cam[1]) / fwd[1]
    d = _read_depth(out, 0).astype(float) / 1000.0
    centre = np.median(d[94:99, 126:131])
    assert centre > 0
    assert abs(centre - expected) < 0.03, (centre, expected)


def _backproject(depth_m, T):
    v, u = np.mgrid[0:192, 0:256]
    K = syn.K_DEPTH
    x = (u - K[0, 2]) / K[0, 0] * depth_m
    y = (v - K[1, 2]) / K[1, 1] * depth_m
    pts = np.stack([x, y, depth_m], axis=-1).reshape(-1, 3)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def _analytic_plane_pixels(T, z_plane, room, margin=0.05):
    """Pixels whose ray reaches the horizontal plane z=z_plane inside the room footprint."""
    v, u = np.mgrid[0:192, 0:256]
    K = syn.K_DEPTH
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)],
                     axis=-1).reshape(-1, 3)
    d = (T[:3, :3] @ d_cam.T).T
    cam = T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_plane - cam[2]) / d[:, 2]
    hit = cam[None, :] + t[:, None] * d
    ox, oy = room.origin
    sx, sy = room.size
    inside = ((hit[:, 0] > ox + margin) & (hit[:, 0] < ox + sx - margin) &
              (hit[:, 1] > oy + margin) & (hit[:, 1] < oy + sy - margin))
    return (t > 0) & np.isfinite(t) & inside


def test_backprojection_floor_and_ceiling(dataset):
    out, _ = dataset
    _, _, poses = syn.load_stray_csv(os.path.join(out, "traj_gt.csv"))
    pitch = np.arcsin(poses[:, 2, 2])          # forward vector z component
    checked = 0
    for idx, z_plane in ((int(np.argmin(pitch)), 0.0), (int(np.argmax(pitch)), ROOM.height)):
        T = poses[idx]
        depth = _read_depth(out, idx).astype(float) / 1000.0
        pts = _backproject(depth, T)
        sel = _analytic_plane_pixels(T, z_plane, ROOM) & (depth.reshape(-1) > 0)
        assert sel.sum() > 500, (idx, z_plane, sel.sum())
        err = np.abs(pts[sel, 2] - z_plane)
        assert np.median(err) < 0.01, (z_plane, np.median(err))
        assert np.percentile(err, 95) < 0.02, (z_plane, np.percentile(err, 95))
        checked += 1
    assert checked == 2


def test_pose_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = syn.pose_from_euler(*rng.normal(size=3), rng.uniform(-math.pi, math.pi),
                                rng.uniform(-1.2, 1.2), rng.uniform(-0.5, 0.5))
        back = syn.stray_pose_to_zup(*syn.zup_pose_to_stray(T))
        assert np.allclose(back, T, atol=1e-9)
    # level camera at 1.4 m looking along our +y is the ARKit identity rotation at (0, 1.4, 0)
    T = syn.pose_from_euler(0.0, 0.0, 1.4, math.pi / 2, 0.0, 0.0)
    x, y, z, qx, qy, qz, qw = syn.zup_pose_to_stray(T)
    assert (x, y, z) == pytest.approx((0.0, 1.4, 0.0), abs=1e-12)
    assert (qx, qy, qz, qw) == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)
    # and its forward axis (OpenCV +z) is world +y, right (+x) is world +x, down (+y) is world -z
    assert np.allclose(T[:3, :3], np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), atol=1e-12)


def test_drift_only_when_enabled(dataset):
    out, gt = dataset
    _, _, odom = syn.load_stray_csv(os.path.join(out, "odometry.csv"))
    _, _, true = syn.load_stray_csv(os.path.join(out, "traj_gt.csv"))
    assert np.allclose(odom[0], true[0])
    assert not np.allclose(odom[-1], true[-1], atol=1e-6)
    dt, dyaw = syn.pose_error(odom[-1], true[-1])
    assert gt["drift"]["enabled"] and dt < 0.5 and dyaw < 5.0


def test_shared_door_mirroring():
    a = syn.RoomSpec("a", "a", (0.0, 0.0), (4.2, 3.6), 2.7, [
        {"wall": "B", "type": "door", "offset": 1.2, "width": 0.85, "height": 2.05, "to": "b"}])
    b = syn.RoomSpec("b", "b", (4.2, 0.3), (2.7, 3.0), 2.7, [], entry_wall="D")
    off = syn.mirror_offset(a, "B", 1.2, 0.85, b, "D")
    assert off == pytest.approx(1.25)
    b.openings.append({"wall": "D", "type": "door", "offset": off, "width": 0.85, "height": 2.05,
                       "to": "a"})
    assert len(syn.check_shared_doors(syn.FlatSpec([a, b]))) == 1
    b.openings[0]["offset"] = 1.0
    with pytest.raises(ValueError):
        syn.check_shared_doors(syn.FlatSpec([a, b]))
