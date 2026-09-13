"""Unit tests for the per-frame geometry on an analytically ray-cast box room (no learned models)."""
import numpy as np

from floorplan.geometry.frame import (
    analyse_frame,
    backproject,
    best_rotation,
    rotate_geom,
    rotz,
    wall_openings,
)


def box_depth(cam_pos, yaw_deg, pitch_deg, Lx, Ly, H, K, W, Hpx, door=None, noise=0.0, seed=0):
    """Depth image of an axis-aligned room [0,Lx]x[0,Ly]x[0,H] seen from cam_pos with yaw (about z)
    and pitch. Camera: OpenCV (x right, y down, z forward). door=(wall, u0, u1, top) punches a hole in
    that wall; rays through it continue to a parallel surface 3 m behind."""
    ys, xs = np.mgrid[0:Hpx, 0:W]
    d = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    base = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)      # cam z → world +y, cam y → world -z
    pr = np.radians(pitch_deg)
    Rp = np.array([[1, 0, 0], [0, np.cos(pr), -np.sin(pr)], [0, np.sin(pr), np.cos(pr)]])
    R = rotz(np.radians(yaw_deg)) @ base @ Rp
    dw = d @ R.T
    c = np.asarray(cam_pos, float)
    t_best = np.full(d.shape[:2], np.inf)
    planes = [((1, 0, 0), Lx, "B"), ((-1, 0, 0), 0.0, "D"), ((0, 1, 0), Ly, "A"), ((0, -1, 0), 0.0, "C"),
              ((0, 0, 1), H, "ceil"), ((0, 0, -1), 0.0, "floor")]
    for n, v, name in planes:
        n = np.array(n, float)
        denom = dw @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (v - c @ n) / denom
        t = np.where((denom > 1e-9) & (t > 0.05), t, np.inf)
        if door is not None and door[0] == name:
            hit = c + t[..., None] * dw
            lat = hit[..., 0] if name in "AC" else hit[..., 1]
            hole = (lat >= door[1]) & (lat <= door[2]) & (hit[..., 2] <= door[3]) & (hit[..., 2] >= 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                t2 = (v + 3.0 - c @ n) / denom
            t = np.where(hole, t2, t)
        t_best = np.minimum(t_best, t)
    depth = t_best.copy()                       # d has z = 1, so t is the z-depth
    if noise:
        depth = depth + np.random.default_rng(seed).normal(0, noise, depth.shape) * depth
    return depth.astype(np.float32)


K = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], float)   # HFOV 94°, VFOV 77°


def test_box_room_frame_geometry():
    Lx, Ly, H = 4.0, 3.0, 2.7
    depth = box_depth((1.5, 0.5, 1.4), yaw_deg=25, pitch_deg=0, Lx=Lx, Ly=Ly, H=H, K=K, W=640, Hpx=480, noise=0.003)
    P = backproject(depth, K)
    g = analyse_frame(P, up_hint=np.array([0, -1.0, 0]), depth_rel_err=0.01)
    assert g is not None
    assert abs(g.cam_height - 1.4) < 0.03, g.cam_height
    assert abs(g.ceil_above - (H - 1.4)) < 0.03, g.ceil_above
    # yaw 25° (view toward (-sin25, cos25)) → wall D (x=0) at 1.5 and wall A at 2.5 from the camera
    dists = sorted(round(w["dist"], 2) for w in g.walls.values())
    assert any(abs(d - 1.5) < 0.05 for d in dists), dists
    assert any(abs(d - 2.5) < 0.05 for d in dists), dists
    # orientation resolution: expected view direction → after rotation, -x wall is D(1.5), +y is A(2.5)
    k = best_rotation(g, (-np.sin(np.radians(25)), np.cos(np.radians(25))))
    g2 = rotate_geom(g, k)
    assert abs(g2.walls["-x"]["dist"] - 1.5) < 0.05 and abs(g2.walls["+y"]["dist"] - 2.5) < 0.05


def test_door_detection_width():
    Lx, Ly, H = 4.0, 3.0, 2.7
    # door on wall A from x=1.5 to x=2.4 (0.9 m), 2.05 m tall; camera looks straight at A
    depth = box_depth((2.0, 0.3, 1.4), yaw_deg=0, pitch_deg=-5, Lx=Lx, Ly=Ly, H=H, K=K, W=640, Hpx=480,
                      door=("A", 1.5, 2.4, 2.05), noise=0.002)
    P = backproject(depth, K)
    g = analyse_frame(P, up_hint=np.array([0, -1.0, 0]), depth_rel_err=0.01)
    k = best_rotation(g, (0, 1))
    g = rotate_geom(g, k)
    assert "+y" in g.walls and abs(g.walls["+y"]["dist"] - 2.7) < 0.05
    ops = wall_openings(g.P, "+y", g.walls["+y"]["dist"], g.cam_height)
    doors = [o for o in ops if o["type"] == "door"]
    assert len(doors) == 1, ops
    d = doors[0]
    assert abs(d["width"] - 0.9) <= 0.06, d
    # lateral position: camera at x=2.0 → hole spans u ∈ [-0.5, 0.4]
    assert abs(d["u0"] - (-0.5)) < 0.08 and abs(d["u1"] - 0.4) < 0.08, d
    assert abs(d["top"] - 2.05) < 0.12, d
