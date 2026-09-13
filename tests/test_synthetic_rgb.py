"""Photo/video-tier synthetic captures: files, EXIF, palm-cover segmentation, staged damage."""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from make_synthetic import flat_mr1, flat_r1  # noqa: E402

from floorplan.io import synthetic as syn  # noqa: E402
from floorplan.io import synthetic_rgb as rgb  # noqa: E402
from floorplan.io.video import load_video_capture  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


@pytest.fixture(scope="module")
def photo_set(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("rgb") / "synthetic_photo_test")
    poses = rgb.render_photo_set(flat_r1(), out, seed=1, verbose=False)
    return out, poses


def test_photos_shape_and_exif(photo_set):
    out, poses = photo_set
    files = sorted(os.listdir(os.path.join(out, "01_room")))
    assert files == [f"{i:02d}.jpg" for i in range(1, 7)]
    for fn in files:
        img = Image.open(os.path.join(out, "01_room", fn))
        assert img.size == (1512, 2016)
        ifd = img.getexif().get_ifd(0x8769)
        assert ifd[0xA405] == 26
        assert float(ifd[0x920A]) == pytest.approx(6.86, abs=1e-3)
        arr = np.asarray(img)
        assert arr.shape == (2016, 1512, 3) and 40 < arr.mean() < 220
    with open(os.path.join(out, "poses.yaml")) as f:
        y = yaml.safe_load(f)
    assert len(y["rooms"]["01_room"]) == 6 and y["intrinsics"]["f35_mm"] == 26
    e = y["rooms"]["01_room"][0]
    assert abs(e["actual"]["x"] - e["nominal"]["x"]) <= 0.02 + 1e-9
    assert abs(e["actual"]["yaw_deg"] - e["nominal"]["yaw_deg"]) <= 2.0 + 1e-9


def test_photo_plan_protocol_positions():
    flat = flat_mr1()
    plan = rgb.photo_plan(flat, rgb.default_furniture(flat))
    r1 = plan["01_room"]
    assert np.allclose(r1[0]["xy"], (3.15, 0.15)) and math.degrees(r1[0]["yaw"]) == pytest.approx(90.0)
    assert np.allclose(r1[1]["xy"], (0.45, 0.45))        # near-left, 0.45 from D and C
    assert np.allclose(r1[2]["xy"], (0.45, 3.15))        # far-left
    assert np.allclose(r1[3]["xy"], (3.75, 3.15))        # far-right
    assert np.allclose(r1[4]["xy"], (3.3, 0.45)) and "shifted" in r1[4]["shot"]   # pushed off the wardrobe
    assert np.allclose(r1[5]["xy"], (2.7, 1.975)) and math.degrees(r1[5]["yaw"]) == pytest.approx(0.0)
    k = plan["02_kitchen"]                                # entry wall D: facing +x, left = +y (wall A)
    assert np.allclose(k[0]["xy"], (4.35, 1.975)) and math.degrees(k[0]["yaw"]) == pytest.approx(0.0)
    assert np.allclose(k[1]["xy"], (4.65, 2.4))           # near-left pushed off the counter
    assert np.allclose(k[4]["xy"], (4.65, 0.75))          # near-right
    b = plan["04_balcony"]
    assert np.allclose(b[5]["xy"], (3.45, 4.5))           # 1.5 m clamped to depth - 0.3


def _project(K, T, pw):
    pc = T[:3, :3].T @ (pw - T[:3, 3])
    return K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]


def test_damage_visible_and_located():
    flat = flat_r1()
    room = flat.rooms[0]
    dmg = {d["class"]: d for d in rgb.damage_ground_truth(flat)}
    assert dmg["water_stain"]["surface"] == "ceiling" and dmg["crack"]["surface"] == "wall:A"
    assert 0.06 < dmg["water_stain"]["area_m2"] < 0.16
    assert 0.85 < dmg["crack"]["length_m"] < 1.0
    W, H = 756, 1008
    K = rgb.photo_K(W, H)
    rng = np.random.default_rng(0)
    on, off = rgb.build_scene(flat, damage=True), rgb.build_scene(flat, damage=False)
    cu, cv_ = dmg["water_stain"]["centre_uv_m"]
    p_stain = np.array([room.origin[0] + cu, room.origin[1] + cv_, room.height])
    T = syn.pose_from_euler(p_stain[0], p_stain[1] - 1.2, 1.45, math.pi / 2, math.radians(47), 0.0)
    cu, cv_ = dmg["crack"]["centre_uv_m"]
    p_crack = np.array([room.origin[0] + cu, room.origin[1] + room.size[1], cv_])
    T2 = syn.pose_from_euler(p_crack[0], p_crack[1] - 1.2, 1.45, math.pi / 2, math.radians(18), 0.0)
    for pose, pw, min_frac in ((T, p_stain, 0.003), (T2, p_crack, 0.0003)):
        a = rgb.render_view(on, K, W, H, pose, rng, noise_sigma=0, vignette=0).astype(float)
        b = rgb.render_view(off, K, W, H, pose, rng, noise_sigma=0, vignette=0).astype(float)
        diff = np.abs(a - b).max(axis=2) / 255
        u, v = _project(K, pose, pw)
        assert 0 < u < W and 0 < v < H
        win = diff[max(0, int(v) - 60):int(v) + 60, max(0, int(u) - 60):int(u) + 60]
        assert win.max() > 0.15, (pw, win.max())
        frac = (diff > 0.05).mean()
        assert min_frac < frac < 0.2, frac
        c = rgb.render_view(off, K, W, H, pose, rng, noise_sigma=0, vignette=0).astype(float)
        assert np.abs(b - c).max() == 0


def _dark_runs(path, thresh=10.0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    means = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        means.append(float(cv2.resize(fr, (64, 36)).mean()))
    cap.release()
    runs, start = [], None
    for i, m in enumerate(means + [255.0]):
        if m < thresh and start is None:
            start = i
        elif m >= thresh and start is not None:
            runs.append((i - start) / fps)
            start = None
    return runs, len(means), fps


@pytest.fixture(scope="module")
def mini_video(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("vid") / "synthetic_video_mini")
    meta = rgb.render_video(flat_mr1(), out, seed=2, fps=4.0, walk_speed=1.0, render_scale=1.0,
                            out_size=(135, 240), verbose=False)
    return out, meta


def test_video_visits_and_segments(mini_video):
    out, meta = mini_video
    assert rgb.video_visits(flat_mr1()) == [("01_room", "loop"), ("02_kitchen", "loop"), ("03_washroom", "loop"),
                                           ("02_kitchen", "transit"), ("01_room", "transit"),
                                           ("04_balcony", "loop"), ("01_room", "final")]
    with open(os.path.join(out, "rooms.txt")) as f:
        assert f.read().split() == ["01_room", "02_kitchen", "03_washroom", "02_kitchen", "01_room",
                                    "04_balcony", "01_room"]
    assert len(meta["covers"]) == 6 and all(abs(c["frames"] / meta["fps"] - 2.0) <= 0.3 for c in meta["covers"])
    assert all(s["frames"] / meta["fps"] >= rgb.MIN_SEGMENT_S - 0.3 for s in meta["segments"])
    runs, n, fps = _dark_runs(os.path.join(out, "walk.mp4"))
    assert n == meta["n_frames"] and fps == 4.0
    assert sum(r >= 1.5 for r in runs) >= 6
    rooms = load_video_capture(out)
    assert len(rooms) == 7, list(rooms)


def test_video_on_disk_mr1():
    out = os.path.join(REPO, "benchmark", "captures", "synthetic_video_MR1")
    if not os.path.exists(os.path.join(out, "walk.mp4")):
        pytest.skip("run scripts/make_synthetic_rgb.py first")
    runs, n, fps = _dark_runs(os.path.join(out, "walk.mp4"))
    assert fps == 10.0
    assert sum(r >= 1.5 for r in runs) >= 6
    cap = cv2.VideoCapture(os.path.join(out, "walk.mp4"))
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (1080, 1920)
    rooms = load_video_capture(out)
    assert len(rooms) == 7, list(rooms)
