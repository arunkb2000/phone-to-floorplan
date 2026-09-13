"""End-to-end checks that do not need the supplied captures or any model weights."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from floorplan.core.room import RoomOut
from floorplan.core.types import Measurement, Opening
from floorplan.geometry.assemble import build_plan, validate
from floorplan.geometry.cloud import _reconstruct, hmaxima_seeds, merge_unwalled
from floorplan.geometry.layout import close_polygon, rectilinear
from floorplan.ingest.stray import stray_pose_to_zup


def test_pose_convention_maps_arkit_up_to_our_z():
    """ARKit's world is y-up; ours is z-up. A phone 1.4 m off the ground must land at z = 1.4."""
    T = stray_pose_to_zup(0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert np.allclose(T[:3, 3], [0.0, 0.0, 1.4]), T[:3, 3]
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0), "the mapping must stay right-handed"
    # the exported rotation is used as-is (not flipped from OpenGL), so the camera's forward axis
    # lands on ARKit's +z, which is our -y. scripts/check_convention.py is the evidence for this.
    assert np.allclose(T[:3, :3] @ np.array([0, 0, 1.0]), [0, -1, 0], atol=1e-9)
    assert np.allclose(T[:3, :3] @ np.array([0, 1.0, 0]), [0, 0, 1], atol=1e-9)


def test_room_seeding_is_stable_under_small_perturbations():
    """The property the threshold cascade lacked, as a test.

    Two captures of the same rooms differ by a few cells of carved free space. That must not change
    how many rooms exist. All three parts of the fix are exercised, because they are one fix: fill
    enclosed unobserved cells, seed by h-maxima rather than an absolute threshold, then merge
    regions no wall separates. Take away the hole filling and random nibbles alone push the seed
    count to four and six.
    """
    from scipy import ndimage
    res = 0.05
    base = np.zeros((160, 80), bool)
    base[10:70, 10:70] = True         # room one, 3 m x 3 m
    base[90:150, 10:70] = True        # room two
    base[70:90, 36:44] = True         # a 0.4 m doorway neck between them
    walls = ~ndimage.binary_dilation(base, np.ones((3, 3)))
    counts = set()
    rng = np.random.default_rng(0)
    for trial in range(6):
        m = base.copy()
        for _ in range(trial * 25):   # nibble cells, as a differently-walked capture would
            i, j = int(rng.integers(0, 160)), int(rng.integers(0, 80))
            if m[i, j] and m[max(0, i - 1):i + 2, max(0, j - 1):j + 2].sum() > 4:
                m[i, j] = False
        m = ndimage.binary_fill_holes(m)
        D = ndimage.distance_transform_edt(m) * res
        from floorplan.geometry.cloud import _absorb_slivers, _priority_flood
        seeds = hmaxima_seeds(D, 0.35, res)          # the shipped default
        lab = _priority_flood(seeds.astype(np.int32), D, m)
        lab = merge_unwalled(lab, walls, res)
        lab = _absorb_slivers(lab, walls, res, 1.5)
        counts.add(len({int(v) for v in np.unique(lab) if v > 0}))
    assert counts == {2}, f"room count changed across perturbations: {counts}"


def test_reconstruction_is_idempotent():
    D = np.random.default_rng(0).random((30, 30))
    r1 = _reconstruct(D - 0.2, D)
    assert np.all(r1 <= D + 1e-9)
    assert np.array_equal(_reconstruct(r1, D), r1)


def test_merge_unwalled_joins_regions_with_no_wall_between_them():
    lab = np.zeros((40, 60), np.int32)     # the shared boundary is 3 m wide, well over max_open_m
    lab[:20] = 1
    lab[20:] = 2
    walls = np.zeros((40, 60), bool)
    assert merge_unwalled(lab.copy(), walls, 0.05).max() == 1, "no wall between them: must merge"
    walls[19:21, :] = True
    assert merge_unwalled(lab.copy(), walls, 0.05).max() == 2, "a full wall between them: must not merge"


def test_rectilinear_absorbs_grid_stubs():
    """A 5 cm jog in the contour is a grid artefact, not a wall."""
    poly = np.array([[0, 0], [4, 0], [4, 1.99], [4.05, 1.99], [4.05, 3], [0, 3]], float)
    edges = rectilinear(poly, min_edge=0.35)
    assert len(edges) == 4, [(e.axis, round(e.coord, 3), round(e.length, 3)) for e in edges]
    ring = close_polygon(edges)
    area = 0.5 * abs(np.dot(ring[:, 0], np.roll(ring[:, 1], -1)) - np.dot(ring[:, 1], np.roll(ring[:, 0], -1)))
    assert abs(area - 12.0) < 0.3, area


def _room(key="01_space", lx=4.0, ly=3.0):
    r = RoomOut(key=key, label="room")
    r.polygon = np.array([[0, 0], [lx, 0], [lx, ly], [0, ly]], float)
    r.wall_ids = ["C", "B", "A", "D"]
    r.wall_len = [Measurement(lx, 0.01, "measured", "m", 100), Measurement(ly, 0.01, "measured", "m", 100),
                  Measurement(lx, 0.01, "measured", "m", 100), Measurement(ly, 0.01, "measured", "m", 100)]
    r.ceiling = Measurement(2.7, 0.005, "measured", "m", 1000)
    r.area = Measurement(lx * ly, 0.05, "measured", "m", 4)
    r.openings = [(0, Opening(wall_id="C", type="door",
                              offset=Measurement(1.0, 0.02, "measured", "m", 50),
                              width=Measurement(0.9, 0.01, "measured", "m", 50), confidence=0.8))]
    return r


def test_plan_validates_against_the_published_schema():
    plan = build_plan([_room()], [], "lidar",
                      {"id": "t", "source": "stray_scanner", "input_sha256": "0" * 16},
                      {"total_s": 1.0, "device": "test"}, None, {"svg": "plan.svg"})
    assert validate(plan) == []


def test_every_measurement_carries_an_interval_that_brackets_its_value():
    plan = build_plan([_room()], [], "photo",
                      {"id": "t", "source": "ios_camera_photos", "input_sha256": "0" * 16},
                      {"total_s": 1.0, "device": "test"}, None, {"svg": "plan.svg"})
    seen = 0

    def walk(node):
        nonlocal seen
        if isinstance(node, dict):
            if {"value", "ci_low", "ci_high"} <= set(node):
                assert node["ci_low"] <= node["value"] <= node["ci_high"], node
                seen += 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(plan)
    assert seen >= 10, f"only {seen} measurements found"


def test_photo_tier_intervals_are_wider_than_lidar_for_the_same_geometry():
    """Thin input must not claim thick-input precision, whatever the propagated sigma says."""
    cap = {"id": "t", "source": "ios_camera_photos", "input_sha256": "0" * 16}
    tm = {"total_s": 1.0, "device": "test"}
    a = build_plan([_room()], [], "lidar", cap, tm, None, {"svg": "s"})
    b = build_plan([_room()], [], "photo", cap, tm, None, {"svg": "s"})
    wa = a["rooms"][0]["walls"][0]["length"]
    wb = b["rooms"][0]["walls"][0]["length"]
    assert (wb["ci_high"] - wb["ci_low"]) > 3 * (wa["ci_high"] - wa["ci_low"])


@pytest.mark.skipif(not os.path.isdir("data/raw/single_room"), reason="supplied captures not present")
def test_lidar_tier_runs_end_to_end_and_is_deterministic():
    from floorplan.cli.main import run
    out = "/tmp/floorplan_test_run"
    run("data/raw/single_room", out=out, damage=False)
    p1 = json.load(open(os.path.join(out, "plan.json")))
    assert p1["rooms"] and validate(p1) == []
    assert all(r["ceiling_height"]["ci_low"] <= r["ceiling_height"]["value"] for r in p1["rooms"])
    run("data/raw/single_room", out=out + "2", damage=False)
    p2 = json.load(open(os.path.join(out + "2", "plan.json")))
    strip = lambda p: json.dumps({k: v for k, v in p.items() if k != "timing"}, sort_keys=True)
    assert strip(p1) == strip(p2), "same input, different output: the pipeline is not deterministic"
