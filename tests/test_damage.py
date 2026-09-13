"""Damage detection (OWLv2) and plane projection tests."""
from __future__ import annotations

import numpy as np
import pytest

from floorplan.core.types import Plane
from floorplan.perception.detect import DamageDetector, box_iou, weights_available
from floorplan.perception.project import merge_regions, regions_from_frame

STAIN_BOX = (150, 120, 330, 300)   # drawing region for the blotch (x0, y0, x1, y1)
CRACK_BOX = (450, 90, 610, 400)    # drawing region for the crack polyline


def tight_boxes(img: np.ndarray) -> dict:
    """Tight pixel bboxes of the drawn damage: brown pixels -> water_stain, dark non-brown -> crack."""
    r, b = img[..., 0].astype(int), img[..., 2].astype(int)
    brown = (r - b) > 30
    dark = (img.mean(-1) < 100) & ~brown
    out = {}
    for cls, m in (("water_stain", brown), ("crack", dark)):
        ys, xs = np.where(m)
        out[cls] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return out


def synthetic_wall(seed: int = 0):
    """640x480 light-gray wall with noise, a brown blotch and a dark jagged crack polyline."""
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), 205, np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    # irregular brown blotch: several overlapping ellipses inside STAIN_BOX
    x0, y0, x1, y1 = STAIN_BOX
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    for _k in range(7):
        ex = int(rng.integers(-25, 26)) + cx
        ey = int(rng.integers(-35, 36)) + cy
        ax = int(rng.integers(40, (x1 - x0) // 2))
        ay = int(rng.integers(40, (y1 - y0) // 2))
        ax = min(ax, ex - x0, x1 - ex)
        ay = min(ay, ey - y0, y1 - ey)
        col = (int(rng.integers(90, 120)), int(rng.integers(60, 80)), int(rng.integers(30, 45)))
        cv2.ellipse(img, (ex, ey), (max(5, ax), max(5, ay)), int(rng.integers(0, 180)), 0, 360, col, -1)
    # jagged crack polyline inside CRACK_BOX
    cx0, cy0, cx1, cy1 = CRACK_BOX
    ys = np.linspace(cy0, cy1, 14).astype(int)
    xs = np.clip(cx0 + 80 + np.cumsum(rng.integers(-22, 23, len(ys))), cx0 + 3, cx1 - 3).astype(int)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, (40, 35, 30), 2, lineType=cv2.LINE_AA)
    return img


def _best_iou(dets, boxes):
    return max([box_iou(d["box"], boxes[d["cls"]]) for d in dets if d["cls"] in boxes] + [0.0])


@pytest.mark.skipif(not weights_available(), reason="OWLv2 weights not downloaded yet")
def test_owlv2_boxes_land_on_drawn_damage():
    img = synthetic_wall()
    boxes = tight_boxes(img)
    det = DamageDetector(score_threshold=0.1)
    dets = det.detect(img)
    assert dets, "no detections at all"
    H, W = img.shape[:2]
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
        assert d["mask"].shape == (H, W) and d["mask"].dtype == bool and d["mask"].any()
    # The box must land where the damage was drawn (guards the square-padding coordinate mapping).
    best = _best_iou(dets, boxes)
    assert best >= 0.3, f"best IoU {best:.2f}; dets={[(d['cls'], round(d['score'], 2), d['box']) for d in dets]}"
    assert np.isfinite(det.last_inference_s)

    # Portrait orientation exercises the other padding direction: rotate image and reference boxes.
    imgT = np.ascontiguousarray(np.rot90(img))          # (x, y) -> (y, W - x)
    boxesT = {c: (b[1], W - b[2], b[3], W - b[0]) for c, b in boxes.items()}
    detsT = det.detect(imgT)
    bestT = _best_iou(detsT, boxesT)
    assert bestT >= 0.3, f"portrait best IoU {bestT:.2f}; dets={[(d['cls'], round(d['score'], 2), d['box']) for d in detsT]}"


def _wall_grid(h=300, w=400, mpp=0.01):
    """Points on the plane x=0: u = y = col*mpp, v = z = (h-1-row)*mpp (v up)."""
    rows, cols = np.mgrid[0:h, 0:w]
    y = cols * mpp
    z = (h - 1 - rows) * mpp
    pts = np.stack([np.zeros_like(y, dtype=np.float64), y, z], axis=-1)
    return pts


def test_regions_from_frame_planar_wall():
    pts = _wall_grid()
    plane_id = np.zeros(pts.shape[:2], int)
    plane = Plane(normal=np.array([1.0, 0.0, 0.0]), d=0.0, kind="wall")
    # mask covering u in [1.0, 1.5), v in [1.0, 1.3): 50 x 30 px at 1 cm/px = 0.15 m2
    mask = np.zeros(pts.shape[:2], bool)
    h = pts.shape[0]
    rows = slice(h - 1 - 129, h - 1 - 99)      # z in [1.0, 1.29]
    cols = slice(100, 150)                     # y in [1.0, 1.49]
    mask[rows, cols] = True
    dets = [{"cls": "water_stain", "score": 0.7, "box": (100, 170, 150, 200), "mask": mask}]
    regs = regions_from_frame(dets, pts, plane_id, [plane], ["R1_wall_A"], "frame_0001.jpg")
    assert len(regs) == 1
    r = regs[0]
    assert r.surface_id == "R1_wall_A" and r.cls == "water_stain"
    assert abs(r.area_m2.value - 0.15) <= 0.2 * 0.15
    assert r.area_m2.source == "measured" and r.area_m2.sigma > 0
    u0, v0, u1, v1 = r.bbox_uv_m
    assert abs(u0 - 1.0) < 0.02 and abs(u1 - 1.5) < 0.02
    assert abs(v0 - 1.0) < 0.02 and abs(v1 - 1.3) < 0.02
    assert r.evidence_frames == ["frame_0001.jpg"]
    assert len(r.polygon_uv_m) >= 4


def test_regions_ignore_unknown_and_tiny():
    pts = _wall_grid()
    pts[:, :200] = np.nan                 # left half unknown
    plane_id = np.zeros(pts.shape[:2], int)
    plane_id[:150] = -1                   # top half unassigned
    plane = Plane(normal=np.array([1.0, 0.0, 0.0]), d=0.0, kind="wall")
    tiny = np.zeros(pts.shape[:2], bool)
    tiny[200:204, 250:254] = True         # 16 px < 30
    inleft = np.zeros(pts.shape[:2], bool)
    inleft[200:260, 20:80] = True         # NaN points only
    ok = np.zeros(pts.shape[:2], bool)
    ok[100:260, 250:300] = True           # straddles the -1 rows; only rows >= 150 count
    dets = [{"cls": "crack", "score": 0.5, "box": (0, 0, 1, 1), "mask": m} for m in (tiny, inleft, ok)]
    regs = regions_from_frame(dets, pts, plane_id, [plane], ["S"], "f")
    assert len(regs) == 1 and regs[0].cls == "crack"
    assert regs[0].area_m2.n_support == 110 * 50


def test_merge_regions_unions_overlapping_same_class():
    pts = _wall_grid()
    plane_id = np.zeros(pts.shape[:2], int)
    plane = Plane(normal=np.array([1.0, 0.0, 0.0]), d=0.0, kind="wall")
    m1 = np.zeros(pts.shape[:2], bool)
    m1[100:160, 100:160] = True
    m2 = np.zeros(pts.shape[:2], bool)
    m2[130:190, 130:190] = True
    m3 = np.zeros(pts.shape[:2], bool)
    m3[100:160, 300:360] = True
    dets = [{"cls": "mold", "score": s, "box": (0, 0, 1, 1), "mask": m} for s, m in ((0.4, m1), (0.6, m2), (0.5, m3))]
    regs = regions_from_frame(dets, pts, plane_id, [plane], ["S"], "f")
    assert len(regs) == 2
    big = max(regs, key=lambda r: r.area_m2.value)
    assert big.cls_conf == 0.6
    assert 0.6 * 0.6 < big.area_m2.value < 2 * 0.6 * 0.6  # union, not sum of two squares only
    assert merge_regions(regs) == regs or len(merge_regions(regs)) == 2


def test_region_to_json_validates_against_schema():
    import json
    from pathlib import Path

    import jsonschema

    from floorplan.perception.project import region_to_json

    schema = json.loads((Path(__file__).resolve().parents[1] / "src/floorplan/schema/output.schema.json").read_text())
    pts = _wall_grid()
    plane_id = np.zeros(pts.shape[:2], int)
    plane = Plane(normal=np.array([1.0, 0.0, 0.0]), d=0.0, kind="wall")
    mask = np.zeros(pts.shape[:2], bool)
    mask[100:160, 100:160] = True
    regs = regions_from_frame([{"cls": "hole", "score": 0.9, "box": (0, 0, 1, 1), "mask": mask}],
                              pts, plane_id, [plane], ["S"], "f.jpg")
    js = region_to_json(regs[0], "D001")
    jsonschema.validate(js, {**schema["$defs"]["damage_region"], "$defs": schema["$defs"]})
    assert js["class"] == "hole" and js["extent"]["area_m2"]["ci_low"] <= js["extent"]["area_m2"]["value"]
    assert len(js["extent"]["bbox_uv_m"]) == 4 and js["evidence_frames"] == ["f.jpg"]


def test_explicit_plane_origin_is_respected():
    """Geometry passes wall origins at the wall start corner on the floor line; uv must be relative to it."""
    pts = _wall_grid()
    plane_id = np.zeros(pts.shape[:2], int)
    plane = Plane(normal=np.array([1.0, 0.0, 0.0]), d=0.0, kind="wall")
    plane.origin = np.array([0.0, 0.4, -1.0])            # u = y - 0.4, v = z + 1.0
    mask = np.zeros(pts.shape[:2], bool)
    h = pts.shape[0]
    mask[h - 1 - 129:h - 1 - 99, 100:150] = True           # y in [1.0, 1.49], z in [1.0, 1.29]
    regs = regions_from_frame([{"cls": "mold", "score": 0.5, "box": (0, 0, 1, 1), "mask": mask}],
                              pts, plane_id, [plane], ["S"], "f", floor_z=5.0)   # floor_z ignored
    u0, v0, u1, v1 = regs[0].bbox_uv_m
    assert abs(u0 - 0.6) < 0.02 and abs(u1 - 1.1) < 0.02
    assert abs(v0 - 2.0) < 0.02 and abs(v1 - 2.3) < 0.02
