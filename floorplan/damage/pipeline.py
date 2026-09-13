"""Glue: run the damage detector on each room's frames, project masks onto that room's surfaces."""
from __future__ import annotations

import numpy as np

from floorplan.core.types import Plane
from floorplan.damage.project import merge_regions, regions_from_frame
from floorplan.geometry.room import CORNER_POS, LETTER, STAND_OFF, _cam_position

_DET = None


def _detector(device):
    global _DET
    if _DET is None:
        from floorplan.damage.detect import DamageDetector, weights_available
        if not weights_available():
            raise RuntimeError("OWLv2 weights not present: run scripts/fetch_weights.sh")
        _DET = DamageDetector(device=device)
    return _DET


def _frame_planes(room, fr, g, ci):
    """Planes (with wall-start origins) and a per-pixel plane id map for one oriented frame."""
    Lx, Ly = room.length_x.value, room.length_y.value
    cx, cy, _, _ = _cam_position(g, fr, ci, Lx, Ly)
    if not np.isfinite(cx):
        cx = 0.5 * Lx
    if not np.isfinite(cy):
        cy = 0.5 * Ly
    ch = g.cam_height if np.isfinite(g.cam_height) else 1.4
    planes, sids = [], []
    corner = {"A": (0, Ly), "B": (Lx, Ly), "C": (Lx, 0), "D": (0, 0)}
    for d, w in g.walls.items():
        letter = LETTER[d]
        axis = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}[d]
        n = -np.array(axis, float)                  # normal toward the camera (into the room)
        dist = w["dist"]
        p = Plane(normal=n, d=float(dist), kind="wall", n_inliers=w["n"], rms=w["rms"])  # n·x + d = 0 at x·axis = dist
        p.origin = np.array([corner[letter][0] - cx, corner[letter][1] - cy, -ch])
        planes.append(p); sids.append(f"{room.key}:wall:{letter}")
    if np.isfinite(g.cam_height):
        p = Plane(normal=np.array([0, 0, 1.0]), d=float(g.cam_height), kind="floor")
        p.origin = np.array([-cx, -cy, -ch]); planes.append(p); sids.append(f"{room.key}:floor")
    if np.isfinite(g.ceil_above):
        p = Plane(normal=np.array([0, 0, -1.0]), d=float(g.ceil_above), kind="ceiling")
        p.origin = np.array([-cx, -cy, g.ceil_above]); planes.append(p); sids.append(f"{room.key}:ceiling")
    P = g.P
    pid = np.full(P.shape[:2], -1, np.int32)
    best = np.full(P.shape[:2], 0.12, np.float32)
    for i, p in enumerate(planes):
        dist = np.abs(P @ p.normal + p.d)
        m = np.isfinite(dist) & (dist < best)
        pid[m] = i; best[m] = dist[m]
    return planes, sids, pid


def damage_for_rooms(rooms, tier: str, device: str = "auto", aux: dict | None = None) -> list[dict]:
    det = _detector(device)
    all_regions = []
    for room in rooms:
        frames, geoms = getattr(room, "_frames", []), getattr(room, "_geoms", [])
        regs = []
        ci = 0
        for fr, g in zip(frames, geoms):
            if g is None or g.P is None:
                continue
            cidx = None
            if fr.role == "corner":
                cidx = ci; ci += 1
            dets = det.detect(fr.rgb)
            if not dets:
                continue
            planes, sids, pid = _frame_planes(room, fr, g, cidx)
            # masks are at rgb resolution; geometry grid may be strided
            sy, sx = fr.rgb.shape[0] / pid.shape[0], fr.rgb.shape[1] / pid.shape[1]
            for dd in dets:
                m = dd["mask"]
                if sy != 1 or sx != 1:
                    ys = (np.arange(pid.shape[0]) * sy).astype(int).clip(0, m.shape[0] - 1)
                    xs = (np.arange(pid.shape[1]) * sx).astype(int).clip(0, m.shape[1] - 1)
                    dd["mask"] = m[ys][:, xs]
            regs += regions_from_frame(dets, g.P.astype(np.float64), pid, planes, sids, fr.path)
        regs = merge_regions(regs)
        for j, r in enumerate(regs):
            all_regions.append({
                "id": f"{room.key}:dmg:{j}", "surface_id": r.surface_id, "class": r.cls,
                "class_confidence": round(float(r.cls_conf), 3),
                "extent": {"area_m2": _mj(r.area_m2), "bbox_uv_m": [round(float(x), 3) for x in r.bbox_uv_m]},
                "polygon_uv_m": [[round(float(a), 3), round(float(b), 3)] for a, b in r.polygon_uv_m],
                "evidence_frames": [str(x) for x in r.evidence_frames][:8],
            })
    return all_regions


def _mj(m):
    return {"value": round(float(m.value), 4), "ci_low": round(float(max(0.0, m.value - 1.645 * m.sigma)), 4),
            "ci_high": round(float(m.value + 1.645 * m.sigma), 4), "sigma": round(float(m.sigma), 4),
            "source": m.source, "method": m.method, "n_support": int(m.n_support)}
