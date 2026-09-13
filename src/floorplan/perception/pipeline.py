"""Run the damage detector over a capture's RGB and project each mask onto a plan surface.

LiDAR: every pixel has depth and a pose, so a mask becomes a world patch, assigned to the room whose
footprint contains it and the surface it lies closest to. Photo and video: the same thing in the
frame's own gravity-aligned coordinates against the room's fused rectangle. Either way the extent is
measured on the surface, in metres.
"""
from __future__ import annotations

import numpy as np

from floorplan.core.types import Measurement
from floorplan.geometry.frame import backproject

_DET = None
MIN_MASK_PTS = 60
MAX_FRAMES = 16
# An open-vocabulary detector asked for "a crack in the wall" will find something in every image of a
# wall. On real interiors the useful operating point is well above the detector's default: below
# about 0.3 an undamaged office produces dozens of regions. Raising it trades recall for a report a
# loss adjuster would not throw away. The threshold and the minimum extent are both here, together,
# because they are one decision.
SCORE_MIN = 0.30
MIN_AREA_M2 = 0.02


def _detector(device):
    global _DET
    if _DET is None:
        from floorplan.perception.detect import DamageDetector, weights_available
        if not weights_available():
            raise RuntimeError("OWLv2 weights are not present: run scripts/fetch_weights.sh")
        _DET = DamageDetector(device=device, score_threshold=SCORE_MIN)
    return _DET


def dedupe(regions: list[dict]) -> list[dict]:
    """One physical patch seen from several frames is one region.

    Regions of the same class on the same surface whose bounding boxes overlap are merged, keeping
    the union extent, the best class confidence, and every contributing frame as evidence.
    """
    out: list[dict] = []
    for r in sorted(regions, key=lambda x: -x["class_confidence"]):
        bb = r["extent"]["bbox_uv_m"]
        hit = None
        for q in out:
            if q["surface_id"] != r["surface_id"] or q["class"] != r["class"]:
                continue
            qb = q["extent"]["bbox_uv_m"]
            if bb[0] <= qb[2] and qb[0] <= bb[2] and bb[1] <= qb[3] and qb[1] <= bb[3]:
                hit = q
                break
        if hit is None:
            out.append(r)
            continue
        qb = hit["extent"]["bbox_uv_m"]
        hit["extent"]["bbox_uv_m"] = [min(qb[0], bb[0]), min(qb[1], bb[1]), max(qb[2], bb[2]), max(qb[3], bb[3])]
        hit["extent"]["area_m2"]["value"] = round(max(hit["extent"]["area_m2"]["value"],
                                                      r["extent"]["area_m2"]["value"]), 4)
        for f in r["evidence_frames"]:
            if f not in hit["evidence_frames"]:
                hit["evidence_frames"].append(f)
    for i, r in enumerate(out):
        rid = r["surface_id"].split(":")[0]
        r["id"] = f"{rid}:dmg:{i}"
        r["evidence_frames"] = r["evidence_frames"][:8]
        r["n_views"] = len(r["evidence_frames"])
    return out


def _measurement(area: float, n: int) -> Measurement:
    rel = 0.15 + 40.0 / max(n, 1)
    return Measurement(float(area), float(area * rel + 0.01), "measured",
                       "open-vocabulary detection, mask refined, projected onto the surface plane", int(n))


def _surface_for(points: np.ndarray, room, floor_z: float, ceil_z: float):
    """Which surface of `room` these world points sit on: a wall edge, the floor, or the ceiling."""
    z = float(np.median(points[:, 2]))
    ring = np.asarray(room.polygon, float)
    n = len(ring)
    if z < floor_z + 0.25:
        return f"{room.key}:floor", ("floor", None)
    if np.isfinite(ceil_z) and z > ceil_z - 0.35:
        return f"{room.key}:ceiling", ("ceiling", None)
    c = points[:, :2].mean(0)
    best, bk = None, -1
    for k in range(n):
        a, b = ring[k], ring[(k + 1) % n]
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        t = float(np.clip(np.dot(c - a, d) / (L * L), 0.0, 1.0))
        dist = float(np.hypot(*(c - (a + t * d))))
        if best is None or dist < best:
            best, bk = dist, k
    if bk < 0 or best > 1.2:
        return None, (None, None)
    return f"{room.key}:wall:{room.wall_ids[bk]}", ("wall", bk)


def _uv(points: np.ndarray, room, kind: str, k, floor_z: float):
    """Surface coordinates: for a wall, u along the wall from its start and v up from the floor;
    for floor and ceiling, u and v are the room's own x and y offsets from its first corner."""
    ring = np.asarray(room.polygon, float)
    if kind == "wall":
        a, b = ring[k], ring[(k + 1) % len(ring)]
        d = b - a
        L = float(np.hypot(*d))
        u = (points[:, :2] - a) @ (d / max(L, 1e-9))
        v = points[:, 2] - floor_z
    else:
        o = ring.min(0)
        u = points[:, 0] - o[0]
        v = points[:, 1] - o[1]
    return np.stack([u, v], 1)


def _region(uv: np.ndarray, surface_id: str, cls: str, conf: float, frame_path: str, rid: str, j: int):
    from shapely.geometry import MultiPoint
    hull = MultiPoint([tuple(p) for p in uv]).convex_hull
    area = float(getattr(hull, "area", 0.0))
    if area <= 1e-4:
        u0, v0 = uv.min(0)
        u1, v1 = uv.max(0)
        area = max(float((u1 - u0) * (v1 - v0)), 1e-4)
        poly = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    else:
        poly = list(hull.exterior.coords)[:-1]
    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    m = _measurement(area, len(uv))
    return {"id": f"{rid}:dmg:{j}", "surface_id": surface_id, "class": cls,
            "class_confidence": round(float(conf), 3),
            "extent": {"area_m2": {"value": round(m.value, 4),
                                   "ci_low": round(max(0.0, m.value - 1.645 * m.sigma), 4),
                                   "ci_high": round(m.value + 1.645 * m.sigma, 4),
                                   "sigma": round(m.sigma, 4), "source": m.source, "method": m.method,
                                   "n_support": m.n_support},
                       "bbox_uv_m": [round(float(u0), 3), round(float(v0), 3), round(float(u1), 3), round(float(v1), 3)]},
            "polygon_uv_m": [[round(float(a), 3), round(float(b), 3)] for a, b in poly][:64],
            "evidence_frames": [frame_path]}


def _lidar_damage(rooms, aux, device):
    from shapely.geometry import Point, Polygon
    det = _detector(device)
    cloud = aux["cloud"]
    frames = aux.get("frames") or []
    frames = [f for f in frames if f.rgb is not None and f.rgb.size > 16]
    if not frames:
        return []
    step = max(1, len(frames) // MAX_FRAMES)
    polys = [(r, Polygon(np.asarray(r.polygon, float))) for r in rooms if len(r.polygon) >= 3]
    out, counter = [], {}
    for f in frames[::step]:
        dets = det.detect(f.rgb)
        if not dets:
            continue
        P = backproject(f.depth, f.K)
        W = P @ f.pose[:3, :3].T + f.pose[:3, 3]
        R = cloud.tilt
        from floorplan.geometry.frame import rotz
        W = (W @ R.T) @ rotz(-cloud.yaw).T
        hd, wd = f.depth.shape[:2]
        hr, wr = f.rgb.shape[:2]
        for d in dets:
            m = d["mask"]
            ys = (np.arange(hd) * hr / hd).astype(int).clip(0, hr - 1)
            xs = (np.arange(wd) * wr / wd).astype(int).clip(0, wr - 1)
            md = m[ys][:, xs]
            pts = W[md & np.isfinite(W).all(-1) & (f.depth > 0.2) & (f.depth < 5.0)]
            if len(pts) < MIN_MASK_PTS:
                continue
            c = pts[:, :2].mean(0)
            room = None
            for r, pg in polys:
                if pg.buffer(0.4).contains(Point(c)):
                    room = r
                    break
            if room is None:
                continue
            sid, (kind, k) = _surface_for(pts, room, cloud.floor_z, cloud.ceil_z)
            if sid is None:
                continue
            uv = _uv(pts, room, kind, k, cloud.floor_z)
            j = counter.get(room.key, 0)
            counter[room.key] = j + 1
            out.append(_region(uv, sid, d["cls"], d["score"], f.path, room.key, j))
    return out


def _mono_damage(rooms, device):
    det = _detector(device)
    out = []
    for room in rooms:
        frames, geoms = getattr(room, "frames", []), getattr(room, "geoms", [])
        np.asarray(room.polygon, float)
        j = 0
        for f, g in zip(frames, geoms, strict=False):
            if g is None or g.P is None:
                continue
            dets = det.detect(f.rgb)
            if not dets:
                continue
            hd, wd = g.P.shape[:2]
            hr, wr = f.rgb.shape[:2]
            for d in dets:
                m = d["mask"]
                ys = (np.arange(hd) * hr / hd).astype(int).clip(0, hr - 1)
                xs = (np.arange(wd) * wr / wd).astype(int).clip(0, wr - 1)
                md = m[ys][:, xs]
                pts = g.P[md & np.isfinite(g.P).all(-1)]
                if len(pts) < MIN_MASK_PTS:
                    continue
                ch = g.cam_height if np.isfinite(g.cam_height) else 1.4
                z = pts[:, 2] + ch
                if float(np.median(z)) < 0.25:
                    sid, kind, k = f"{room.key}:floor", "floor", None
                elif room.ceiling and float(np.median(z)) > room.ceiling.value - 0.35:
                    sid, kind, k = f"{room.key}:ceiling", "ceiling", None
                else:
                    # nearest wall of the rectangle, in the frame's own coordinates
                    k = int(np.argmax([abs(np.median(pts @ np.array(v))) for v in
                                       ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))]))
                    k = min(k, len(room.wall_ids) - 1)
                    sid, kind = f"{room.key}:wall:{room.wall_ids[k]}", "wall"
                u = pts[:, 0] if kind != "wall" else pts[:, 1]
                v = z if kind == "wall" else pts[:, 1]
                uv = np.stack([u - u.min(), v], 1)
                out.append(_region(uv, sid, d["cls"], d["score"], f.path, room.key, j))
                j += 1
    return out


def damage_for_rooms(rooms, tier: str, device: str = "auto", aux: dict | None = None) -> list[dict]:
    regions = (_lidar_damage(rooms, aux, device) if tier == "lidar" and aux and aux.get("cloud") is not None
               else _mono_damage(rooms, device))
    regions = [r for r in regions if r["extent"]["area_m2"]["value"] >= MIN_AREA_M2]
    return dedupe(regions)
