"""Adjacency and placement.

With poses (LiDAR, video) the rooms are already in one frame, so stitching is just working out which
rooms are connected and through which opening. Without poses (photo) the rooms arrive one folder at
a time in walk order and have to be laid out; see floorplan/tiers/photo.py.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from floorplan.core.room import RoomOut


def _poly(r: RoomOut) -> Polygon:
    p = Polygon(np.asarray(r.polygon, float))
    return p if p.is_valid else p.buffer(0)


def adjacency_from_placement(rooms: list[RoomOut], touch: float = 0.35) -> list[dict]:
    """Two rooms are adjacent when their footprints touch and at least one of them has an opening
    on the shared stretch of wall."""
    polys = [_poly(r) for r in rooms]
    edges = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if not polys[i].buffer(touch).intersects(polys[j]):
                continue
            shared = polys[i].buffer(touch).intersection(polys[j].buffer(touch))
            if shared.is_empty or shared.area < 0.05:
                continue
            c = np.array(shared.centroid.coords[0])
            best, bid = None, ""
            for ri, r in ((i, rooms[i]), (j, rooms[j])):
                ring = np.asarray(r.polygon, float)
                for k, o in r.openings:
                    if o.type not in ("door", "passage"):
                        continue
                    a, b = ring[k], ring[(k + 1) % len(ring)]
                    d = b - a
                    L = float(np.hypot(*d))
                    if L < 1e-6:
                        continue
                    m = a + d * ((o.offset.value + 0.5 * o.width.value) / L)
                    dist = float(np.hypot(*(m - c)))
                    if best is None or dist < best:
                        best, bid = dist, f"{r.key}:{r.wall_ids[k]}:{o.type}"
            conf = 0.9 if (best is not None and best < 1.6) else 0.55
            edges.append({"room_a": rooms[i].key, "room_b": rooms[j].key, "via_opening_id": bid,
                          "evidence": "shared_wall" if bid else "trajectory",
                          "confidence": round(conf, 2)})
    return edges


def overlap_fraction(rooms: list[RoomOut]) -> float:
    polys = [_poly(r) for r in rooms]
    total = sum(p.area for p in polys)
    ov = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            ov += polys[i].intersection(polys[j]).area
    return float(ov / total) if total else 0.0


def footprint_area(rooms: list[RoomOut]) -> float:
    from shapely.ops import unary_union
    polys = [_poly(r) for r in rooms]
    return float(unary_union(polys).area) if polys else 0.0
