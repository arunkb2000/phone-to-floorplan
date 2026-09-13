"""Whole-property stitch. Rooms are rectangles in their own frame (x∈[0,Lx], y∈[0,Ly], entrance on
wall C at y=0). Without poses (photo, video) rooms are chained in walk order through shared doors:
room i's exit door coincides with room i+1's entrance door and the rooms lie on opposite sides of
that wall. Overlaps are penalised and the least-overlapping exit wall wins when no exit door was
detected. With poses (LiDAR) placement is absolute and this only builds adjacency."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from floorplan.core.types import Opening, RoomEstimate

WALL_DIR = {"A": (0, 1), "B": (1, 0), "C": (0, -1), "D": (-1, 0)}   # outward normals in room frame


def wall_segment(room: RoomEstimate, letter: str):
    """Start/end of a wall in the room's local frame following the clockwise offset convention."""
    Lx, Ly = room.length_x.value, room.length_y.value
    return {"A": ((0, Ly), (Lx, Ly)), "B": ((Lx, Ly), (Lx, 0)), "C": ((Lx, 0), (0, 0)), "D": ((0, 0), (0, Ly))}[letter]


def opening_local_center(room: RoomEstimate, op: Opening):
    (x0, y0), (x1, y1) = wall_segment(room, op.wall_id)
    L = float(np.hypot(x1 - x0, y1 - y0))
    t = (op.offset.value + 0.5 * op.width.value) / max(L, 1e-6)
    t = min(max(t, 0.0), 1.0)
    return np.array([x0 + t * (x1 - x0), y0 + t * (y1 - y0)])


def transform(pt, k, tx, ty):
    c, s = [1, 0, -1, 0][k % 4], [0, 1, 0, -1][k % 4]
    x, y = pt
    return np.array([c * x - s * y + tx, s * x + c * y + ty])


def room_polygon(room: RoomEstimate):
    k, tx, ty = room.placement.get("rot_k", 0), room.placement.get("x", 0.0), room.placement.get("y", 0.0)
    Lx, Ly = room.length_x.value, room.length_y.value
    return [transform(p, k, tx, ty) for p in ((0, 0), (Lx, 0), (Lx, Ly), (0, Ly))]


def _entrance(room: RoomEstimate):
    doors = [o for o in room.openings if o.wall_id == "C" and o.type in ("door", "passage")]
    if doors:
        return max(doors, key=lambda o: o.confidence)
    return None


def _exit_candidates(room: RoomEstimate):
    ent = _entrance(room)
    cands = [o for o in room.openings if o.type in ("door", "passage") and o is not ent]
    cands.sort(key=lambda o: -o.confidence)
    return cands


def stitch(rooms: list[RoomEstimate], absolute: bool = False) -> list[dict]:
    """Sets room.placement in place. Returns adjacency edges."""
    edges = []
    if absolute:
        for r in rooms:
            r.placement = {"rot_k": 0, "x": r.origin_xy[0], "y": r.origin_xy[1], "confidence": 0.9}
        # adjacency: rooms whose rectangles touch along a wall that has a door
        polys = [Polygon(room_polygon(r)) for r in rooms]
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if polys[i].buffer(0.25).intersects(polys[j]):
                    door = next((o for o in rooms[i].openings if o.type in ("door", "passage")), None)
                    edges.append({"room_a": rooms[i].key, "room_b": rooms[j].key,
                                  "via_opening_id": f"{rooms[i].key}:{door.wall_id}:{door.type}" if door else "",
                                  "evidence": "trajectory", "confidence": 0.85 if door else 0.6})
        return edges
    if not rooms:
        return edges
    rooms[0].placement = {"rot_k": 0, "x": 0.0, "y": 0.0, "confidence": 1.0}
    placed = [Polygon(room_polygon(rooms[0]))]
    for i in range(1, len(rooms)):
        prev, cur = rooms[i - 1], rooms[i]
        ent = _entrance(cur)
        # entrance door centre in cur's local frame, or the middle of wall C
        if ent is not None:
            ec = opening_local_center(cur, ent)
        else:
            ec = np.array([0.5 * cur.length_x.value, 0.0])
        cands = _exit_candidates(prev)
        options = []
        if cands:
            for o in cands:
                options.append((o.wall_id, opening_local_center(prev, o), o))
        for w in ("A", "B", "D", "C"):
            if not any(opt[0] == w for opt in options):
                (x0, y0), (x1, y1) = wall_segment(prev, w)
                options.append((w, np.array([(x0 + x1) / 2, (y0 + y1) / 2]), None))
        best = None
        for wall, pc_local, op in options:
            kp, txp, typ = prev.placement["rot_k"], prev.placement["x"], prev.placement["y"]
            pc = transform(pc_local, kp, txp, typ)
            nx, ny = WALL_DIR[wall]
            n_glob = transform((nx, ny), kp, 0, 0)               # outward normal of prev's exit wall
            # cur's entrance wall C has outward normal (0,-1); rotate cur so that becomes -n_glob
            target = -n_glob
            k = next(kk for kk in range(4) if np.allclose(transform((0, -1), kk, 0, 0), target))
            ec_rot = transform(ec, k, 0, 0)
            tx, ty = pc - ec_rot
            cur.placement = {"rot_k": k, "x": float(tx), "y": float(ty)}
            poly = Polygon(room_polygon(cur))
            overlap = sum(poly.intersection(p).area for p in placed) / max(poly.area, 1e-6)
            score = overlap * 10 + (0 if op is not None else 1.0) + (0 if op is None else (1 - op.confidence) * 0.5)
            if best is None or score < best[0]:
                best = (score, k, float(tx), float(ty), wall, op, overlap)
        score, k, tx, ty, wall, op, overlap = best
        cur.placement = {"rot_k": k, "x": tx, "y": ty,
                         "confidence": float(max(0.2, (0.9 if op is not None else 0.45) - 2 * overlap))}
        prev.exit_wall = wall
        placed.append(Polygon(room_polygon(cur)))
        edges.append({"room_a": prev.key, "room_b": cur.key,
                      "via_opening_id": f"{prev.key}:{wall}:door",
                      "evidence": "protocol_order" if op is None else "visual_match",
                      "confidence": cur.placement["confidence"]})
        if overlap > 0.05:
            cur.warnings.append(f"placement overlaps a previous room by {overlap:.0%}; adjacency uncertain")
    return edges
