"""Rule engine: concealed-damage flags and scope line items from the schema JSON dicts.

Inputs are the JSON shapes in floorplan/schema/output.schema.json (rooms, adjacency, damage_regions).
Outputs validate against the schema's `concealed_flag` and `scope_item` definitions.
"""
from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
WET_WORDS = ("kitchen", "washroom", "bathroom", "toilet", "bath")
WASHROOM_WORDS = ("washroom", "bathroom", "toilet", "bath", "shower", "wc")
EXTERIOR_WORDS = ("balcony", "terrace", "outside", "exterior", "outdoor")
Z = 1.645  # 90% two-sided normal


@cache
def load_rules(path: str | None = None) -> list[dict]:
    with open(path or _HERE / "rules.yaml") as f:
        return yaml.safe_load(f)["rules"]


@cache
def load_catalogue(path: str | None = None) -> dict:
    with open(path or _HERE / "catalogue.yaml") as f:
        return yaml.safe_load(f)["items"]


def _val(m) -> float:
    if isinstance(m, dict):
        return float(m.get("value", 0.0))
    return float(m or 0.0)


def measurement(value: float, method: str, n_support: int = 0, source: str = "inferred") -> dict:
    sigma = 0.1 * abs(value)
    return {
        "value": round(float(value), 4), "ci_low": round(max(0.0, value - Z * sigma), 4),
        "ci_high": round(value + Z * sigma, 4), "sigma": round(sigma, 4), "source": source,
        "method": method, "n_support": int(n_support),
    }


# ----------------------------------------------------------------------------- indexing
class _Index:
    def __init__(self, rooms: list[dict], adjacency: list[dict], damage: list[dict]):
        self.rooms = {r["id"]: r for r in rooms}
        self.surface = {}      # surface_id -> (surface, room)
        self.wall_by_surface = {}  # surface_id -> (wall, room)
        self.opening = {}      # opening_id -> (opening, room)
        for r in rooms:
            for s in r.get("surfaces", []):
                self.surface[s["id"]] = (s, r)
            for w in r.get("walls", []):
                self.wall_by_surface[w["surface_id"]] = (w, r)
            for o in r.get("openings", []):
                self.opening[o["id"]] = (o, r)
        self.adj: dict[str, set] = {rid: set() for rid in self.rooms}
        self.adj_via_opening: dict[str, str] = {}   # opening_id -> other room id (from edges)
        for e in adjacency:
            a, b = e.get("room_a"), e.get("room_b")
            if a is not None and b is not None:
                self.adj.setdefault(a, set()).add(b)
                self.adj.setdefault(b, set()).add(a)
                oid = e.get("via_opening_id")
                if oid and oid in self.opening:
                    _, orr = self.opening[oid]
                    self.adj_via_opening[oid] = b if orr["id"] == a else a
        for r in rooms:
            for w in r.get("walls", []):
                sw = w.get("shared_with_room_id")
                if sw and sw in self.rooms:
                    self.adj.setdefault(r["id"], set()).add(sw)
                    self.adj.setdefault(sw, set()).add(r["id"])
        self.damage_by_surface: dict[str, list[dict]] = {}
        for d in damage:
            self.damage_by_surface.setdefault(d["surface_id"], []).append(d)

    # room typing
    @staticmethod
    def label_types(label: str) -> set:
        lab = (label or "").lower()
        t = set()
        if any(w in lab for w in WET_WORDS):
            t.add("wet")
        if any(w in lab for w in WASHROOM_WORDS):
            t.add("washroom")
        if any(w in lab for w in EXTERIOR_WORDS):
            t.add("exterior")
        return t

    def room_types(self, room_id: str | None) -> set:
        if room_id is None:
            return set()
        room = self.rooms.get(room_id)
        if room is None:  # not a room id: treat the string itself as a label (e.g. "exterior")
            return self.label_types(room_id)
        return self.label_types(room.get("label", "")) | ({"any_room"} if room else set())

    def surface_kind(self, sid: str) -> str:
        s = self.surface.get(sid)
        return s[0].get("kind", "") if s else ""

    def surface_room(self, sid: str) -> dict | None:
        s = self.surface.get(sid)
        if s:
            return s[1]
        w = self.wall_by_surface.get(sid)
        return w[1] if w else None

    def surface_area(self, sid: str) -> float:
        s = self.surface.get(sid)
        return _val(s[0].get("area")) if s else 0.0

    def wall(self, sid: str):
        w = self.wall_by_surface.get(sid)
        return w[0] if w else None

    def wall_openings(self, wall: dict, room: dict) -> list[dict]:
        return [o for o in room.get("openings", []) if o.get("wall_id") == wall.get("id")]

    def other_side(self, wall: dict, room: dict) -> set:
        """Types of whatever is on the other side of ``wall`` (room labels -> types, + any_room)."""
        types = set()
        sw = wall.get("shared_with_room_id")
        if sw:
            types |= self.room_types(sw) | {"any_room"}
        for o in self.wall_openings(wall, room):
            other = o.get("connects_to_room_id") or self.adj_via_opening.get(o.get("id"))
            if other:
                types |= self.room_types(other) | {"any_room"}
            if o.get("type") == "window":
                types.add("exterior")
        return types

    def other_side_rooms(self, wall: dict, room: dict) -> list[dict]:
        ids = set()
        sw = wall.get("shared_with_room_id")
        if sw in self.rooms:
            ids.add(sw)
        for o in self.wall_openings(wall, room):
            other = o.get("connects_to_room_id") or self.adj_via_opening.get(o.get("id"))
            if other in self.rooms:
                ids.add(other)
        return [self.rooms[i] for i in ids]

    def other_side_damage_classes(self, wall: dict, room: dict) -> set:
        classes = set()
        for other in self.other_side_rooms(wall, room):
            cand = [w for w in other.get("walls", []) if w.get("shared_with_room_id") == room["id"]]
            if not cand:  # fall back to geometric coincidence in the shared global frame
                cand = [w for w in other.get("walls", []) if _segments_coincide(w, wall)]
            for w in cand:
                for d in self.damage_by_surface.get(w["surface_id"], []):
                    classes.add(d.get("class"))
        return classes


def _segments_coincide(w1: dict, w2: dict, tol: float = 0.3) -> bool:
    try:
        a0, a1 = [float(x) for x in w1["start"]], [float(x) for x in w1["end"]]
        b0, b1 = [float(x) for x in w2["start"]], [float(x) for x in w2["end"]]
    except (KeyError, TypeError, ValueError):
        return False

    def dist_to_seg(p, s0, s1):
        dx, dy = s1[0] - s0[0], s1[1] - s0[1]
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            return math.hypot(p[0] - s0[0], p[1] - s0[1])
        t = max(0.0, min(1.0, ((p[0] - s0[0]) * dx + (p[1] - s0[1]) * dy) / L2))
        return math.hypot(p[0] - (s0[0] + t * dx), p[1] - (s0[1] + t * dy))

    near = sum(dist_to_seg(p, b0, b1) < tol for p in (a0, a1)) + sum(dist_to_seg(p, a0, a1) < tol for p in (b0, b1))
    return near >= 2


# ----------------------------------------------------------------------------- rules
def _bbox(d: dict):
    bb = (d.get("extent") or {}).get("bbox_uv_m")
    if bb and len(bb) == 4:
        return [float(x) for x in bb]
    return None


def _match(rule: dict, d: dict, idx: _Index) -> bool:
    cond = rule.get("when", {})
    cls = d.get("class")
    dc = cond.get("damage_class", "any")
    if dc != "any" and cls not in dc:
        return False
    sid = d["surface_id"]
    kind = idx.surface_kind(sid)
    if "surface_kind" in cond and kind not in cond["surface_kind"]:
        return False
    room = idx.surface_room(sid)
    if room is None:
        return False
    bb = _bbox(d)

    if "max_height_from_floor" in cond and (bb is None or bb[3] > float(cond["max_height_from_floor"])):
        return False
    if "room_type" in cond and not (idx.room_types(room["id"]) & set(cond["room_type"])):
        return False
    if "room_adjacent_to" in cond:
        want = set(cond["room_adjacent_to"])
        if not any(idx.room_types(o) & want for o in idx.adj.get(room["id"], ())):
            return False

    wall = idx.wall(sid)
    if "wall_shared_with" in cond and (
        wall is None or not (idx.other_side(wall, room) & set(cond["wall_shared_with"]))
    ):
        return False
    if "other_side_has" in cond and (
        wall is None or not (idx.other_side_damage_classes(wall, room) & set(cond["other_side_has"]))
    ):
        return False
    loc = cond.get("location")
    if loc == "corner":
        if wall is None or bb is None:
            return False
        margin = float(cond.get("corner_margin_m", 0.4))
        L = _val(wall.get("length"))
        if not (bb[0] <= margin or (L > 0 and bb[2] >= L - margin)):
            return False
    elif loc == "opening_corner":
        if wall is None or bb is None:
            return False
        margin = float(cond.get("opening_margin_m", 0.5))
        edges = []
        for o in idx.wall_openings(wall, room):
            off, wid = _val(o.get("offset_along_wall")), _val(o.get("width"))
            edges += [off, off + wid]
        u0, u1 = bb[0], bb[2]
        if not any(max(0.0, e - u1, u0 - e) <= margin for e in edges):
            return False
    return True


def concealed_flags(rooms_json: list[dict], adjacency: list[dict], damage: list[dict],
                    rules: list[dict] | None = None) -> list[dict]:
    """Schema `concealed_flag` dicts, one per (rule, surface), triggered_by = damage ids."""
    idx = _Index(rooms_json, adjacency, damage)
    rules = rules if rules is not None else load_rules()
    hits: dict[tuple, list[str]] = {}
    for rule in rules:
        for d in damage:
            if _match(rule, d, idx):
                hits.setdefault((rule["id"], d["surface_id"]), []).append(d["id"])
    flags = []
    by_id = {r["id"]: r for r in rules}
    for n, ((rid, sid), dids) in enumerate(hits.items(), 1):
        rule = by_id[rid]
        conf = float(rule.get("confidence", 0.5))
        flags.append({
            "id": f"F{n:03d}", "rule_id": rid, "rule_text": rule["text"],
            "triggered_by": list(dict.fromkeys(dids)), "surface_ids": [sid],
            "confidence": round(min(0.8, max(0.4, conf)), 3),
        })
    return flags


# ----------------------------------------------------------------------------- scope items
_REGION_VARS = ("region_area", "bbox_w", "bbox_h", "bbox_longest")


def _eval_qty(expr, env: dict) -> float:
    if isinstance(expr, (int, float)):
        return float(expr)
    return float(eval(str(expr), {"__builtins__": {}}, {**env, "max": max, "min": min, "abs": abs}))


def scope_items(rooms_json: list[dict], damage: list[dict], catalogue: dict | None = None) -> list[dict]:
    """Schema `scope_item` dicts keyed to surfaces; quantities carry a 10% sigma interval."""
    idx = _Index(rooms_json, [], damage)
    cat = catalogue if catalogue is not None else load_catalogue()
    acc: dict[tuple, dict] = {}  # (surface_id, code) -> accumulator
    for d in damage:
        sid, cls = d["surface_id"], d.get("class", "other")
        kind = idx.surface_kind(sid)
        items = cat.get(f"{cls}/{kind}") or cat.get(f"{cls}/*") or []
        bb = _bbox(d) or [0.0, 0.0, 0.0, 0.0]
        bw, bh = max(0.0, bb[2] - bb[0]), max(0.0, bb[3] - bb[1])
        env = {
            "surface_area": idx.surface_area(sid),
            "region_area": _val((d.get("extent") or {}).get("area_m2")),
            "bbox_w": bw, "bbox_h": bh, "bbox_longest": max(bw, bh),
        }
        for it in items:
            expr = it.get("qty", 1)
            q = _eval_qty(expr, env)
            per_region = isinstance(expr, str) and any(v in expr for v in _REGION_VARS)
            key = (sid, it["code"])
            a = acc.get(key)
            if a is None:
                acc[key] = {"surface_id": sid, "code": it["code"], "description": it["description"],
                            "unit": it["unit"], "qty": q, "derived_from": [d["id"]], "per_region": per_region}
            else:
                a["derived_from"].append(d["id"])
                if per_region:
                    a["qty"] += q
                else:
                    a["qty"] = max(a["qty"], q)
    out = []
    for n, a in enumerate(acc.values(), 1):
        out.append({
            "id": f"S{n:03d}", "surface_id": a["surface_id"], "code": a["code"],
            "description": a["description"],
            "quantity": measurement(a["qty"], f"catalogue:{a['code']}", len(a["derived_from"])),
            "unit": a["unit"], "derived_from": a["derived_from"],
        })
    return out
