"""Build the plan.json document (schema 0.1.0) from RoomEstimates, placements and damage."""
from __future__ import annotations

import json
import os

import numpy as np

from floorplan.core.types import Measurement, RoomEstimate
from floorplan.geometry.stitch import room_polygon, transform, wall_segment

Z = 1.645  # 90 % two-sided


def _factors(tier: str) -> dict:
    p = os.path.join(os.path.dirname(__file__), "..", "calib", "factors.json")
    try:
        with open(p) as f:
            return json.load(f).get(tier, {})
    except Exception:
        return {}


def _m(m: Measurement, tier: str, kind: str, factors: dict) -> dict:
    k = float(factors.get(kind, 1.0))
    m.ci_low = m.value - Z * m.sigma * k
    m.ci_high = m.value + Z * m.sigma * k
    d = m.to_json()
    d["method"] = m.method
    return d


def build_plan(rooms: list[RoomEstimate], edges: list[dict], tier: str, capture: dict, timing: dict,
               drift: dict | None, render: dict, damage_regions: list[dict] | None = None,
               warnings: list[str] | None = None) -> dict:
    factors = _factors(tier)
    out_rooms = []
    surfaces_index = {}
    for r in rooms:
        poly = room_polygon(r)
        walls = []
        surfaces = []
        rid = r.key
        Lx, Ly = r.length_x, r.length_y
        for letter in "ABCD":
            (x0, y0), (x1, y1) = wall_segment(r, letter)
            k, tx, ty = r.placement.get("rot_k", 0), r.placement.get("x", 0.0), r.placement.get("y", 0.0)
            s, e = transform((x0, y0), k, tx, ty), transform((x1, y1), k, tx, ty)
            L = Lx if letter in "AC" else Ly
            lm = Measurement(L.value, L.sigma, L.source, L.method, L.n_support)
            sid = f"{rid}:wall:{letter}"
            walls.append({"id": f"{rid}:{letter}", "start": [round(float(s[0]), 4), round(float(s[1]), 4)],
                          "end": [round(float(e[0]), 4), round(float(e[1]), 4)],
                          "length": _m(lm, tier, "wall", factors), "surface_id": sid})
            area = Measurement(L.value * r.ceiling.value, np.hypot(L.sigma * r.ceiling.value, r.ceiling.sigma * L.value),
                               "measured", "wall length × ceiling height", 1)
            surfaces.append({"id": sid, "room_id": rid, "kind": "wall", "area": _m(area, tier, "area", factors)})
            surfaces_index[sid] = (rid, letter)
        fa = Measurement(Lx.value * Ly.value, np.hypot(Lx.sigma * Ly.value, Ly.sigma * Lx.value),
                         "measured" if (Lx.source == "measured" and Ly.source == "measured") else "inferred",
                         "length × width", min(Lx.n_support, Ly.n_support))
        for kind in ("floor", "ceiling"):
            sid = f"{rid}:{kind}"
            surfaces.append({"id": sid, "room_id": rid, "kind": kind,
                             "area": _m(Measurement(fa.value, fa.sigma, fa.source, fa.method, fa.n_support), tier, "area", factors)})
        openings = []
        for j, o in enumerate(r.openings):
            d = {"id": f"{rid}:{o.wall_id}:{o.type}:{j}", "wall_id": f"{rid}:{o.wall_id}", "type": o.type,
                 "offset_along_wall": _m(o.offset, tier, "offset", factors),
                 "width": _m(o.width, tier, "opening", factors),
                 "detection_confidence": round(float(o.confidence), 3)}
            if o.height is not None:
                d["height"] = _m(o.height, tier, "opening", factors)
            if o.sill is not None:
                d["sill_height"] = _m(o.sill, tier, "opening", factors)
            if o.connects_to:
                d["connects_to_room_id"] = o.connects_to
            openings.append(d)
        out_rooms.append({
            "id": rid, "label": r.label, "label_confidence": 0.9 if r.label else 0.3,
            "polygon": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in poly],
            "walls": walls,
            "ceiling_height": _m(r.ceiling, tier, "ceiling", factors),
            "ceiling_height_spread_m": round(float(r.ceiling_spread), 4),
            "floor_area": _m(fa, tier, "area", factors),
            "openings": openings, "surfaces": surfaces,
            "quality": {"coverage_fraction": round(float(r.coverage), 3), "n_frames": int(r.n_frames),
                        "exposure_score": round(float(r.exposure), 3), "mirror_suspected": bool(r.mirror_suspected)},
            "placement_confidence": round(float(r.placement.get("confidence", 0.5)), 3),
            "warnings": list(r.warnings),
        })
    plan = {
        "schema_version": "0.1.0",
        "capture": capture,
        "tier": tier,
        "units": "m",
        "coordinate_frame": "gravity-aligned, z up, x/y in metres; origin at the first room's corner (walls D and C)",
        "rooms": out_rooms,
        "adjacency": edges,
        "damage_regions": damage_regions or [],
        "concealed_damage_flags": [],
        "scope_items": [],
        "calibration": {"nominal_coverage": 0.9, "method": "propagated sigma × per-tier conformal factor",
                        "calibration_set_id": "bench/latest", "tier_floor_fraction": {"photo": 0.035, "video": 0.02, "lidar": 0.004}[tier],
                        "factors": factors},
        "timing": timing,
        "render": render,
        "warnings": warnings or [],
    }
    if drift is not None:
        plan["drift"] = drift
    return plan


def validate(plan: dict) -> list[str]:
    import jsonschema
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schema", "output.schema.json")
    with open(schema_path) as f:
        schema = json.load(f)
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in v.iter_errors(plan)]
