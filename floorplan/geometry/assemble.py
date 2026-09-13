"""Build the plan.json document (schema 0.1.0) from RoomOut objects."""
from __future__ import annotations

import json
import os

import numpy as np

from floorplan.core.room import RoomOut
from floorplan.core.types import Measurement

Z = 1.645  # two-sided 90 %
FACTORS_PATH = os.path.join(os.path.dirname(__file__), "..", "calib", "factors.json")


def load_factors(tier: str) -> dict:
    try:
        with open(FACTORS_PATH) as f:
            return json.load(f).get(tier, {})
    except Exception:
        return {}


def mj(m: Measurement, tier: str, kind: str, factors: dict, floor_rel: float = 0.0) -> dict:
    """Measurement -> JSON with a calibrated 90 % interval."""
    k = float(factors.get(kind, factors.get("_default", 1.0)))
    sigma = max(float(m.sigma) * k, floor_rel * abs(float(m.value)), SIGMA_FLOOR_M.get(kind, 0.0))
    if m.source == "prior":
        sigma = max(sigma, float(m.sigma))
    return {"value": round(float(m.value), 4),
            "ci_low": round(float(m.value - Z * sigma), 4),
            "ci_high": round(float(m.value + Z * sigma), 4),
            "sigma": round(float(sigma), 4),
            "source": m.source, "method": m.method, "n_support": int(m.n_support)}


# Relative interval floor per tier: thin input may never claim thick-input precision, whatever the
# propagated sigma says.
TIER_FLOOR = {"photo": 0.045, "video": 0.018, "lidar": 0.0}

# Absolute interval floor, in metres, per measurement kind. Averaging a hundred thousand LiDAR
# returns drives the STATISTICAL spread of a plane fit into the tenths of a millimetre, which is
# nonsense as an accuracy claim: it says nothing about the sensor's own bias, the floor datum, or the
# fact that a plastered wall is not a plane. Apple publishes no accuracy figure for the sensor;
# independent measurements put it around a centimetre at these ranges, and that is the number below
# which no amount of averaging is meaningful. These are asserted from the sensor's physics, not
# fitted to our own benchmark, which is why they are here and not in calib/factors.json.
SIGMA_FLOOR_M = {"wall": 0.008, "ceiling": 0.010, "opening": 0.010, "offset": 0.012, "area": 0.02}


def build_plan(rooms: list[RoomOut], edges_adj: list[dict], tier: str, capture: dict, timing: dict,
               drift: dict | None, render: dict, damage_regions=None, warnings=None) -> dict:
    factors = load_factors(tier)
    fl = TIER_FLOOR[tier]
    out_rooms = []
    for r in rooms:
        rid = r.key
        poly = np.asarray(r.polygon, float)
        walls, surfaces = [], []
        n = len(poly)
        for k, wid in enumerate(r.wall_ids):
            s, e = poly[k], poly[(k + 1) % n]
            sid = f"{rid}:wall:{wid}"
            walls.append({"id": f"{rid}:{wid}",
                          "start": [round(float(s[0]), 4), round(float(s[1]), 4)],
                          "end": [round(float(e[0]), 4), round(float(e[1]), 4)],
                          "length": mj(r.wall_len[k], tier, "wall", factors, fl),
                          "surface_id": sid})
            L, H = r.wall_len[k], r.ceiling
            area = Measurement(L.value * H.value, float(np.hypot(L.sigma * H.value, H.sigma * L.value)),
                               "measured" if L.source == "measured" and H.source == "measured" else "inferred",
                               "wall length x ceiling height", L.n_support)
            surfaces.append({"id": sid, "room_id": rid, "kind": "wall", "area": mj(area, tier, "area", factors, fl)})
        for kind in ("floor", "ceiling"):
            surfaces.append({"id": f"{rid}:{kind}", "room_id": rid, "kind": kind,
                             "area": mj(r.area, tier, "area", factors, fl)})
        openings = []
        for j, (k, o) in enumerate(r.openings):
            d = {"id": f"{rid}:{r.wall_ids[k]}:{o.type}:{j}", "wall_id": f"{rid}:{r.wall_ids[k]}", "type": o.type,
                 "offset_along_wall": mj(o.offset, tier, "offset", factors, fl),
                 "width": mj(o.width, tier, "opening", factors, fl),
                 "detection_confidence": round(float(o.confidence), 3)}
            if o.height is not None:
                d["height"] = mj(o.height, tier, "opening", factors, fl)
            if o.sill is not None:
                d["sill_height"] = mj(o.sill, tier, "opening", factors, fl)
            if o.connects_to:
                d["connects_to_room_id"] = o.connects_to
            openings.append(d)
        out_rooms.append({
            "id": rid, "label": r.label or rid, "label_confidence": 0.9 if r.label else 0.3,
            "polygon": [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in poly],
            "walls": walls,
            "ceiling_height": mj(r.ceiling, tier, "ceiling", factors, fl),
            "ceiling_height_spread_m": round(float(r.ceiling_spread), 4),
            "floor_area": mj(r.area, tier, "area", factors, fl),
            "openings": openings, "surfaces": surfaces,
            "quality": {"coverage_fraction": round(float(r.coverage), 3), "n_frames": int(r.n_frames),
                        "exposure_score": round(float(r.exposure), 3), "mirror_suspected": bool(r.mirror_suspected)},
            "placement_confidence": round(float(r.placement_confidence), 3),
            "warnings": list(r.warnings),
        })
    plan = {
        "schema_version": "0.1.0",
        "capture": capture,
        "tier": tier,
        "units": "m",
        "coordinate_frame": "gravity-aligned, z up, metres; x and y on the property's dominant (Manhattan) axes",
        "rooms": out_rooms,
        "adjacency": edges_adj,
        "damage_regions": damage_regions or [],
        "concealed_damage_flags": [],
        "scope_items": [],
        "calibration": {"nominal_coverage": 0.9,
                        "method": "propagated sigma x per-tier split-conformal factor, with a per-tier relative floor",
                        "calibration_set_id": "bench/latest", "tier_floor_fraction": fl, "factors": factors},
        "timing": timing,
        "render": render,
        "warnings": warnings or [],
    }
    if drift is not None:
        plan["drift"] = drift
    return plan


def validate(plan: dict) -> list[str]:
    import jsonschema
    with open(os.path.join(os.path.dirname(__file__), "..", "schema", "output.schema.json")) as f:
        schema = json.load(f)
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in v.iter_errors(plan)]
