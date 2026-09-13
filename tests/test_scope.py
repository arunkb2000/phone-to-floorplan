"""Concealed-damage rules and scope catalogue on minimal schema JSON."""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np

from floorplan.scope.engine import concealed_flags, scope_items

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "src/floorplan/schema/output.schema.json").read_text())


def m(v):
    return {"value": v, "ci_low": v * 0.9, "ci_high": v * 1.1, "sigma": v * 0.05, "source": "measured"}


def _room(rid, label, L=4.0, Wd=3.0, openings=(), shared=None):
    shared = shared or {}
    corners = [(0, 0), (L, 0), (L, Wd), (0, Wd)]
    walls, surfaces = [], []
    for k, name in enumerate("ABCD"):
        s, e = corners[k], corners[(k + 1) % 4]
        sid = f"{rid}_wall_{name}"
        w = {"id": f"{rid}_{name}", "start": list(s), "end": list(e),
             "length": m(float(np.hypot(e[0] - s[0], e[1] - s[1]))), "surface_id": sid}
        if name in shared:
            w["shared_with_room_id"] = shared[name]
        walls.append(w)
        surfaces.append({"id": sid, "room_id": rid, "kind": "wall", "area": m(w["length"]["value"] * 2.6)})
    surfaces.append({"id": f"{rid}_ceiling", "room_id": rid, "kind": "ceiling", "area": m(L * Wd)})
    surfaces.append({"id": f"{rid}_floor", "room_id": rid, "kind": "floor", "area": m(L * Wd)})
    return {"id": rid, "label": label, "polygon": [list(c) for c in corners], "walls": walls,
            "ceiling_height": m(2.6), "floor_area": m(L * Wd), "openings": list(openings), "surfaces": surfaces}



def fixture():
    door = {"id": "O1", "wall_id": "R1_C", "type": "door", "offset_along_wall": m(1.0), "width": m(0.9),
            "detection_confidence": 0.9, "connects_to_room_id": "R2"}
    rooms = [_room("R1", "bedroom", openings=[door], shared={"C": "R2"}),
             _room("R2", "kitchen", shared={"A": "R1"})]
    adjacency = [{"room_a": "R1", "room_b": "R2", "via_opening_id": "O1", "evidence": "trajectory",
                  "confidence": 0.9}]
    damage = [
        {"id": "D1", "surface_id": "R1_ceiling", "class": "water_stain", "class_confidence": 0.7,
         "extent": {"area_m2": m(0.3), "bbox_uv_m": [1.0, 1.0, 1.6, 1.5]}},
        {"id": "D2", "surface_id": "R1_wall_C", "class": "crack", "class_confidence": 0.6,
         "extent": {"area_m2": m(0.02), "bbox_uv_m": [2.0, 1.6, 2.3, 2.4]}},   # 0.1 m from door edge 1.9
    ]
    return rooms, adjacency, damage


def test_rules_r1_r4_fire_and_validate():
    rooms, adjacency, damage = fixture()
    flags = concealed_flags(rooms, adjacency, damage)
    for f in flags:
        jsonschema.validate(f, {**SCHEMA["$defs"]["concealed_flag"], "$defs": SCHEMA["$defs"]})
        assert 0.4 <= f["confidence"] <= 0.8
    by_rule = {f["rule_id"]: f for f in flags}
    assert "R1" in by_rule and by_rule["R1"]["triggered_by"] == ["D1"]
    assert by_rule["R1"]["surface_ids"] == ["R1_ceiling"]
    assert "R4" in by_rule and by_rule["R4"]["triggered_by"] == ["D2"]
    assert "R2" not in by_rule and "R3" not in by_rule and "R6" not in by_rule


def test_r4_needs_crack_near_opening_edge():
    rooms, adjacency, damage = fixture()
    damage[1]["extent"]["bbox_uv_m"] = [3.0, 1.6, 3.3, 2.4]      # 1.1 m from door edge
    assert "R4" not in {f["rule_id"] for f in concealed_flags(rooms, adjacency, damage)}


def test_r2_r5_r6_paths():
    rooms, adjacency, damage = fixture()
    damage = [
        {"id": "D3", "surface_id": "R1_wall_C", "class": "peeling_paint", "class_confidence": 0.5,
         "extent": {"area_m2": m(0.1), "bbox_uv_m": [0.2, 0.0, 0.6, 0.3]}},       # low, shared with kitchen
        {"id": "D4", "surface_id": "R2_wall_A", "class": "water_stain", "class_confidence": 0.5,
         "extent": {"area_m2": m(0.1), "bbox_uv_m": [1.0, 1.0, 1.4, 1.3]}},       # other side has peeling
    ]
    rooms[1]["label"] = "bathroom"
    rules = {f["rule_id"]: f for f in concealed_flags(rooms, adjacency, damage)}
    assert rules["R2"]["triggered_by"] == ["D3"]
    assert rules["R5"]["triggered_by"] == ["D4"]
    assert rules["R6"]["triggered_by"] == ["D4"] and rules["R6"]["surface_ids"] == ["R2_wall_A"]


def test_scope_items_catalogue():
    rooms, _, damage = fixture()
    items = scope_items(rooms, damage)
    for it in items:
        jsonschema.validate(it, {**SCHEMA["$defs"]["scope_item"], "$defs": SCHEMA["$defs"]})
        q = it["quantity"]
        assert abs(q["sigma"] - 0.1 * q["value"]) < 1e-6
        assert abs(q["ci_high"] - (q["value"] + 1.645 * q["sigma"])) < 1e-3
    stain = [it for it in items if it["code"] == "PNT-STAINBLOCK-CEIL"]
    assert len(stain) == 1 and stain[0]["surface_id"] == "R1_ceiling" and stain[0]["unit"] == "m2"
    assert abs(stain[0]["quantity"]["value"] - 12.0) < 1e-6                       # whole ceiling area
    assert any(it["code"] == "PLB-LEAK-INVESTIGATE" and it["unit"] == "ea" for it in items)
    crack = [it for it in items if it["code"] == "CRK-RAKE-FILL-TAPE"]
    assert len(crack) == 1 and crack[0]["unit"] == "lm" and crack[0]["surface_id"] == "R1_wall_C"
    assert abs(crack[0]["quantity"]["value"] - 0.8) < 1e-6                        # bbox longest side
    assert crack[0]["derived_from"] == ["D2"]


def test_scope_items_aggregate_per_surface():
    rooms, _, _ = fixture()
    damage = [
        {"id": "M1", "surface_id": "R1_wall_A", "class": "mold", "class_confidence": 0.5,
         "extent": {"area_m2": m(0.2), "bbox_uv_m": [0, 0, 1, 1]}},
        {"id": "M2", "surface_id": "R1_wall_A", "class": "mold", "class_confidence": 0.5,
         "extent": {"area_m2": m(0.4), "bbox_uv_m": [2, 0, 3, 1]}},
    ]
    items = {it["code"]: it for it in scope_items(rooms, damage)}
    assert abs(items["MLD-TREAT"]["quantity"]["value"] - (0.2 + 0.4) * 2.5) < 1e-6   # summed
    assert items["MLD-TREAT"]["derived_from"] == ["M1", "M2"]
    assert len([c for c in items if c == "PNT-REPAINT"]) == 1                          # once per surface
    assert abs(items["PNT-REPAINT"]["quantity"]["value"] - 4.0 * 2.6) < 1e-6
