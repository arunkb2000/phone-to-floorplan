"""Tests for floorplan.render.plan: fixture validity and SVG/PNG rendering."""
import json
import os
from pathlib import Path

import jsonschema
import pytest
from PIL import Image

from floorplan.render import render_plan

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "floorplan" / "schema" / "output.schema.json"
FIXTURE = ROOT / "tests" / "fixtures" / "sample_plan.json"


@pytest.fixture(scope="module")
def plan() -> dict:
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_validates_against_schema(plan):
    with open(SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(plan)


def test_render_plan_writes_svg_and_png(plan, tmp_path):
    out_svg = tmp_path / "plan.svg"
    out_png = tmp_path / "plan.png"
    render_plan(plan, str(out_svg), str(out_png))

    assert out_svg.exists() and out_svg.stat().st_size > 0
    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")

    with Image.open(out_png) as im:
        assert im.width > 400
        assert im.height > 100


def test_render_plan_tolerates_missing_optional_fields(tmp_path):
    """Minimal document: no labels, intervals, openings, damage or warnings."""
    plan = {
        "tier": "lidar",
        "rooms": [
            {
                "id": "r1",
                "polygon": [[0, 0], [3, 0], [3, 2], [0, 2]],
                "walls": [{"id": "w1", "start": [0, 0], "end": [3, 0]}],
                "openings": [{"id": "o1", "wall_id": "w1", "type": "door", "width": {"value": 0.8}}],
            },
            {"id": "bad", "polygon": [[0, 0]]},  # degenerate, must be skipped
        ],
        "damage_regions": [{"id": "d1", "surface_id": "does-not-exist", "class": "mold"}],
    }
    out_svg, out_png = tmp_path / "min.svg", tmp_path / "min.png"
    render_plan(plan, str(out_svg), str(out_png))
    assert os.path.getsize(out_svg) > 0 and os.path.getsize(out_png) > 0
