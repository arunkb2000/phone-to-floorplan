"""Split-conformal interval calibration.

Each measurement leaves the geometry with a propagated sigma. Those sigmas are honest about the
sources of error we modelled and silent about the ones we did not, so the intervals they imply do
not cover at their nominal rate. We fix that the standard way: on a calibration split we compute
r = |error| / (1.645 * sigma) for every measurement of a given kind, and take the quantile of r that
makes 90 % of the calibration residuals fall inside. That quantile is the factor the tier multiplies
its sigmas by. It is a per-tier, per-kind scalar, stored in floorplan/calib/factors.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "factors.json")
KINDS = ("opening_width", "ceiling_height", "wall_length")
MAP = {"opening_width": "opening", "ceiling_height": "ceiling", "wall_length": "wall"}


def fit_factors(results_path: str, coverage: float = 0.9, out_path: str = OUT, holdout: str = "") -> dict:
    results = json.load(open(results_path))
    acc: dict = {}
    used: dict = {}
    for name, v in results.items():
        if holdout and holdout in name:
            continue
        if v.get("variant", {}).get("drift", "on") == "off":
            continue
        for r in v["rows"]:
            if r["status"] != "matched" or r["sigma"] in (None, 0):
                continue
            acc.setdefault(v["tier"], {}).setdefault(MAP.get(r["kind"], r["kind"]), []).append(
                abs(r["err"]) / (1.645 * r["sigma"]))
            used.setdefault(v["tier"], {}).setdefault(MAP.get(r["kind"], r["kind"]), []).append(name)
    factors = {}
    for tier, kinds in acc.items():
        factors[tier] = {}
        allr = []
        for kind, ratios in kinds.items():
            allr += ratios
            n = len(ratios)
            q = min(1.0, np.ceil((n + 1) * coverage) / n)
            factors[tier][kind] = round(float(max(1.0, np.quantile(ratios, q))), 3)
            factors[tier][kind + "_n"] = n
        if allr:
            n = len(allr)
            q = min(1.0, np.ceil((n + 1) * coverage) / n)
            factors[tier]["_default"] = round(float(max(1.0, np.quantile(allr, q))), 3)
    prev = {}
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
    for t in ("photo", "video", "lidar"):
        prev.setdefault(t, {})
        prev[t].update(factors.get(t, {}))
    prev["_meta"] = {"fitted_from": results_path, "coverage": coverage, "holdout": holdout or None}
    with open(out_path, "w") as f:
        json.dump(prev, f, indent=1)
    print("calibration factors:", json.dumps(factors, indent=1))
    return prev
