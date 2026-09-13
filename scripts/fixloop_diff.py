"""Render the fix-loop before/after comparison into analysis/fixloop/DIFF.md."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np


def repeatability(results):
    from shapely.geometry import Polygon
    out = []
    for name, v in results.items():
        if not v.get("repeat_of"):
            continue
        a, b = v["repeat_of"], name
        A = json.load(open(os.path.join(v["_dir"], a, "plan.json")))
        B = json.load(open(os.path.join(v["_dir"], b, "plan.json")))
        PA = [(r["id"], Polygon(r["polygon"]), r) for r in A["rooms"] if len(r["polygon"]) >= 3]
        PB = [(r["id"], Polygon(r["polygon"]), r) for r in B["rooms"] if len(r["polygon"]) >= 3]
        ious, cds, wds = [], [], []
        paired = 0
        for _ida, ga, ra in PA:
            best, br = 0.0, None
            for _idb, gb, rb in PB:
                inter = ga.buffer(0).intersection(gb.buffer(0)).area
                iou = inter / max(ga.area + gb.area - inter, 1e-6)
                if iou > best:
                    best, br = iou, rb
            ious.append(best)
            if br is None or best < 0.3:
                continue
            paired += 1
            cds.append(abs(ra["ceiling_height"]["value"] - br["ceiling_height"]["value"]))
            from floorplan.cli.bench import _wall_diffs
            wds += _wall_diffs(ra, br)
        out.append(dict(n_rooms_a=len(PA), n_rooms_b=len(PB), paired=paired, n_wall_pairs=len(wds),
                        median_iou=float(np.median(ious)) if ious else 0.0,
                        max_ceiling_diff=float(max(cds)) if cds else float("nan"),
                        median_wall_diff=float(np.median(wds)) if wds else float("nan"),
                        max_wall_diff=float(max(wds)) if wds else float("nan")))
    return out[0] if out else {}


def load(d):
    r = json.load(open(os.path.join(d, "results.json")))
    for v in r.values():
        v["_dir"] = d
    return r


def main(before_dir="analysis/fixloop/before", after_dir="analysis/fixloop/after", out="analysis/fixloop/DIFF.md"):
    A, B = load(before_dir), load(after_dir)
    ra, rb = repeatability(A), repeatability(B)
    L = ["# Fix loop: before and after", "",
         f"`before` = tag `fixloop-before` ({rev('fixloop-before')}), `after` = tag `fixloop-after` "
         f"({rev('fixloop-after')}). Both runs regenerate with `make before` and `make after`.", "",
         "The declaration, written before the fix existed, is in [DECLARATION.md](DECLARATION.md). "
         "What these numbers mean, including the prediction that was wrong and the row that "
         "regressed, is in [POSTMORTEM.md](POSTMORTEM.md).", "",
         "## The gate that was declared", "",
         "Repeatability, LiDAR tier: two passes over the same rooms sharing no frames.", "",
         "| Metric | Before | Predicted | After | Gate | Verdict |", "|---|---|---|---|---|---|"]

    def row(label, b, p, a, gate, better_is_lower=True, fmt="{:.2f}", ok=None):
        L.append(f"| {label} | {b} | {p} | {a} | {gate} | {ok} |")

    same_before = ra.get("n_rooms_a") == ra.get("n_rooms_b")
    same_after = rb.get("n_rooms_a") == rb.get("n_rooms_b")
    row("Rooms found, pass A vs pass B", f"{ra.get('n_rooms_a')} vs {ra.get('n_rooms_b')}", "3 vs 3",
        f"{rb.get('n_rooms_a')} vs {rb.get('n_rooms_b')}", "must agree",
        ok="PASS" if same_after else ("no change" if same_before == same_after else "FAIL"))
    row("Median paired-room footprint IoU", f"{ra.get('median_iou', 0):.2f}", ">= 0.85",
        f"{rb.get('median_iou', 0):.2f}", "-", ok="PASS" if rb.get("median_iou", 0) >= 0.85 else "short")
    row("Worst ceiling difference between passes", f"{ra.get('max_ceiling_diff', float('nan'))*100:.1f} cm",
        "<= 1 cm", f"{rb.get('max_ceiling_diff', float('nan'))*100:.1f} cm", "<= 1 cm",
        ok="PASS" if rb.get("max_ceiling_diff", 9) <= 0.01 else "FAIL")
    row("Median wall difference between passes", "-", "<= 3 cm",
        f"{rb.get('median_wall_diff', float('nan'))*100:.1f} cm", "-",
        ok="PASS" if rb.get("median_wall_diff", 9) <= 0.03 else "short")
    row("Worst wall difference between passes", f"{ra.get('max_wall_diff', float('nan'))*100:.1f} cm",
        "<= 15 cm (gate not expected to pass)", f"{rb.get('max_wall_diff', float('nan'))*100:.1f} cm",
        "<= 1 cm", ok="PASS" if rb.get("max_wall_diff", 9) <= 0.01 else "FAIL, as predicted")

    L += ["", "## Everything else, so the fix cannot hide a regression", "",
          "| Capture | Rooms before | Rooms after | Footprint before | Footprint after | Openings before | Openings after | Overlap before | Overlap after | Seconds before | Seconds after |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(set(A) | set(B)):
        a = A.get(k, {}).get("summary", {})
        b = B.get(k, {}).get("summary", {})
        def g(d, f, fmt="{}"):
            return (fmt.format(d[f]) if f in d else "-")
        L.append(f"| {k} | {g(a,'n_rooms')} | {g(b,'n_rooms')} | {g(a,'footprint_m2','{:.2f}')} | "
                 f"{g(b,'footprint_m2','{:.2f}')} | {g(a,'n_openings')} | {g(b,'n_openings')} | "
                 f"{g(a,'overlap_pct','{:.1f} %')} | {g(b,'overlap_pct','{:.1f} %')} | "
                 f"{g(a,'total_s','{:.1f}')} | {g(b,'total_s','{:.1f}')} |")

    L += ["", "## Code diff", "", "```", sh("git diff --stat fixloop-before fixloop-after -- floorplan"), "```", "",
          "The whole fix is in the room-seeding half of `floorplan/geometry/cloud.py`. The pre-fix "
          "seeding is kept as `_threshold_cascade_seeds` and is still reachable with "
          "`floorplan run ... --room-seeds cascade`, so the before-run is reproducible from the "
          "after-run's code.", "",
          "```diff", sh("git diff fixloop-before fixloop-after -- floorplan/geometry/cloud.py"), "```"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {out}")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return f"(diff unavailable: {e})"


def rev(tag):
    return sh(f"git rev-parse --short {tag}") or "unset"


if __name__ == "__main__":
    main(*(sys.argv[1:4] if len(sys.argv) > 1 else []))
