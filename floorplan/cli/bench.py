"""Benchmark harness: run every capture in the manifest, score it, write the gate tables.

Scoring against a reference that is a SET of values, not a labelled drawing
--------------------------------------------------------------------------
The reference (scripts/make_reference_gt.py) measures the property one depth frame at a time, so it
knows that a 0.89 m opening exists but not which wall of which room it is on. Scoring is therefore
set matching: every reference value is matched to the closest prediction of the same kind that is
not already taken, and the error is the difference.

Two consequences we do not paper over:

  * A reference value with no prediction within the matching window is a MISS, and counts against
    the opening-detection gate exactly as the brief requires.
  * A prediction with no reference value is reported as UNMATCHED, not as a phantom. The reference
    is not exhaustive - it only sees what a single frame could resolve - so calling every unmatched
    prediction a false positive would be as dishonest as ignoring it. Both counts are printed, and
    the gate is computed twice: once treating unmatched predictions as phantoms (the strict reading
    of the brief) and once not.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import yaml

MATCH_WINDOW = {"opening_width": 0.12, "ceiling_height": 0.25, "wall_to_wall": 0.40}
GATE = {"opening_width": 0.02, "ceiling_height": 0.015}


def load_plan(d):
    with open(os.path.join(d, "plan.json")) as f:
        return json.load(f)


def predictions(plan: dict) -> dict:
    ow, ch, wl = [], [], []
    for r in plan["rooms"]:
        ch.append(dict(room=r["id"], m=r["ceiling_height"]))
        for o in r["openings"]:
            ow.append(dict(room=r["id"], wall=o["wall_id"], type=o["type"], m=o["width"]))
        for w in r["walls"]:
            wl.append(dict(room=r["id"], wall=w["id"], m=w["length"]))
    return {"opening_width": ow, "ceiling_height": ch, "wall_length": wl}


def match(ref_clusters, preds, kind):
    """Greedy nearest matching of reference values to predictions. Returns rows + counts."""
    win = MATCH_WINDOW.get(kind, 0.2)
    rows, used = [], set()
    for c in ref_clusters:
        gt = float(c["value"])
        best, bi = None, -1
        for i, p in enumerate(preds):
            if i in used:
                continue
            e = abs(float(p["m"]["value"]) - gt)
            if e <= win and (best is None or e < best):
                best, bi = e, i
        if bi < 0:
            rows.append(dict(kind=kind, gt=gt, pred=None, err=None, sigma=None, covered=False,
                             gt_sigma=c.get("sigma_m"), gt_n=c.get("n_frames"), status="missed", where=""))
            continue
        used.add(bi)
        p = preds[bi]
        v = float(p["m"]["value"])
        rows.append(dict(kind=kind, gt=gt, pred=v, err=v - gt, sigma=float(p["m"]["sigma"]),
                         covered=bool(p["m"]["ci_low"] <= gt <= p["m"]["ci_high"]),
                         gt_sigma=c.get("sigma_m"), gt_n=c.get("n_frames"), status="matched",
                         where=f"{p.get('room','')}:{p.get('wall','')}"))
    unmatched = [p for i, p in enumerate(preds) if i not in used]
    return rows, unmatched


def score_capture(plan: dict, gt: dict):
    preds = predictions(plan)
    rows, unmatched = [], {}
    for kind in ("opening_width", "ceiling_height"):
        key = {"opening_width": "opening_width", "ceiling_height": "ceiling_height"}[kind]
        pk = {"opening_width": "opening_width", "ceiling_height": "ceiling_height"}[kind]
        r, u = match(gt.get(key) or [], preds[pk], kind)
        rows += r
        unmatched[kind] = u
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = [Polygon(r["polygon"]) for r in plan["rooms"] if len(r["polygon"]) >= 3]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    ov = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            ov += polys[i].intersection(polys[j]).area
    tot = sum(p.area for p in polys)
    summary = dict(
        n_rooms=len(plan["rooms"]),
        footprint_m2=round(float(unary_union(polys).area) if polys else 0.0, 3),
        sum_room_area_m2=round(float(tot), 3),
        overlap_pct=round(100 * ov / max(tot, 1e-6), 2),
        n_openings=sum(len(r["openings"]) for r in plan["rooms"]),
        n_adjacency=len(plan["adjacency"]),
        n_damage=len(plan["damage_regions"]),
        n_flags=len(plan["concealed_damage_flags"]),
        n_scope=len(plan["scope_items"]),
        unmatched_openings=len(unmatched.get("opening_width", [])),
        unmatched_ceilings=len(unmatched.get("ceiling_height", [])),
        ceiling_sources=sorted({r["ceiling_height"]["source"] for r in plan["rooms"]}),
        mean_coverage=round(float(np.mean([r["quality"]["coverage_fraction"] for r in plan["rooms"]])), 3) if plan["rooms"] else 0.0,
        total_s=plan["timing"]["total_s"],
    )
    return rows, summary


def run_bench(out: str, captures_dir: str, gt_dir: str, manifest: str = "benchmark/bench.yaml",
              only: str = "", do_calibrate: bool = False):
    from floorplan.cli.main import run as run_cmd
    man = yaml.safe_load(open(manifest))
    os.makedirs(out, exist_ok=True)
    results = {}
    for cap in man["captures"]:
        if only and only not in cap["name"]:
            continue
        for variant in cap.get("variants", [{"suffix": "", "drift": "on"}]):
            name = cap["name"] + variant.get("suffix", "")
            odir = os.path.join(out, name)
            t0 = time.time()
            run_cmd(cap["path"], out=odir, tier=cap["tier"], drift_correction=variant.get("drift", "on"),
                    damage=cap.get("damage", False))
            wall = time.time() - t0
            plan = load_plan(odir)
            gt = yaml.safe_load(open(os.path.join(gt_dir, cap["gt"]))) if cap.get("gt") else {}
            rows, summary = score_capture(plan, gt)
            for r in rows:
                r.update(capture=name, tier=cap["tier"])
            summary["wall_s"] = round(wall, 1)
            results[name] = dict(tier=cap["tier"], role=cap.get("role", ""), variant=variant,
                                 repeat_of=cap.get("repeat_of"), gt=cap["gt"], path=cap["path"],
                                 summary=summary, rows=rows)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    if do_calibrate:
        from floorplan.calib.fit import fit_factors
        fit_factors(os.path.join(out, "results.json"))
    write_report(out, results, man)
    print(f"\nwrote {out}/gates.md and {out}/results.json")
    return results


def _repeatability(results):
    """Per-room agreement between the two independent passes: rooms are paired by footprint overlap."""
    from shapely.geometry import Polygon
    pairs = [(k, v["repeat_of"]) for k, v in results.items() if v.get("repeat_of")]
    out = []
    for b, a in pairs:
        if a not in results:
            continue
        pa = load_plan(os.path.join(os.path.dirname(""), "")) if False else None
        A = results[a]["_plan"]
        B = results[b]["_plan"]
        PA = [(r["id"], Polygon(r["polygon"])) for r in A["rooms"] if len(r["polygon"]) >= 3]
        PB = [(r["id"], Polygon(r["polygon"])) for r in B["rooms"] if len(r["polygon"]) >= 3]
        for ida, ga in PA:
            best, bid = 0.0, None
            for idb, gb in PB:
                inter = ga.buffer(0).intersection(gb.buffer(0)).area
                iou = inter / max(ga.area + gb.area - inter, 1e-6)
                if iou > best:
                    best, bid = iou, idb
            if bid is None or best < 0.3:
                out.append(dict(room_a=ida, room_b=None, iou=round(best, 3)))
                continue
            ra = next(r for r in A["rooms"] if r["id"] == ida)
            rb = next(r for r in B["rooms"] if r["id"] == bid)
            la = sorted(w["length"]["value"] for w in ra["walls"])
            lb = sorted(w["length"]["value"] for w in rb["walls"])
            n = min(len(la), len(lb))
            dif = [abs(la[i] - lb[i]) for i in range(n)]
            out.append(dict(room_a=ida, room_b=bid, iou=round(best, 3),
                            area_a=round(ra["floor_area"]["value"], 3), area_b=round(rb["floor_area"]["value"], 3),
                            area_diff_pct=round(100 * abs(ra["floor_area"]["value"] - rb["floor_area"]["value"])
                                                / max(ra["floor_area"]["value"], 1e-6), 2),
                            ceil_a=round(ra["ceiling_height"]["value"], 4), ceil_b=round(rb["ceiling_height"]["value"], 4),
                            ceil_diff_m=round(abs(ra["ceiling_height"]["value"] - rb["ceiling_height"]["value"]), 4),
                            wall_max_diff_m=round(max(dif), 4) if dif else None,
                            wall_median_diff_m=round(float(np.median(dif)), 4) if dif else None,
                            n_walls_a=len(la), n_walls_b=len(lb)))
    return out


def write_report(out: str, results: dict, man: dict):
    for k, v in results.items():
        v["_plan"] = load_plan(os.path.join(out, k))
    L = ["# Benchmark gates", "",
         f"Generated by `floorplan bench --out {out}`. Every number below regenerates from the raw captures "
         "with `make reproduce`.", "",
         "Read `docs/report/benchmark.md` first: it states what the reference ground truth is, and which "
         "gates the supplied data can and cannot settle.", ""]

    L += ["## What ran", "", "| Capture | Tier | Rooms | Openings | Footprint m2 | Overlap | Mean wall coverage | Pipeline s |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        s = v["summary"]
        L.append(f"| {k} | {v['tier']} | {s['n_rooms']} | {s['n_openings']} | {s['footprint_m2']:.2f} | "
                 f"{s['overlap_pct']:.1f} % | {s['mean_coverage']:.0%} | {s['total_s']:.1f} |")
    L.append("")

    for tier in ("lidar", "video", "photo"):
        caps = {k: v for k, v in results.items() if v["tier"] == tier and v["variant"].get("drift", "on") == "on"}
        if not caps:
            continue
        rows = [r for v in caps.values() for r in v["rows"]]
        L += [f"## Tier: {tier}", "", "| Gate | Requirement | Result | Verdict |", "|---|---|---|---|"]
        ow = [r for r in rows if r["kind"] == "opening_width"]
        matched = [r for r in ow if r["status"] == "matched"]
        missed = [r for r in ow if r["status"] == "missed"]
        unmatched = sum(v["summary"]["unmatched_openings"] for v in caps.values())
        if ow:
            w2 = sum(abs(r["err"]) <= GATE["opening_width"] for r in matched)
            lenient = w2 / max(len(ow), 1)
            strict = w2 / max(len(ow) + unmatched, 1)
            mae = np.mean([abs(r["err"]) for r in matched]) * 100 if matched else float("nan")
            L.append(f"| Opening width | <= 2 cm on >= 85 % | {w2}/{len(matched)} matched within 2 cm, "
                     f"{len(missed)} reference openings missed, {unmatched} predictions unmatched. "
                     f"Lenient {lenient:.0%}, strict {strict:.0%}. MAE {mae:.1f} cm | "
                     f"{'PASS' if strict >= 0.85 else 'FAIL'} |")
        ch = [r for r in rows if r["kind"] == "ceiling_height" and r["status"] == "matched"]
        chm = [r for r in rows if r["kind"] == "ceiling_height" and r["status"] == "missed"]
        if ch or chm:
            ok = sum(abs(r["err"]) <= GATE["ceiling_height"] for r in ch)
            mae = np.mean([abs(r["err"]) for r in ch]) * 100 if ch else float("nan")
            bias = np.mean([r["err"] for r in ch]) * 100 if ch else float("nan")
            L.append(f"| Ceiling height | <= 1.5 cm per room | {ok}/{len(ch)} matched within 1.5 cm "
                     f"({len(chm)} reference heights unmatched); MAE {mae:.1f} cm, bias {bias:+.1f} cm | "
                     f"{'PASS' if ch and ok == len(ch) else 'FAIL'} |")
        scored = [r for r in rows if r["status"] == "matched"]
        if scored:
            cov = 100 * np.mean([r["covered"] for r in scored])
            L.append(f"| Interval calibration | 90 % nominal coverage | {cov:.0f} % of {len(scored)} matched "
                     f"measurements contain the reference value | {'PASS' if 75 <= cov <= 99 else 'FAIL'} |")
        for k, v in caps.items():
            s = v["summary"]
            L.append(f"| Room overlap ({k}) | stitched rooms must not overlap | {s['overlap_pct']:.1f} % of the "
                     f"total room area | {'PASS' if s['overlap_pct'] < 1.0 else 'FAIL'} |")
        L.append("")

    rep = _repeatability(results)
    if rep:
        L += ["## Repeatability", "",
              "Two passes over the same rooms with disjoint frames (`scripts/split_capture.py`). Rooms are "
              "paired by footprint overlap.", "",
              "| Room A | Room B | IoU | Area A | Area B | Area diff | Ceiling A | Ceiling B | Ceiling diff | Worst wall diff | Verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rep:
            if r.get("room_b") is None:
                L.append(f"| {r['room_a']} | none | {r['iou']} | | | | | | | | UNPAIRED |")
                continue
            ok = (r["wall_max_diff_m"] is not None and r["wall_max_diff_m"] <= 0.01) and r["ceil_diff_m"] <= 0.01
            L.append(f"| {r['room_a']} | {r['room_b']} | {r['iou']} | {r['area_a']:.2f} | {r['area_b']:.2f} | "
                     f"{r['area_diff_pct']:.1f} % | {r['ceil_a']:.3f} | {r['ceil_b']:.3f} | "
                     f"{r['ceil_diff_m']*100:.1f} cm | {r['wall_max_diff_m']*100:.1f} cm | "
                     f"{'PASS' if ok else 'FAIL'} |")
        L.append("")

    abl = {k: v for k, v in results.items() if v["variant"].get("drift") == "off"}
    if abl:
        L += ["## Drift ablation", "",
              "`--drift-correction off` uses the ARKit poses exactly as exported. The brief calls that an "
              "automatic fail, so it is here only as the ablation.", "",
              "| Run | Drift correction | Rooms | Footprint m2 | Sum of room areas | Overlap | Openings |",
              "|---|---|---|---|---|---|---|"]
        for k, v in results.items():
            base = k.replace("_driftoff", "")
            if v["tier"] != "lidar" or (k not in abl and base + "_driftoff" not in results):
                continue
            s = v["summary"]
            L.append(f"| {k} | {v['variant'].get('drift','on')} | {s['n_rooms']} | {s['footprint_m2']:.2f} | "
                     f"{s['sum_room_area_m2']:.2f} | {s['overlap_pct']:.1f} % | {s['n_openings']} |")
        L.append("")

    L += ["## Output contract completeness", "",
          "| Capture | Rooms | Walls dimensioned | Openings | Adjacency | Damage regions | Concealed flags | Scope items | Ceiling source |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        s, p = v["summary"], v["_plan"]
        nw = sum(len(r["walls"]) for r in p["rooms"])
        L.append(f"| {k} | {s['n_rooms']} | {nw} | {s['n_openings']} | {s['n_adjacency']} | {s['n_damage']} | "
                 f"{s['n_flags']} | {s['n_scope']} | {', '.join(s['ceiling_sources'])} |")
    L.append("")

    L += ["## Every scored measurement", "",
          "| Capture | Tier | Kind | Reference | Reference n | Predicted | Error | sigma | In 90 % interval | Where |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        for r in v["rows"]:
            pr = "MISSED" if r["pred"] is None else f"{r['pred']:.3f}"
            er = "" if r["err"] is None else f"{r['err']*100:+.1f} cm"
            sg = "" if r["sigma"] is None else f"{r['sigma']*100:.1f} cm"
            L.append(f"| {k} | {v['tier']} | {r['kind']} | {r['gt']:.3f} | {r['gt_n']} | {pr} | {er} | {sg} | "
                     f"{'yes' if r['covered'] else 'no'} | {r['where']} |")
    for v in results.values():
        v.pop("_plan", None)
    with open(os.path.join(out, "gates.md"), "w") as f:
        f.write("\n".join(L) + "\n")
