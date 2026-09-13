"""Benchmark harness: run every capture in benchmark/bench.yaml, score against ground truth,
write gate tables, repeatability, calibration coverage, drift ablation and timing.

Ground-truth YAML (one per flat): rooms: [{key, label, walls: {A,B,C,D}, ceiling_height,
openings: [{wall, type, offset, width, height, sill}], floor_area}], footprint_area, adjacency: [[a,b],...]
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import yaml
from shapely.geometry import Polygon

from floorplan.cli.main import run as run_cmd

GATES = {
    "wall_lidar": "≤ 1 cm or 0.5 % per wall (repeatability-class accuracy)",
    "opening": "≤ 2 cm on ≥ 85 % of openings; misses and phantoms count as failures",
    "ceiling": "≤ 1.5 cm per room; cross-capture spread ≤ 1 cm",
    "repeat": "two captures agree within 1 cm or 0.5 % per wall",
    "photo_walls": "wall lengths within ±8 %", "video_walls": "wall lengths within ±3 %",
    "photo_footprint": "stitched footprint within ±8 %, no overlaps, correct adjacency",
}


def _load_plan(out_dir):
    with open(os.path.join(out_dir, "plan.json")) as f:
        return json.load(f)


def _room_dims(room):
    """(len_x, len_y) with measurement dicts from a plan room (walls A,B,C,D order)."""
    w = {x["id"].split(":")[-1]: x["length"] for x in room["walls"]}
    return w["A"], w["B"]


def _err(pred, gt):
    return pred["value"] - gt


def _covered(pred, gt):
    return pred["ci_low"] <= gt <= pred["ci_high"]


def score_capture(plan: dict, gt: dict, tier: str, orientation_free: bool):
    rows = []      # every scored measurement: dict(kind, room, name, pred, gt, err, sigma, covered)
    op_stats = dict(matched=0, within2=0, missed=0, phantom=0)
    gt_rooms = {r["key"]: r for r in gt["rooms"]}
    pred_rooms = {r["id"]: r for r in plan["rooms"]}
    # match rooms by key; if keys differ (lidar generic names), match in order
    keys = list(gt_rooms)
    pkeys = list(pred_rooms)
    pairs = [(k, k) for k in keys if k in pred_rooms]
    if not pairs:
        pairs = list(zip(keys, pkeys))
    for gk, pk in pairs:
        g, p = gt_rooms[gk], pred_rooms[pk]
        A, B = _room_dims(p)
        gA, gB = float(g["walls"]["A"]), float(g["walls"]["B"])
        if orientation_free and abs(A["value"] - gB) + abs(B["value"] - gA) < abs(A["value"] - gA) + abs(B["value"] - gB):
            A, B = B, A
            swapped = True
        else:
            swapped = False
        for name, pred, gtv in (("A", A, gA), ("C", A, float(g["walls"]["C"])), ("B", B, gB), ("D", B, float(g["walls"]["D"]))):
            rows.append(dict(kind="wall", room=gk, name=name, pred=pred["value"], gt=gtv, err=_err(pred, gtv),
                             sigma=pred["sigma"], covered=_covered(pred, gtv)))
        ch = p["ceiling_height"]; gch = float(g["ceiling_height"])
        rows.append(dict(kind="ceiling", room=gk, name="h", pred=ch["value"], gt=gch, err=_err(ch, gch),
                         sigma=ch["sigma"], covered=_covered(ch, gch)))
        fa = p["floor_area"]; gfa = float(g.get("floor_area", gA * gB))
        rows.append(dict(kind="area", room=gk, name="area", pred=fa["value"], gt=gfa, err=_err(fa, gfa),
                         sigma=fa["sigma"], covered=_covered(fa, gfa)))
        # openings: greedy match by (wall, type, offset) or, orientation-free, by (type, width)
        gops = list(g.get("openings", []))
        pops = list(p.get("openings", []))
        used = set()
        for go in gops:
            best, bj = None, -1
            for j, po in enumerate(pops):
                if j in used or po["type"] != go["type"]:
                    continue
                pw = po["wall_id"].split(":")[-1]
                if orientation_free:
                    c = abs(po["width"]["value"] - float(go["width"]))
                    if c > 0.35:
                        continue
                else:
                    if pw != go["wall"]:
                        continue
                    c = abs(po["offset_along_wall"]["value"] - float(go["offset"])) + abs(po["width"]["value"] - float(go["width"]))
                    if c > 1.0:
                        continue
                if best is None or c < best:
                    best, bj = c, j
            if bj < 0:
                op_stats["missed"] += 1
                rows.append(dict(kind="opening_missed", room=gk, name=f'{go["wall"]}:{go["type"]}', pred=np.nan,
                                 gt=float(go["width"]), err=np.nan, sigma=np.nan, covered=False))
                continue
            used.add(bj)
            po = pops[bj]
            e = po["width"]["value"] - float(go["width"])
            op_stats["matched"] += 1
            op_stats["within2"] += int(abs(e) <= 0.02)
            rows.append(dict(kind="opening", room=gk, name=f'{go["wall"]}:{go["type"]}', pred=po["width"]["value"],
                             gt=float(go["width"]), err=e, sigma=po["width"]["sigma"], covered=_covered(po["width"], float(go["width"]))))
            if not orientation_free:
                eo = po["offset_along_wall"]["value"] - float(go["offset"])
                rows.append(dict(kind="offset", room=gk, name=f'{go["wall"]}:{go["type"]}', pred=po["offset_along_wall"]["value"],
                                 gt=float(go["offset"]), err=eo, sigma=po["offset_along_wall"]["sigma"],
                                 covered=_covered(po["offset_along_wall"], float(go["offset"]))))
        op_stats["phantom"] += len(pops) - len(used)
    # footprint, overlap, adjacency
    polys = [Polygon(r["polygon"]) for r in plan["rooms"]]
    pred_fp = sum(pl.area for pl in polys)
    gt_fp = float(gt.get("footprint_area", sum(float(r["walls"]["A"]) * float(r["walls"]["B"]) for r in gt["rooms"])))
    overlap = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            overlap += polys[i].intersection(polys[j]).area
    gt_adj = {tuple(sorted(e)) for e in gt.get("adjacency", [])}
    pred_adj = {tuple(sorted((e["room_a"], e["room_b"]))) for e in plan["adjacency"]}
    adj_ok = len(gt_adj & pred_adj); adj_missing = len(gt_adj - pred_adj); adj_extra = len(pred_adj - gt_adj)
    summary = dict(footprint_pred=pred_fp, footprint_gt=gt_fp, footprint_err_pct=100 * (pred_fp - gt_fp) / gt_fp,
                   overlap_m2=overlap, overlap_pct=100 * overlap / max(pred_fp, 1e-6),
                   adjacency_correct=adj_ok, adjacency_missing=adj_missing, adjacency_extra=adj_extra,
                   n_rooms_pred=len(plan["rooms"]), n_rooms_gt=len(gt["rooms"]), openings=op_stats,
                   total_s=plan["timing"]["total_s"])
    return rows, summary


def wall_gate_ok(err, gt, tier):
    a = abs(err)
    if tier == "lidar":
        return a <= max(0.01, 0.005 * gt)
    if tier == "video":
        return a <= 0.03 * gt
    return a <= 0.08 * gt


def run_bench(out: str, captures_dir: str, gt_dir: str, fast: bool = False, manifest: str = "benchmark/bench.yaml"):
    with open(manifest) as f:
        man = yaml.safe_load(f)
    os.makedirs(out, exist_ok=True)
    results = {}
    all_rows = []
    for cap in man["captures"]:
        name, tier = cap["name"], cap["tier"]
        if fast and cap.get("slow"):
            continue
        cdir = os.path.join(captures_dir, name)
        for variant in cap.get("variants", [{"suffix": "", "drift": "on"}]):
            odir = os.path.join(out, name + variant.get("suffix", ""))
            t0 = time.time()
            run_cmd(cdir, out=odir, tier=tier, drift_correction=variant.get("drift", "on"),
                    depth_model=cap.get("depth_model", "small"), damage=cap.get("damage", True))
            plan = _load_plan(odir)
            with open(os.path.join(gt_dir, cap["gt"])) as f:
                gt = yaml.safe_load(f)
            rows, summary = score_capture(plan, gt, tier, orientation_free=(tier == "lidar"))
            for r in rows:
                r.update(capture=name + variant.get("suffix", ""), tier=tier)
            all_rows += rows
            summary["wall_s"] = round(time.time() - t0, 1)
            results[name + variant.get("suffix", "")] = dict(tier=tier, summary=summary, rows=rows, role=cap.get("role", ""),
                                                             variant=variant)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    write_report(out, results, man)
    return results


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_report(out: str, results: dict, man: dict):
    L = ["# Benchmark gates", "", f"Generated by `floorplan bench` into `{out}`. Every number regenerates from the raw captures.", ""]
    # per tier gate table
    for tier in ("lidar", "video", "photo"):
        caps = {k: v for k, v in results.items() if v["tier"] == tier and v["variant"].get("drift", "on") == "on"}
        if not caps:
            continue
        rows = [r for v in caps.values() for r in v["rows"]]
        walls = [r for r in rows if r["kind"] == "wall"]
        ceil = [r for r in rows if r["kind"] == "ceiling"]
        ops = [r for r in rows if r["kind"] == "opening"]
        opm = sum(1 for r in rows if r["kind"] == "opening_missed")
        phantom = sum(v["summary"]["openings"]["phantom"] for v in caps.values())
        L += [f"## Tier: {tier}", "", "| Gate | Requirement | Result | Pass |", "|---|---|---|---|"]
        if walls:
            ok = sum(wall_gate_ok(r["err"], r["gt"], tier) for r in walls)
            mae = np.mean([abs(r["err"]) for r in walls]); mpe = np.mean([abs(r["err"]) / r["gt"] for r in walls]) * 100
            req = {"lidar": "≤ 1 cm or 0.5 %", "video": "±3 %", "photo": "±8 %"}[tier]
            L.append(f"| Wall lengths | {req} | {ok}/{len(walls)} walls pass; MAE {mae*100:.1f} cm ({mpe:.1f} %) | {'PASS' if ok == len(walls) else 'FAIL'} |")
        if ceil:
            ok = sum(abs(r["err"]) <= 0.015 for r in ceil)
            mae = np.mean([abs(r["err"]) for r in ceil]); bias = np.mean([r["err"] for r in ceil])
            L.append(f"| Ceiling height | ≤ 1.5 cm per room | {ok}/{len(ceil)} rooms; MAE {mae*100:.1f} cm, bias {bias*100:+.1f} cm | {'PASS' if ok == len(ceil) else 'FAIL'} |")
        n_gt = len(ops) + opm
        if n_gt:
            w2 = sum(abs(r["err"]) <= 0.02 for r in ops)
            denom = n_gt + phantom
            frac = w2 / max(denom, 1)
            L.append(f"| Opening widths | ≤ 2 cm on ≥ 85 %, misses + phantoms count | {w2}/{n_gt} within 2 cm, {opm} missed, {phantom} phantom → {frac*100:.0f} % | {'PASS' if frac >= 0.85 else 'FAIL'} |")
        # footprint / adjacency / overlap for multi-room captures
        for k, v in caps.items():
            s = v["summary"]
            if s["n_rooms_gt"] >= 2:
                ok = abs(s["footprint_err_pct"]) <= 8 and s["overlap_pct"] < 1 and s["adjacency_missing"] == 0
                L.append(f"| Whole-property stitch ({k}) | footprint ±8 %, no overlap, adjacency correct | footprint {s['footprint_err_pct']:+.1f} %, overlap {s['overlap_pct']:.1f} %, adjacency {s['adjacency_correct']} ok / {s['adjacency_missing']} missing / {s['adjacency_extra']} extra | {'PASS' if ok else 'FAIL'} |")
        # calibration coverage
        scored = [r for r in rows if r["kind"] in ("wall", "ceiling", "opening", "area") and np.isfinite(r["err"])]
        if scored:
            cov = np.mean([r["covered"] for r in scored]) * 100
            L.append(f"| Interval calibration | 90 % nominal coverage | {cov:.0f} % of {len(scored)} measurements inside their 90 % interval | {'PASS' if 80 <= cov <= 98 else 'FAIL'} |")
        L.append("")
    # repeatability
    reps = [(c["name"], c["repeat_of"]) for c in man["captures"] if c.get("repeat_of")]
    if reps:
        L += ["## Repeatability (same room, same tier, two captures)", "", "| Tier | Room | Wall | Capture 1 | Capture 2 | Δ | Pass (1 cm or 0.5 %) |", "|---|---|---|---|---|---|---|"]
        for a, b in reps:
            if a not in results or b not in results:
                continue
            ra = {(r["room"], r["name"]): r for r in results[a]["rows"] if r["kind"] in ("wall", "ceiling")}
            rb = {(r["room"], r["name"]): r for r in results[b]["rows"] if r["kind"] in ("wall", "ceiling")}
            for key in sorted(set(ra) & set(rb)):
                x, y = ra[key]["pred"], rb[key]["pred"]
                d = abs(x - y)
                ok = d <= max(0.01, 0.005 * max(x, y)) if key[1] != "h" else d <= 0.01
                L.append(f"| {results[a]['tier']} | {key[0]} | {key[1]} | {x:.3f} | {y:.3f} | {d*100:.1f} cm | {'PASS' if ok else 'FAIL'} |")
        L.append("")
    # drift ablation
    abl = {k: v for k, v in results.items() if v["variant"].get("drift") == "off"}
    if abl:
        L += ["## Drift ablation (LiDAR multi-room)", "", "| Variant | Footprint err | Wall MAE | Overlap | Adjacency |", "|---|---|---|---|---|"]
        for k, v in results.items():
            if v["tier"] != "lidar" or v["summary"]["n_rooms_gt"] < 2:
                continue
            s = v["summary"]; walls = [r for r in v["rows"] if r["kind"] == "wall"]
            mae = np.mean([abs(r["err"]) for r in walls]) * 100 if walls else float("nan")
            L.append(f"| {k} (drift correction {v['variant'].get('drift','on')}) | {s['footprint_err_pct']:+.1f} % | {mae:.1f} cm | {s['overlap_pct']:.1f} % | {s['adjacency_correct']} ok / {s['adjacency_missing']} missing |")
        L.append("")
    # timing
    L += ["## Timing", "", "| Capture | Tier | Frames | Pipeline s |", "|---|---|---|---|"]
    for k, v in results.items():
        L.append(f"| {k} | {v['tier']} | — | {v['summary']['total_s']:.1f} |")
    L.append("")
    # per-measurement detail
    L += ["## Every scored measurement", "", "| Capture | Room | Kind | Name | Pred | GT | Err (cm) | σ (cm) | In 90 % CI |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        for r in v["rows"]:
            L.append(f"| {k} | {r['room']} | {r['kind']} | {r['name']} | {_fmt(r['pred'])} | {_fmt(r['gt'])} | {_fmt(r['err']*100 if np.isfinite(r['err']) else np.nan,1)} | {_fmt(r['sigma']*100 if np.isfinite(r['sigma']) else np.nan,1)} | {'yes' if r['covered'] else 'no'} |")
    with open(os.path.join(out, "gates.md"), "w") as f:
        f.write("\n".join(L))


def calibrate(results_path: str, out_path: str):
    """Split-conformal factors per tier and kind: 90th percentile of |err| / (1.645 σ)."""
    with open(results_path) as f:
        results = json.load(f)
    fac = {}
    for v in results.values():
        for r in v["rows"]:
            if r["kind"] not in ("wall", "ceiling", "opening", "area", "offset") or r["err"] is None or not np.isfinite(r["err"]) or not r["sigma"]:
                continue
            fac.setdefault(v["tier"], {}).setdefault(r["kind"], []).append(abs(r["err"]) / (1.645 * r["sigma"]))
    out = {}
    for tier, kinds in fac.items():
        out[tier] = {}
        for kind, ratios in kinds.items():
            n = len(ratios)
            q = min(1.0, np.ceil((n + 1) * 0.9) / n)
            out[tier][kind] = round(float(max(0.5, np.quantile(ratios, q))), 3)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out
