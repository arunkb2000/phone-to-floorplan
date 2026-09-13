"""Print a markdown table of per-gate metrics before vs after."""
import json
import sys

import numpy as np

from floorplan.cli.bench import wall_gate_ok


def metrics(results):
    out = {}
    for tier in ("lidar", "video", "photo"):
        rows = [r for v in results.values() if v["tier"] == tier and v["variant"].get("drift", "on") == "on" for r in v["rows"]]
        if not rows:
            continue
        walls = [r for r in rows if r["kind"] == "wall"]
        ceil = [r for r in rows if r["kind"] == "ceiling"]
        ops = [r for r in rows if r["kind"] == "opening"]
        opm = sum(1 for r in rows if r["kind"] == "opening_missed")
        phantom = sum(v["summary"]["openings"]["phantom"] for v in results.values() if v["tier"] == tier and v["variant"].get("drift", "on") == "on")
        if walls:
            out[f"{tier}: walls passing"] = f"{sum(wall_gate_ok(r['err'], r['gt'], tier) for r in walls)}/{len(walls)}"
            out[f"{tier}: wall MAE (cm)"] = f"{np.mean([abs(r['err']) for r in walls])*100:.2f}"
        if ceil:
            out[f"{tier}: ceiling MAE (cm)"] = f"{np.mean([abs(r['err']) for r in ceil])*100:.2f}"
        n = len(ops) + opm
        if n:
            w2 = sum(abs(r["err"]) <= 0.02 for r in ops)
            out[f"{tier}: openings ≤2 cm"] = f"{w2}/{n} (+{phantom} phantom) = {100*w2/max(n+phantom,1):.0f} %"
        scored = [r for r in rows if r["kind"] in ("wall", "ceiling", "opening", "area") and r["err"] is not None and np.isfinite(r["err"])]
        if scored:
            out[f"{tier}: 90 % CI coverage"] = f"{100*np.mean([r['covered'] for r in scored]):.0f} %"
        for k, v in results.items():
            if v["tier"] == tier and v["summary"]["n_rooms_gt"] >= 2 and v["variant"].get("drift", "on") == "on":
                out[f"{tier}: footprint err ({k})"] = f"{v['summary']['footprint_err_pct']:+.1f} %"
    return out


a, b = (json.load(open(p)) for p in sys.argv[1:3])
ma, mb = metrics(a), metrics(b)
print("| Metric | Before | After |")
print("|---|---|---|")
for k in sorted(set(ma) | set(mb)):
    print(f"| {k} | {ma.get(k, '—')} | {mb.get(k, '—')} |")
