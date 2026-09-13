"""Compare our plan against a consumer scanning app's export, dimension by dimension.

The comparison the brief asks for needs a LiDAR phone inside the benchmark property, and we have
neither. This is the other half of that row: everything except the data. Point it at our plan.json,
the app's dimensions, and the tape measurements, and it prints the table.

The app's numbers go in a small YAML file rather than being parsed out of a PDF, because every app
exports a different PDF and transcribing twelve numbers by hand is honest and takes two minutes:

    app: magicplan
    version: "2026.3.1"
    exported: exports/magicplan/plan.pdf
    rooms:
      - name: living
        dimensions:                  # metres; `truth` is the tape, `app` is theirs
          - {label: "wall A", truth: 4.21, app: 4.18}
          - {label: "ceiling", truth: 2.62, app: 2.60}
          - {label: "door width", truth: 0.82, app: 0.85}
        match: 01_space              # which room in our plan.json this is

    uv run python scripts/head_to_head.py analysis/results/demo/plan.json \\
        data/benchmark/app_exports/magicplan.yaml --out docs/report/head_to_head.md
"""
from __future__ import annotations

import argparse
import json
import re

import yaml

TIE_M = 0.005  # within 5 mm of each other counts as a tie, not a win


def ours(plan: dict, room_id: str, label: str):
    """Find the measurement in our plan that `label` names."""
    room = next((r for r in plan["rooms"] if r["id"] == room_id), None)
    if room is None:
        return None
    low = label.lower()
    if "ceiling" in low:
        return room["ceiling_height"]
    if "area" in low:
        return room["floor_area"]
    m = re.search(r"wall\s+(\w+)", low)
    if m:
        want = m.group(1).upper()
        for w in room["walls"]:
            if w["id"].split(":")[-1].upper() == want:
                return w["length"]
    kind = "door" if "door" in low else ("window" if "window" in low else None)
    if kind:
        cands = [o for o in room["openings"] if o["type"] == kind]
        if cands:
            best = max(cands, key=lambda o: o["detection_confidence"])
            return best["height"] if "height" in low and "height" in best else best["width"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan_json")
    ap.add_argument("app_yaml")
    ap.add_argument("--out", default="docs/report/head_to_head.md")
    args = ap.parse_args()
    plan = json.load(open(args.plan_json))
    app = yaml.safe_load(open(args.app_yaml))

    rows, wins, ties, losses, missing = [], 0, 0, 0, 0
    for room in app["rooms"]:
        for d in room["dimensions"]:
            truth, theirs = float(d["truth"]), float(d["app"])
            mine = ours(plan, room["match"], d["label"])
            if mine is None:
                missing += 1
                rows.append((room["name"], d["label"], truth, None, theirs, None, abs(theirs - truth), "not produced"))
                continue
            mv = float(mine["value"])
            eo, et = abs(mv - truth), abs(theirs - truth)
            if abs(eo - et) <= TIE_M:
                verdict = "tie"
                ties += 1
            elif eo < et:
                verdict = "ours"
                wins += 1
            else:
                verdict = "theirs"
                losses += 1
            rows.append((room["name"], d["label"], truth, mv, theirs, eo, et, verdict))

    scored = wins + ties + losses
    beat_or_tie = (wins + ties) / scored if scored else 0.0
    L = [f"# Head to head: phone-to-floorplan against {app['app']} {app.get('version', '')}".rstrip(), "",
         f"App export: `{app.get('exported', 'n/a')}`. Ground truth is tape. A tie is declared when the "
         f"two errors are within {TIE_M * 100:.1f} cm of each other.", "",
         "| Room | Dimension | Tape | Ours | Theirs | Our error | Their error | Winner |",
         "|---|---|---|---|---|---|---|---|"]
    def m(v):
        return "-" if v is None else f"{v:.3f}"

    def cm(v):
        return "-" if v is None else f"{v * 100:.1f} cm"

    for name, label, truth, mv, theirs, eo, et, verdict in rows:
        L.append(f"| {name} | {label} | {truth:.3f} | {m(mv)} | {theirs:.3f} | {cm(eo)} | {cm(et)} | {verdict} |")
    L += ["", f"**Beat or tie on {wins + ties} of {scored} shared dimensions "
              f"({beat_or_tie:.0%}); the gate is 70 %.** "
              f"{wins} wins, {ties} ties, {losses} losses"
              + (f", {missing} dimensions we did not produce at all." if missing else ".")]
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-1:]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
