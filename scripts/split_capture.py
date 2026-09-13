"""Split one Stray Scanner capture into two independent captures over the same rooms.

The repeatability gate asks for two captures of the same room at the same tier. We were supplied
three captures and no way to re-visit the property, so we make the pair the only honest way the data
allows: cut a long walk into two halves that both cover the same spaces, and run the pipeline on
each independently. The halves share no frames, accumulate drift separately, see the rooms from
different positions and at different times, and neither knows the other exists.

What this is not: it is not two separate walks, so it cannot expose an error that repeats because
the walker always stands in the same place. The report says so.

    uv run python scripts/split_capture.py data/raw/single_scan_with_ceiling --out data/benchmark/captures
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out", default="data/benchmark/captures")
    ap.add_argument("--name", default="")
    ap.add_argument("--mode", choices=["halves", "interleave"], default="interleave")
    ap.add_argument("--link", action="store_true", default=True)
    args = ap.parse_args()
    base = args.name or os.path.basename(args.capture.rstrip("/"))
    rows = list(csv.reader(open(os.path.join(args.capture, "odometry.csv"))))
    header, body = rows[0], [r for r in rows[1:] if len(r) >= 9]
    if args.mode == "halves":
        parts = {"a": body[:len(body) // 2], "b": body[len(body) // 2:]}
    else:
        # interleaved blocks of 2 s: both halves visit every room, neither shares a frame
        blocks, cur, t0 = [], [], float(body[0][0])
        for r in body:
            if float(r[0]) - t0 > 2.0:
                blocks.append(cur)
                cur, t0 = [], float(r[0])
            cur.append(r)
        blocks.append(cur)
        parts = {"a": [r for i, b in enumerate(blocks) if i % 2 == 0 for r in b],
                 "b": [r for i, b in enumerate(blocks) if i % 2 == 1 for r in b]}
    for tag, part in parts.items():
        d = os.path.join(args.out, f"{base}_rep{tag}")
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(args.capture, "camera_matrix.csv"), d)
        with open(os.path.join(d, "odometry.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(part)
        for sub in ("depth", "confidence"):
            sd = os.path.join(d, sub)
            os.makedirs(sd, exist_ok=True)
            for r in part:
                fi = int(float(r[1]))
                src = os.path.join(args.capture, sub, f"{fi:06d}.png")
                dst = os.path.join(sd, f"{fi:06d}.png")
                if os.path.exists(src) and not os.path.exists(dst):
                    (os.link if args.link else shutil.copy)(src, dst)
        print(f"  {d}: {len(part)} frames")


if __name__ == "__main__":
    main()
