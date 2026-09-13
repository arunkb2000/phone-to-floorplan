#!/usr/bin/env python
"""Build the synthetic Stray Scanner benchmark captures.

    uv run python scripts/make_synthetic.py [--out benchmark/captures] [--fps 5] [--seed 0]

Writes
  synthetic_MR1          4 rooms, drift on
  synthetic_MR1_nodrift  same frames (hard-linked), odometry.csv = true poses
  synthetic_R1_repeat    01_room alone, different seed (a second capture of the same room)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from floorplan.io.synthetic import (FlatSpec, RoomSpec, check_shared_doors, generate,  # noqa: E402
                                     load_stray_csv, mirror_offset)


def room1() -> RoomSpec:
    return RoomSpec("01_room", "room", (0.0, 0.0), (4.2, 3.6), 2.72, [
        {"wall": "C", "type": "door", "offset": 0.6, "width": 0.9, "height": 2.05},
        {"wall": "A", "type": "window", "offset": 1.4, "width": 1.2, "height": 1.1, "sill": 0.9},
        {"wall": "B", "type": "door", "offset": 1.2, "width": 0.85, "height": 2.05, "to": "02_kitchen"},
        {"wall": "A", "type": "door", "offset": 3.0, "width": 0.9, "height": 2.1, "to": "04_balcony"},
    ], entry_wall="C")


def flat_mr1() -> FlatSpec:
    r1 = room1()
    kitchen = RoomSpec("02_kitchen", "kitchen", (4.2, 0.3), (2.7, 3.0), 2.72, [], entry_wall="D")
    washroom = RoomSpec("03_washroom", "washroom", (4.2, 3.3), (2.7, 1.8), 2.4, [], entry_wall="C")
    balcony = RoomSpec("04_balcony", "balcony", (0.0, 3.6), (4.2, 1.2), 2.6, [], entry_wall="C")

    # kitchen entrance = the door on 01_room wall B, seen from the kitchen's wall D
    d_rk = r1.openings[2]
    kitchen.openings.append({"wall": "D", "type": "door", "width": d_rk["width"],
                             "height": d_rk["height"], "to": "01_room",
                             "offset": mirror_offset(r1, "B", d_rk["offset"], d_rk["width"],
                                                     kitchen, "D")})
    # kitchen -> washroom through kitchen wall A; washroom entrance on its wall C
    kitchen.openings.append({"wall": "A", "type": "door", "offset": 0.8, "width": 0.75,
                             "height": 2.0, "to": "03_washroom"})
    washroom.openings.append({"wall": "C", "type": "door", "width": 0.75, "height": 2.0,
                              "to": "02_kitchen",
                              "offset": mirror_offset(kitchen, "A", 0.8, 0.75, washroom, "C")})
    # 01_room -> balcony through 01_room wall A; balcony entrance on its wall C
    d_rb = r1.openings[3]
    balcony.openings.append({"wall": "C", "type": "door", "width": d_rb["width"],
                             "height": d_rb["height"], "to": "01_room",
                             "offset": mirror_offset(r1, "A", d_rb["offset"], d_rb["width"],
                                                     balcony, "C")})
    flat = FlatSpec([r1, kitchen, washroom, balcony])
    pairs = check_shared_doors(flat)
    assert len(pairs) == 3, pairs
    # hand-computed expectations (see docstring of floorplan/io/synthetic.py for the convention)
    assert abs(kitchen.openings[0]["offset"] - 1.25) < 1e-9, kitchen.openings[0]
    assert abs(washroom.openings[0]["offset"] - 1.15) < 1e-9, washroom.openings[0]
    assert abs(balcony.openings[0]["offset"] - 0.3) < 1e-9, balcony.openings[0]
    return flat


def flat_r1() -> FlatSpec:
    r1 = room1()
    # alone: the doors to the kitchen/balcony lead nowhere (holes to nothing), drop "to"
    for op in r1.openings:
        op.pop("to", None)
    return FlatSpec([r1])


def folder_size(path: str) -> tuple[int, int]:
    total, n = 0, 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
            n += 1
    return total, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="benchmark/captures")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-mm", type=float, default=8.0)
    args = ap.parse_args()

    jobs = [
        ("synthetic_MR1", flat_mr1(), dict(seed=args.seed, drift=True), None),
        ("synthetic_MR1_nodrift", flat_mr1(), dict(seed=args.seed, drift=False), "synthetic_MR1"),
        ("synthetic_R1_repeat", flat_r1(), dict(seed=args.seed + 101, drift=True), None),
    ]
    results = []
    for name, flat, kw, reuse in jobs:
        out = os.path.join(args.out, name)
        t0 = time.time()
        print(f"== {name} -> {out}")
        gt = generate(flat, out, fps=args.fps, noise_mm=args.noise_mm,
                      reuse_frames_from=os.path.join(args.out, reuse) if reuse else None, **kw)
        results.append((name, out, gt, time.time() - t0))

    # the drift ablation must share the exact true trajectory with MR1
    _, _, p_a = load_stray_csv(os.path.join(args.out, "synthetic_MR1", "traj_gt.csv"))
    _, _, p_b = load_stray_csv(os.path.join(args.out, "synthetic_MR1_nodrift", "traj_gt.csv"))
    assert p_a.shape == p_b.shape and abs(p_a - p_b).max() < 1e-9

    print()
    print(f"{'dataset':24s} {'frames':>7s} {'dur s':>7s} {'path m':>7s} {'size MB':>8s} "
          f"{'files':>6s} {'drift end (cm, deg)':>20s} {'gen s':>6s}")
    grand = 0
    for name, out, gt, dt in results:
        size, nfiles = folder_size(out)
        grand += size
        d = gt["drift"]
        print(f"{name:24s} {gt['n_frames']:7d} {gt['duration_s']:7.1f} {gt['path_length_m']:7.1f} "
              f"{size / 1e6:8.1f} {nfiles:6d} "
              f"{100 * d['end_translation_error_m']:9.1f}, {d['end_yaw_error_deg']:6.2f}  {dt:6.0f}")
    print(f"total on disk (nodrift frames are hard links of MR1): {grand / 1e6:.1f} MB nominal")


if __name__ == "__main__":
    main()
