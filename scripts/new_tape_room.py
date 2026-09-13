"""Scaffold a tape-measured room: folders to drop captures into, and a form to fill in.

    uv run python scripts/new_tape_room.py myroom --openings 2

Creates data/benchmark/captures/myroom_photo/01_<name>/, myroom_video/, a ground-truth form, and
prints the two bench.yaml rows to paste in. Fill the form, drop the files in, then `make tape`.
"""
from __future__ import annotations

import argparse
import os

TEMPLATE = """# Tape ground truth for {name}. Measure in METRES to the nearest 5 mm.
#
# Stand in the doorway you enter through, facing into the room:
#   wall A is ahead, B on your right, C behind you (it holds that door), D on your left.
#
# Leave a line's value at 0.000 and it is ignored, so measure what you can and skip what you cannot.
capture: {name}
method: tape measure, 5 mm resolution

ceiling_height:
  # floor to ceiling at three separate spots; put each one on its own line
  - {{value: 0.000, n_frames: 1, spread_m: 0.0, sigma_m: 0.005}}
  - {{value: 0.000, n_frames: 1, spread_m: 0.0, sigma_m: 0.005}}
  - {{value: 0.000, n_frames: 1, spread_m: 0.0, sigma_m: 0.005}}

wall_to_wall:
  # corner to corner at floor level. A and C are the same wall pair, B and D the other.
  - {{value: 0.000, n_frames: 1, spread_m: 0.0, sigma_m: 0.005}}   # wall A
  - {{value: 0.000, n_frames: 1, spread_m: 0.0, sigma_m: 0.005}}   # wall B
  - {{value: 0.000, n_frames: 1, spread_m: 0.0, sigma_m: 0.005}}   # wall C
  - {{value: 0.000, n_frames: 1, spread_m: 0.0, sigma_m: 0.005}}   # wall D

opening_width:
  # jamb to jamb for every door, frame to frame for every window
{openings}
"""

ROW = """  - name: {name}_photo
    path: data/benchmark/captures/{name}_photo
    tier: photo
    gt: {name}.yaml
    role: tape
  - name: {name}_video
    path: data/benchmark/captures/{name}_video
    tier: video
    gt: {name}.yaml
    role: tape"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="short room name, no spaces, e.g. myroom")
    ap.add_argument("--openings", type=int, default=2, help="how many doors and windows the room has")
    ap.add_argument("--rooms", nargs="*", default=["room"], help="space names in walk order")
    args = ap.parse_args()
    n = args.name

    for i, r in enumerate(args.rooms, start=1):
        d = f"data/benchmark/captures/{n}_photo/{i:02d}_{r}"
        os.makedirs(d, exist_ok=True)
        print(f"  photos  -> {d}/")
    os.makedirs(f"data/benchmark/captures/{n}_video", exist_ok=True)
    print(f"  video   -> data/benchmark/captures/{n}_video/walk.mov")

    ops = "\n".join("  - {value: 0.000, n_frames: 1, spread_m: 0.0, sigma_m: 0.005}"
                    + f"   # opening {i}" for i in range(1, args.openings + 1))
    gt = f"data/benchmark/ground_truth/{n}.yaml"
    if os.path.exists(gt):
        print(f"  form    -> {gt} (already exists, left alone)")
    else:
        with open(gt, "w") as f:
            f.write(TEMPLATE.format(name=n, openings=ops))
        print(f"  form    -> {gt}")

    print("\nPaste these two rows at the end of data/benchmark/bench.yaml:\n")
    print(ROW.format(name=n))
    print("\nThen: make tape")


if __name__ == "__main__":
    main()
