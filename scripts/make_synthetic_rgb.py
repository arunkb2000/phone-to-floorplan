#!/usr/bin/env python
"""Build the photo-tier and video-tier synthetic captures (same flats as make_synthetic.py).

    uv run python scripts/make_synthetic_rgb.py [--out benchmark/captures] [--fps 10]
                                                [--video-scale 0.5] [--only photo|video]

Writes
  synthetic_photo_MR1/<room>/01..06.jpg + poses.yaml     4 rooms, protocol section 2
  synthetic_photo_R1_repeat/01_room/                     01_room again, other seed / jitter
  synthetic_video_MR1/walk.mp4 + rooms.txt + meta.yaml   protocol section 3, palm covers
  synthetic_video_R1_repeat/walk.mp4                     01_room only
  benchmark/ground_truth/synthetic_damage.yaml           staged damage extents
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from make_synthetic import flat_mr1, flat_r1, folder_size  # noqa: E402

from floorplan.io.synthetic_rgb import render_photo_set, render_video, write_damage_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="benchmark/captures")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--video-scale", type=float, default=0.5,
                    help="render scale for the video (0.5 = 540x960 upscaled to 1080x1920)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["photo", "video"], default=None)
    args = ap.parse_args()

    gt_path = os.path.join(_HERE, "..", "benchmark", "ground_truth", "synthetic_damage.yaml")
    dmg = write_damage_yaml(flat_mr1(), os.path.normpath(gt_path))
    print(f"damage ground truth -> {os.path.normpath(gt_path)}")
    for d in dmg:
        print(f"  {d['room']} {d['surface']:8s} {d['class']:12s} bbox_uv {d['bbox_uv_m']} "
              f"area {d['area_m2']} m2 length {d['length_m']} m")

    rows = []
    if args.only != "video":
        for name, flat, kw in (
                ("synthetic_photo_MR1", flat_mr1(), {"seed": args.seed, "jitter_pos": 0.02, "jitter_yaw_deg": 2.0}),
                ("synthetic_photo_R1_repeat", flat_r1(), {"seed": args.seed + 7, "jitter_pos": 0.03,
                                                          "jitter_yaw_deg": 3.0})):
            out = os.path.join(args.out, name)
            t0 = time.time()
            print(f"== {name} -> {out}")
            poses = render_photo_set(flat, out, jitter_pitch_deg=2.0, **kw)
            n = sum(len(v) for v in poses["rooms"].values())
            rows.append((name, out, f"{n} photos", time.time() - t0))
    if args.only != "photo":
        for name, flat, seed in (("synthetic_video_MR1", flat_mr1(), args.seed),
                                 ("synthetic_video_R1_repeat", flat_r1(), args.seed + 11)):
            out = os.path.join(args.out, name)
            t0 = time.time()
            print(f"== {name} -> {out}")
            meta = render_video(flat, out, seed=seed, fps=args.fps, render_scale=args.video_scale)
            rows.append((name, out, f"{meta['n_frames']} frames, {meta['duration_s']:.0f} s, "
                                    f"{len(meta['segments'])} segments", time.time() - t0))

    print()
    print(f"{'dataset':28s} {'content':40s} {'size MB':>8s} {'files':>6s} {'gen s':>6s}")
    for name, out, content, dt in rows:
        size, nfiles = folder_size(out)
        print(f"{name:28s} {content:40s} {size / 1e6:8.1f} {nfiles:6d} {dt:6.0f}")


if __name__ == "__main__":
    main()
