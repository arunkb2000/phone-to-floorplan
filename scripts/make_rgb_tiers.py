"""Derive the photo-tier and video-tier captures from a LiDAR capture's RGB stream.

The assessment supplies three LiDAR captures and no photo or video captures, but the gates require
the same rooms at all three tiers. Those RGB frames are what an iPhone camera did record standing in
those places, so we cut the other two tiers' inputs out of them and hand each tier nothing else:
JPEGs with EXIF, or an mp4. No depth, no poses, no LiDAR intrinsics cross the boundary.

Choosing which frames become the six photos per room does use the LiDAR room labels, because the
protocol asks a person to stand in the doorway and in each corner and something must decide where
those are. That is benchmark construction, not a shortcut inside the tier.

    uv run python scripts/make_rgb_tiers.py data/raw/single_scan_with_ceiling --name flatA
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image

from floorplan.geometry.cloud import build_cloud, build_grid, segment_rooms
from floorplan.io.stray import RGBSource, load_stray_capture

F35 = 26


def exif_jpeg(rgb: np.ndarray, path: str, f35: int = F35, quality: int = 92):
    img = Image.fromarray(rgb)
    ex = Image.Exif()
    ex[0x010F] = "Apple"
    ex[0x0110] = "iPhone"
    sub = {0xA405: int(f35), 0x920A: (int(f35 * 100 * 6.86 / 26), 100), 0xA002: rgb.shape[1], 0xA003: rgb.shape[0]}
    ex[0x8769] = sub
    img.save(path, quality=quality, exif=ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out", default="data/benchmark/captures")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--per-room", type=int, default=6)
    ap.add_argument("--name", default="")
    ap.add_argument("--video-fps", type=float, default=6.0)
    ap.add_argument("--video-max-side", type=int, default=1280)
    args = ap.parse_args()
    base = args.name or os.path.basename(args.capture.rstrip("/"))

    frames = load_stray_capture(args.capture, target_fps=args.fps)
    cloud = build_cloud(frames)
    g = build_grid(cloud)
    lab = segment_rooms(g)
    ci, cj = g.to_cell(cloud.cams[:, 0], cloud.cams[:, 1])
    ok = (ci >= 0) & (ci < lab.shape[0]) & (cj >= 0) & (cj < lab.shape[1])
    room_of = np.zeros(len(frames), np.int32)
    room_of[ok] = lab[ci[ok], cj[ok]]

    rgb = RGBSource(args.capture)
    photo_dir = os.path.join(args.out, f"photo_{base}")
    order = []
    for r in range(1, int(lab.max()) + 1):
        idx = np.where(room_of == r)[0]
        if len(idx) < args.per_room:
            continue
        order.append(r)
    for slot, r in enumerate(order, start=1):
        idx = np.where(room_of == r)[0]
        xy = cloud.cams[idx, :2]
        lo, hi = xy.min(0), xy.max(0)
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]], [hi[0], lo[1]]])
        picks = [int(idx[0])]
        for c in corners:
            j = int(idx[np.argmin(np.linalg.norm(xy - c, axis=1))])
            if j not in picks:
                picks.append(j)
        if len(picks) < args.per_room:
            picks.append(int(idx[-1]))
        picks = picks[:args.per_room]
        d = os.path.join(photo_dir, f"{slot:02d}_space")
        os.makedirs(d, exist_ok=True)
        for k, j in enumerate(picks, start=1):
            img = rgb.get(frames[j].frame_index, max_side=2016)
            if img is None:
                continue
            exif_jpeg(img, os.path.join(d, f"{k:02d}.jpg"))
        print(f"  photo room {slot:02d}: {len(picks)} stills")

    vid_dir = os.path.join(args.out, f"video_{base}")
    os.makedirs(vid_dir, exist_ok=True)
    src = cv2.VideoCapture(os.path.join(args.capture, "rgb.mp4"))
    sfps = src.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(sfps / args.video_fps)))
    int(src.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w0, h0 = int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, args.video_max_side / max(w0, h0))
    W, H = int(w0 * s) // 2 * 2, int(h0 * s) // 2 * 2
    out = cv2.VideoWriter(os.path.join(vid_dir, "walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (W, H))
    # a palm cover wherever the walker crossed from one room to another, as the protocol asks
    {f.frame_index: i for i, f in enumerate(frames)}
    transit = set()
    prev = 0
    for i, f in enumerate(frames):
        r = room_of[i]
        if r and prev and r != prev:
            transit.add(f.frame_index)
        if r:
            prev = r
    black = np.zeros((H, W, 3), np.uint8)
    n_written = n_cuts = 0
    i = 0
    while True:
        okr, bgr = src.read()
        if not okr:
            break
        if i % step == 0:
            if i in transit:
                for _ in range(int(args.video_fps * 2)):
                    out.write(black)
                n_cuts += 1
            out.write(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA))
            n_written += 1
        i += 1
    src.release()
    out.release()
    print(f"  video: {n_written} frames at {args.video_fps} fps, {n_cuts} palm-cover cuts -> {vid_dir}/walk.mp4")


if __name__ == "__main__":
    main()
