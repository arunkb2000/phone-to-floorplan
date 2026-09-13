# Benchmark: what it is, and what it can and cannot settle

Read this before any number in `analysis/bench/after/gates.md`.

## The data we were given, and the hole in it

The assessment supplied three Stray Scanner captures of one property and no photo or video captures,
and we have no physical access to that property. Two things follow, and neither is negotiable:

**There is no tape ground truth, and there cannot be.** The brief asks for laser or tape measurement
of everything. For a property on the other side of a dataset, that measurement does not exist at any
price. We did not quietly drop the gates that need it. We built the most independent reference the
data allows, stated its limits, and leaned the benchmark's weight onto the gates that need no
external truth at all.

**There are no photo or video captures.** But the gates require the same rooms at all three tiers,
so we cut the photo and video inputs out of the RGB video inside the LiDAR captures
(`scripts/make_rgb_tiers.py`). Those frames are what an iPhone camera did record standing in those
places. Nothing else crosses the boundary: the photo tier receives JPEGs and reads its focal length
from EXIF, the video tier receives an mp4 and reads its focal length from a per-device table. No
depth, no poses, no LiDAR intrinsics. Choosing *which* frames become the six photos per room does
use the LiDAR room labels, because the protocol asks a person to stand in the doorway and in each
corner and something has to decide where those are; that is benchmark construction, not a shortcut
in the tier.

## The reference

`scripts/make_reference_gt.py` measures the property **one depth frame at a time**:

- **Ceiling height** from the floor and ceiling planes fitted in a single frame, or from the vertical
  extent of one wall plane that runs floor to ceiling in that frame.
- **Wall to wall** from two opposite, parallel wall planes seen in the same frame.
- **Opening width** from the gap across one wall plane in that same frame.

It uses the depth image, the intrinsics, and the rotation part of the pose as a gravity direction.
It uses no translation, no pose graph, no drift correction, no multi-frame fusion, no room
segmentation and no `floorplan` geometry module; it implements its own plane fitting in sixty lines.
So it is a fair test of everything the pipeline adds on top of the sensor, which is almost all of the
pipeline. It is **not** independent of the sensor's own range accuracy: a systematic LiDAR bias would
move the reference and the pipeline together and nothing here would see it.

Each reference value is the median across every frame that produced it; the stated uncertainty is the
spread across those frames.

### Where the reference is weak, stated plainly

The ceiling-height reference is the weakest of the three. Floor and ceiling almost never appear in
the same frame, because a portrait-held phone at chest height looks either down or up, not both; on
the supplied captures that variant fired zero times in 427 frames. So most ceiling references come
from the vertical extent of a wall, which **under-reads** whenever furniture, a soffit or the frame
edge cuts the wall short, and it carries a 16 cm spread across frames. A ceiling-height error of a
few centimetres against this reference is therefore not, on its own, proof of a pipeline error. We
report it, we do not explain it away, and we do not use it as the fix-loop target.

## The gates that need no external truth

These carry the weight, and they are the ones the brief describes as the operational meaning of the
system working:

- **Repeatability.** Two passes over the same rooms with disjoint frames
  (`scripts/split_capture.py` interleaves two-second blocks, so both passes visit every room, share
  no frame, and accumulate drift separately). Same rooms in, same plan out, or not.
- **Room overlap and adjacency.** Stitched rooms must not overlap, at any tier.
- **Drift ablation.** The same capture with the correction on and off.
- **Interval calibration.** Whether a 90 % interval contains the reference 90 % of the time.
- **Determinism.** The same input twice through the pipeline is byte-identical.

What the repeatability pair is not: two separate walks. It cannot expose an error that repeats
because the walker always stands in the same place. It can and does expose everything that depends
on which frames arrived, which is what was broken.

## Composition against what the brief asked for

| The brief asks for | What we have | Honest status |
|---|---|---|
| One multi-room capture, 3+ rooms plus a connector | `single_scan_with_ceiling`, 215 s, 100 m walk, 3 resolved spaces plus connecting circulation; `single_scan_floor_only`, 4 spaces | met |
| One furnished room with staged damage, two classes | The supplied captures are of a furnished working office; we could not stage damage in a property we cannot enter | **not met.** The damage path runs and is demonstrated on the captures' own RGB, but no staged two-class ground truth exists |
| Same rooms at all three tiers, multi-room included | LiDAR as supplied; photo and video derived from the same RGB stream | met, with the derivation disclosed above |
| At least one room captured twice at the same tier | Interleaved-frame pair over the same rooms | met in the sense that matters operationally, not by a second physical walk |
| Laser or tape ground truth on everything | Single-frame sensor reference | **not met.** See above |
| Raw sensor data submitted | `data/raw/` as supplied, untouched | met |
