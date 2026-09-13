# Fix declaration

Written against `analysis/bench/before/gates.md`, before the fix was written. Tag `fixloop-before`.

## 1. The single worst-performing gate, with the failing number

**Repeatability, LiDAR tier.** The brief: *two captures of the same room at the same tier agree
within 1 cm or 0.5 % per wall. Same room in, same plan out.*

Two passes over the same rooms, sharing no frames (`scripts/split_capture.py` interleaves two-second
blocks, so both passes visit everything):

| | Pass A | Pass B |
|---|---|---|
| Rooms found | 3 | **2** |

| Paired room | Footprint IoU | Area A | Area B | Area difference | Ceiling difference | Worst wall difference |
|---|---|---|---|---|---|---|
| 01 ↔ 01 | 0.391 | 28.50 m² | 39.99 m² | 40.3 % | **18.6 cm** | **281.1 cm** |
| 02 ↔ 01 | 0.373 | 16.96 m² | 39.99 m² | 135.7 % | **23.5 cm** | **252.4 cm** |
| 03 ↔ 02 | 0.446 | 10.47 m² | 13.94 m² | 33.2 % | 0.3 cm | 49.3 cm |

**The failing number: worst wall difference 281 cm against a 1 cm gate, and the two passes do not
even agree on how many rooms exist.** This is the worst gate in the benchmark by a wide margin, and
it is the one that needs no external ground truth to fail, so there is nowhere to hide.

It also drags two other gates down with it. Ceiling height differs by 18.6 and 23.5 cm between
passes, not because the ceiling plane is hard to fit — where the two passes agree on the room
(IoU 0.45) the ceiling agrees to **0.3 cm** — but because when the room boundary moves, the room mask
swallows a neighbouring space whose real ceiling is 3.07 m instead of 2.26 m and the median moves.

## 2. Root cause, and the evidence

**The room segmentation picks its watershed seeds with a threshold cascade, and the cascade lands on
a different rung for each capture.**

`segment_rooms` in `floorplan/geometry/cloud.py` builds a distance transform `D` of the free space
and looks for room cores as `D > core_r`, trying `core_r` in 1.1, 0.95, 0.8, 0.65, 0.5, 0.4 and
stopping at the first radius that yields at least two cores of usable size. Instrumenting that loop
on the two passes and on the full capture:

| Capture | core_r = 1.1 | 0.95 | 0.8 | 0.65 | Radius selected |
|---|---|---|---|---|---|
| Pass A | 0 kept | 1 kept | 1 kept | **3 kept** | **0.65 m** |
| Pass B | 0 kept | 1 kept | **2 kept** | — | **0.80 m** |
| Full capture | 0 kept | 1 kept | **3 kept** | — | **0.80 m** |

The two passes have nearly identical free space — 49.5 m² and 50.1 m², both with the same 1.25 m
maximum clearance — and still fall off different rungs of the ladder. One centimetre of difference in
where the carved free space ends decides whether a core survives `D > 0.8`, and that decision changes
the number of rooms in the plan and moves room boundaries by metres.

The mechanism is a hard threshold on an absolute clearance, in a pipeline whose free space is built
from wherever the walker happened to point the sensor. The wall planes themselves are stable to
millimetres — that is why the one well-paired room agrees to 0.3 cm on its ceiling. The instability
is entirely in which cells get grouped into which room.

## 3. The fix we intend to ship, and the number we predict

Make the seeds depend on the property's structure rather than on an absolute clearance threshold:

1. **Replace the threshold cascade with h-maxima seeding.** Extract regional maxima of the distance
   transform that stand at least `h = 0.25 m` above the saddle that connects them to any deeper
   maximum, by morphological reconstruction. This is scale-free: there is no ladder to fall off, and
   a centimetre of extra free space cannot create or destroy a seed.
2. **Merge regions that no wall separates.** After the watershed, for every pair of adjacent rooms,
   look at the boundary the watershed drew and ask whether there is wall evidence along it. A
   boundary that is wide and carries no wall is not a doorway, it is an artefact of the free-space
   shape, and the two regions are merged. Walls are the stable signal; free-space shape is not.
3. **Close enclosed unobserved cells** before the transform, so a patch the sensor missed inside a
   room cannot pinch the distance transform and split the room in two.

### Predicted numbers after the fix

| Metric | Before | Predicted after | Gate |
|---|---|---|---|
| Rooms found, pass A vs pass B | 3 vs 2 | **3 vs 3** | must agree |
| Median paired-room footprint IoU | 0.39 | **≥ 0.85** | — |
| Ceiling difference between passes | 18.6 / 23.5 / 0.3 cm | **≤ 1 cm on every paired room** | ≤ 1 cm, PASS |
| Worst wall difference between passes | 281 cm | **≤ 15 cm** | ≤ 1 cm, still FAIL |
| Median wall difference between passes | — | **≤ 3 cm** | — |

**We predict the ceiling-spread gate moves from fail to pass and the per-wall gate does not.** The
honest reason: once the rooms agree, a wall's *length* is still set by where its two perpendicular
neighbours were found, and a short return wall that one pass resolves and the other absorbs into its
neighbour changes a length by tens of centimetres without either answer being wrong about the
physical plane. Getting per-wall agreement to 1 cm needs the polygon topology itself to be stable, not
just the room grouping, and we are calling that out of scope for this loop rather than pretending.
