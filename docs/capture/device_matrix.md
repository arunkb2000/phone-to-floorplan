# Device matrix

Which tier runs on which hardware, and what each tier honestly delivers.

## Hardware

| Device | Photos | Video | LiDAR | Why |
|---|---|---|---|---|
| iPhone 15, 15 Plus, 16, 16e, 16 Plus, 17, Air | yes | yes | **no** | No LiDAR sensor. Stray Scanner will not record depth. |
| iPhone 15 Pro / Pro Max, 16 Pro / Pro Max, 17 Pro / Pro Max | yes | yes | yes | LiDAR, ARKit poses and per-frame intrinsics, all exported by Stray Scanner. |
| iPad Pro with LiDAR (2020 onward) | yes | yes | yes | Same export path. Not benchmarked by me. |
| Anything older, or Android | photos only, at reduced confidence | no | no | The photo tier only needs JPEGs. Without EXIF focal length it falls back to a 26 mm-equivalent prior and widens every interval. |

The pipeline itself runs on the laptop, not the phone. On an Apple Silicon Mac the monocular depth
network uses the Metal backend; on a machine with an NVIDIA GPU it uses CUDA; with neither it runs
on CPU, several times slower but with identical output.

## Accuracy, measured

From `analysis/bench/after/gates.md`. **Read [docs/report/benchmark.md](../report/benchmark.md) first.** These
are not tape measurements: the supplied captures are of a property I cannot enter, so the reference
is a single-frame sensor measurement, and the ceiling-height reference in particular under-reads
because it is usually the visible vertical extent of a wall. A ceiling error of a few centimetres
against it is not, on its own, evidence of a pipeline error.

| Tier | Opening width, mean abs error | Ceiling height, mean abs error | Interval coverage | Cold run |
|---|---|---|---|---|
| LiDAR | 3.9 cm (7 matched) | 12.5 cm (6 matched) | 62 % | 6.5 s, 9745 frames |
| Video | 10.5 cm (2 matched) | 1.4 cm (1 matched) **PASS** | 100 % | 6.6 s |
| Photo | 3.7 cm (2 matched) | 3.9 cm (1 matched) | 100 % | 2.2 s, 18 stills |

Coverage is the fraction of my 90 % intervals that overlap the reference's own 90 % interval. The
stricter reading, which ignores the reference's uncertainty and asks whether its point value falls
inside mine, gives 15 % at the LiDAR tier. Both are printed in the gate tables; the overlap figure is
the fair one when the reference is itself a measurement with a 2 cm spread, and 62 % is still a fail.

The video tier is the only row that passes a headline gate outright: its ceiling height lands within
1.4 cm of the reference. Do not over-read a single matched measurement.

Wall lengths have **no** reference on the supplied data: a single depth frame almost never sees two
opposite walls of these rooms, so the ±1 cm, ±3 % and ±8 % wall gates cannot be scored here at all.
Saying so is more useful than inventing a number.

The interval coverage row is a fail and I am not going to hide it. Two things drive it. The first
is real: a plane fitted to 10⁵ LiDAR returns has a statistical spread in the tenths of a millimetre,
which is not an accuracy claim anyone should make, so absolute sigma floors are asserted per
measurement kind from the sensor's physics (±1 cm class). The second is the reference: with 13
matched measurements against a reference whose own ceiling spread is 16 cm, there is not enough
signal to fit conformal factors without simply fitting to my own noise, so
`floorplan/calibration/factors.json` ships empty and the tier floors do the work. That is the honest state
of the calibration, not a claim that it is calibrated.

### What I can say without any external reference

These need no ground truth and carry the real weight (`analysis/bench/after/gates.md`, `analysis/fixloop/DIFF.md`):

| Property | Result |
|---|---|
| Ceiling agreement between two independent passes, on correctly paired rooms | 0.3 cm, 0.5 cm, 1.7 cm |
| Floor-datum stability across a 215 s capture | 2 cm (was 83 cm before the fix loop) |
| Stitched room overlap | 0.0 to 1.3 % of total room area |
| Same input twice through the pipeline | byte-identical |

## What each tier can and cannot do

**LiDAR.** Metric out of the box. Absolute room placement, so the stitch is a consequence of the
poses rather than an inference. Non-rectangular rooms, alcoves and bays are resolved because the
footprint comes from a contour, not a rectangle. Fails on: glass (no return, so a window reads as a
hole and is recovered from the surrounding wall), mirrors (a phantom room behind the glass, culled
by the wall-plane test), and rooms whose ceiling the walker never looked at, which the output
reports as a prior with a 30 cm interval rather than a measurement.

**Video.** No depth and no poses; a monocular metric-depth network supplies both. Many frames per
room, so the fusion averages down the per-frame noise, which is why the intervals are tighter than
the photo tier. Room boundaries come from the palm-cover markers in the protocol, or from geometry
change detection if the walker skipped them. Rooms are modelled as rectangles.

**Photo.** The floor of the system: two to eight stills per room, no depth, no poses, no timestamps
that mean anything. Extent comes from frames that happen to see two opposite walls at once, so a
room photographed from only one corner has one extent measured and one inferred, and the output says
which. Rooms are rectangles. The whole-property stitch is chained through the doorway photos the
protocol asks for; if those are missing it falls back to folder order with non-overlap enforced, and
drops the placement confidence accordingly. It never refuses input.
