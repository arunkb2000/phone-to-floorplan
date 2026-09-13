# Device matrix

Which tier runs on which hardware, and what each tier honestly delivers.

## Hardware

| Device | Photos | Video | LiDAR | Why |
|---|---|---|---|---|
| iPhone 15, 15 Plus, 16, 16e, 16 Plus, 17, Air | yes | yes | **no** | No LiDAR sensor. Stray Scanner will not record depth. |
| iPhone 15 Pro / Pro Max, 16 Pro / Pro Max, 17 Pro / Pro Max | yes | yes | yes | LiDAR, ARKit poses and per-frame intrinsics, all exported by Stray Scanner. |
| iPad Pro with LiDAR (2020 onward) | yes | yes | yes | Same export path. Not benchmarked by us. |
| Anything older, or Android | photos only, at reduced confidence | no | no | The photo tier only needs JPEGs. Without EXIF focal length it falls back to a 26 mm-equivalent prior and widens every interval. |

The pipeline itself runs on the laptop, not the phone. On an Apple Silicon Mac the monocular depth
network uses the Metal backend; on a machine with an NVIDIA GPU it uses CUDA; with neither it runs
on CPU, several times slower but with identical output.

## Accuracy, measured

Filled from `bench/after/gates.md`. Every figure is against the reference described in
`docs/report/benchmark.md`, which is a single-frame sensor reference, not a tape. Read that section
before quoting these numbers.

| Tier | Wall length | Opening width | Ceiling height | Whole-property stitch | 90 % interval coverage |
|---|---|---|---|---|---|
| LiDAR | see benchmark | see benchmark | see benchmark | absolute, from poses | see benchmark |
| Video | see benchmark | see benchmark | see benchmark | chained along the walk | see benchmark |
| Photo | see benchmark | see benchmark | see benchmark | chained through doorways | see benchmark |

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
