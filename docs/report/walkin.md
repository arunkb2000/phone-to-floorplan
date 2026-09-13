# Walk-in readiness

At the defense you capture a space we have never seen, choose the tier on the day, and we run cold.
This page is what we checked.

## All three tiers run from one command

```bash
floorplan run <your capture folder> --out results
```

The tier is detected from the folder contents. Nothing else changes between tiers, so there is no
"which script do I run" moment in front of you.

## Timings, measured on this machine

Apple M5, 16 GB, Metal backend. From `bench/after/gates.md`.

| Capture | Tier | Input size | Wall clock |
|---|---|---|---|
| `single_scan_with_ceiling` | LiDAR | 9745 frames, 215 s walk, 530 MB | 6.3 s |
| `single_scan_floor_only` | LiDAR | 5251 frames, 115 s walk, 289 MB | 3.5 s |
| `single_room` | LiDAR | 1715 frames, 37 s walk, 93 MB | 1.0 s |
| `photo_flatA` | photo | 18 stills, 3 rooms | 1.0 s |
| `video_flatA` | video | 1218 frames at 6 fps | 4.5 s |

On a CPU-only machine the two monocular tiers are several times slower because of the depth network;
the LiDAR tier is unaffected because it runs no network at all.

## Offline

After `make setup` and `make weights`, nothing reaches the network. Model weights load from
`weights/`; there is no telemetry and no API call. You can pull the ethernet cable.

## Determinism, and the cache

Model outputs are cached under `cache/<hash of the input pixels>/` and replay exactly. The live path
is the default for anything not already cached, which is what a capture we have never seen will be,
so the walk-in run is a live run whether or not the cache exists. `LIVE=1 make reproduce` clears the
cache and forces the live path for every number in the submission.

Seeded RANSAC, sorted file iteration and fixed histogram binning mean the same input gives the same
output byte for byte.

## What will happen on a capture we have never seen

Honest expectations, not a sales pitch:

- **LiDAR tier.** This is the strong one. Room polygons come from measured wall planes; ceiling
  heights land within a couple of centimetres when you look at the ceiling, and are reported as a
  2.70 m prior with a ±49 cm interval and a warning when you do not.
- **Video tier.** Rooms are cut at the palm-cover markers. Skip them and it falls back to change
  detection, which is less reliable and says so.
- **Photo tier.** The floor of the system. Rooms are rectangles. Extents are only measured where a
  photo saw two opposite walls, so follow the corner-to-corner instruction or you will get inferred
  extents with wide intervals, correctly labelled.
- **Everything.** No capture makes the pipeline refuse or crash. Missing evidence becomes a wider
  interval, a `source: inferred` or `prior`, and a warning in the room, not a confident number.

## What we would still be nervous about

A property with strong non-Manhattan geometry, a large mirror on a wall the pipeline has not yet
established, or a capture where the walker never looks at a ceiling in any room. All three are in
the known-failure-modes section of the technical report, and all three degrade to a wide interval
rather than a wrong answer, but they will cost accuracy.
