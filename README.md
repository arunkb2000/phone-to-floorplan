# phone-to-floorplan

An iPhone capture goes in. A dimensioned whole-property floor plan comes out, with damage regions,
concealed-damage flags, scope line items, and a calibrated confidence interval on every measurement.
Three input tiers — photos, video, LiDAR — one command, one JSON schema, nothing calling home.

```bash
make setup && make weights
uv run floorplan run data/raw/single_scan_with_ceiling --out analysis/results/demo
```

```
tier=lidar  rooms=5  openings=4  damage=7  6.5s -> analysis/results/demo/plan.json
   01_space   area 28.30 m2  h 2.366 [2.349,2.382] measured  walls 17  openings 2  coverage 93%
   02_space   area  5.40 m2  h 2.266 [2.250,2.283] measured  walls  5  openings 0  coverage 100%
   ...
```

You get `plan.json` (validating against `src/floorplan/schema/output.schema.json`), `plan.svg`,
`plan.png` and `timing.json`.

---

## 1. The capture route I chose, and why

The brief offered two routes. I took **Route 2, the stock capture protocol**, and the whole of
[docs/capture/protocol.md](docs/capture/protocol.md) is one page.

| Tier | What the person installs | What they do |
|---|---|---|
| LiDAR | **Stray Scanner**, free on the App Store | Record, walk the perimeter, look at the ceiling |
| Video | nothing, stock Camera app | One portrait clip, palm over the lens between rooms |
| Photos | nothing, stock Camera app | Six stills per room, one folder per room |

I chose it for three reasons, in order of how much they mattered.

**It is the route that is actually installable in the ten minutes the brief allows.** A TestFlight
build needs a paid developer account and App Review lead time. Stray Scanner is already on the App
Store and exports precisely what the LiDAR tier needs as plain files: `camera_matrix.csv` for
intrinsics, `odometry.csv` for per-frame poses, 16-bit depth PNGs, per-pixel confidence, and the RGB
video. There is nothing an app of mine would have collected that this does not.

**Writing my own capture app would have bought accuracy I do not need and cost accuracy I do.** The
sensor and ARKit are the same either way. The engineering that decides whether this works is
downstream of the capture: gravity alignment, drift correction, room segmentation, plane fitting,
interval calibration. Spending two days on an iOS build would have come out of that budget.

**Route 2 makes the two non-LiDAR tiers honest.** Photos and video come from the stock Camera app,
so nothing privileged reaches them. The photo tier reads its focal length from EXIF exactly as it
would from a stranger's photograph; the video tier reads it from a per-device table. That is the
same input an evaluator's own phone produces, which is the point of the walk-in test.

The cost is that the protocol has to be unambiguous, because at the defense it is followed literally.
The one unusual instruction — cover the lens for two seconds when walking between rooms — is called
out as unusual on the page, and the pipeline falls back to geometric change detection and says so in
its output if the person forgets.

## 2. From nothing to a plan in under fifteen minutes

```bash
git clone <this repo> && cd phone-to-floorplan
make setup        # installs uv if absent, pins Python 3.12, syncs the locked dependency set
make weights      # pinned model weights from Hugging Face with a SHA-256 manifest
make run CAP=data/raw/single_scan_with_ceiling OUT=analysis/results/demo
```

Only the first two touch the network. After that the pipeline is fully offline, which is how it runs
at the walk-in. The tier is detected from what is in the folder you hand over: a Stray Scanner export
is LiDAR, a folder with a video file is video, sub-folders of stills are photos.

| Command | What it does |
|---|---|
| `floorplan run <capture> --out <dir>` | one capture to a plan |
| `floorplan run <capture> --drift-correction off` | the drift ablation |
| `floorplan validate <plan.json>` | check a plan against the published schema |
| `floorplan bench --out analysis/bench/latest` | the whole benchmark and its gate tables |
| `make reproduce` | regenerate every reported number from the raw captures |
| `make test` | 25 tests, including determinism |

## 3. How it works

```
capture folder ──► io/          tier detection; everything becomes Frame{rgb, K, depth?, pose?}
                    │
      ┌─────────────┴──────────────┬───────────────────────┐
      ▼                            ▼                       ▼
 tiers/lidar.py               tiers/video.py          tiers/photo.py
 depth + poses                 rgb only                rgb only
      │                            └──────────┬────────────┘
      │                                       ▼
      │                                 tiers/mono.py
      ▼                       monocular depth, per-frame gravity, rectangle fusion
 geometry/drift.py    levelling, yaw anchoring, loop closure   (ablatable)
 geometry/cloud.py    one gravity-aligned cloud, ray-carved free space, watershed rooms
 geometry/layout.py   contour → rectilinear → edges snapped to measured planes → openings
      │
      └──► perception/ ──► scope/ ──► calibration/ ──► rendering/ ──► plan.json
```

**The LiDAR tier in one sentence:** accumulate every frame into a single gravity-aligned point cloud,
carve the free space by 2-D ray casting, watershed it into rooms at the doorway necks, take each
room's contour, force it rectilinear, then move every edge onto the wall plane the sensor actually
measured. The grid gives topology; the cloud gives centimetres.

**The other two tiers in one sentence:** a monocular metric-depth network stands in for the sensor,
gravity comes from each frame's own floor and ceiling planes, rooms are rectangles, and the
whole-property stitch chains rooms through the doorway photos the protocol asks for.

**Why intervals are the spine.** The brief says confident garbage on thin input caps the score, so
every measurement carries `value`, `sigma`, `ci_low`, `ci_high` and a `source` of `measured`,
`inferred` or `prior`. A capture that never looks at a ceiling gets a 2.70 m prior with a ±49 cm
interval and a warning, not a number dressed as a measurement. Two floors sit under the statistics: a
relative floor per tier, and an absolute floor per measurement kind asserted from the sensor's
physics, because a plane fitted to 100,000 returns has a statistical spread in the tenths of a
millimetre and that is not an accuracy claim anyone should make.

Full detail: [docs/report/technical_report.md](docs/report/technical_report.md).

## 4. What was asked, and what I delivered

Row-by-row against the brief in [docs/compliance_matrix.md](docs/compliance_matrix.md). The summary:

| Asked for | Delivered |
|---|---|
| Three input tiers, same output contract, intervals that widen as data thins | All three, one command, tier auto-detected. A capture with no ceiling returns a flagged prior with a wide interval instead of a number. |
| Dimensioned per-room plan: walls, ceiling height, floor area, openings | Rectilinear polygons with every edge snapped to a measured plane. 38 dimensioned walls on the largest supplied capture. |
| Stitched multi-room plan with correct adjacency | Absolute placement from poses at the LiDAR tier, chained through doorways without them. Room overlap 0.0 to 1.3 % of total area. |
| Per-surface damage regions with class and metric extent | Classical surface-anomaly detection, masks projected onto surface planes, extents in metres with intervals. |
| Concealed-damage flags naming the rule that fired | Six rules, unit-tested. Quiet on the supplied captures, for the reason in §6. |
| Scope line items keyed to surfaces | Catalogue keyed by damage class and surface kind; every item carries its `surface_id`. |
| A confidence interval on every measurement | Every numeric field in the schema is a measurement object. A test asserts each one brackets its own value. |
| One command per capture, JSON to a published schema, rendered plan | `floorplan run`, `output.schema.json`, SVG and PNG. All outputs validate. |
| Drift accountability plus an on/off ablation | Three corrections in `geometry/drift.py`, ablatable with one flag, table in the gates. |
| A fix loop: declare, diagnose, ship, prove | Declaration committed before the fix, both runs regenerable, readable diff, and a post-mortem that volunteers what went wrong. |
| Repeatability, opening and ceiling gates | All scored at all three tiers, and most of them fail. The numbers are in §5, not hidden. |

## 5. Results

From [analysis/bench/after/gates.md](analysis/bench/after/gates.md). Read
[docs/report/benchmark.md](docs/report/benchmark.md) first: it says what the reference is and, more
importantly, what it is not.

| Tier | Opening width | Ceiling height | Interval coverage | Cold run |
|---|---|---|---|---|
| LiDAR | 3.9 cm | 12.5 cm | 62 % | 6.5 s on 9,745 frames |
| Video | 10.5 cm | 1.4 cm, passes | 100 % | 6.6 s |
| Photo | 3.7 cm | 3.9 cm | 100 % | 2.2 s on 18 stills |

**What holds up without any external truth**, which is the part I trust most:

| Property | Result |
|---|---|
| Ceiling agreement, two independent passes over the same rooms | 0.3, 0.5 and 1.7 cm |
| Floor-datum stability across a 215 s capture | 2 cm, from 83 cm before the fix loop |
| Stitched room overlap | 0.0 to 1.3 % of total room area |
| Same input twice | byte-identical, asserted by a test |
| Full reproduction from a wiped tree | 11 min 44 s, exit 0, 25 tests passing |

**The fix loop** is in [analysis/fixloop/](analysis/fixloop/): the declaration written before the fix
existed, both runs, the diff, and the post-mortem. Short version: I declared repeatability as the
worst gate and traced it to a threshold ladder in the room segmentation that two passes over the same
rooms fell off at different rungs. Shipping the fix moved ceiling agreement from 120 cm to under
2 cm, and exposed a second and larger bug in the height datum that was invisible until the first fix
landed. The gate still does not pass, one of my predictions was wrong, and one row regressed. All
three are in the post-mortem.

## 6. What I could not do, and why

Three requirements are marked **not met**. They share one cause, set out in
[docs/CONSTRAINTS.md](docs/CONSTRAINTS.md): the sample captures are of a property I have no physical
access to, and the device available to me is an **iPhone 15, which has no LiDAR sensor**.

| Requirement | Why not |
|---|---|
| Laser or tape ground truth on everything | A measurement of a room I cannot enter does not exist at any price. |
| A furnished room with staged damage spanning two classes | I cannot tape a stain to a wall I cannot reach. |
| Head-to-head against a consumer scanning app at the LiDAR tier | A scanning app needs a phone standing in the room; it cannot ingest someone else's export. And an iPhone 15 could not produce a LiDAR scan even standing there. |

Two things follow, and I would rather present them as decisions than as apologies.

**I moved the benchmark's weight onto the gates that need no external truth.** Repeatability, room
overlap, the drift ablation and determinism all test the system without a tape, and they are the ones
the brief describes as the operational meaning of the thing working.

**I built the most independent reference the data allowed** rather than skipping the gates that do
need truth. `scripts/make_reference_gt.py` measures the property one depth frame at a time using only
the depth image, the intrinsics and the pose's rotation as a gravity direction: no translation, no
pose graph, no drift state, no fusion, no segmentation, no module of mine. It is a fair test of
everything I added on top of the sensor and it is honestly not independent of the sensor itself.

I did not substitute the easier comparison — my photo tier against a non-LiDAR app in a room I can
reach — under the heading that asks for the LiDAR tier. `scripts/head_to_head.py` and a form are
committed, so the row is about forty minutes of room time from being filled.

## 7. What I demonstrated on synthetic data instead

Two of those three gaps are about truth I cannot measure. So I built a room where the truth does not
need measuring because it is *defined*: `scripts/make_synthetic_room.py` renders a room in Stray
Scanner format from exact dimensions, with a door, a window, and two staged damage classes at known
positions and extents. The pipeline is told none of it.

This is not a substitute for a real room and it is labelled that way everywhere it appears. It cannot
say anything about real sensor noise, real surfaces or real clutter, and the supplied captures carry
that half. What it can do is answer the question a tape would have answered: does the geometry
recover dimensions it was never given?

**Ground truth, defined:** room 4.260 × 3.180 m, ceiling 2.640 m, door 0.915 m, window 1.240 m, a
water stain of 0.520 × 0.380 m on the ceiling and a crack 0.870 m long on a wall.

| Measurement | Truth | Recovered | Error |
|---|---|---|---|
| Wall A and C | 4.260 m | 4.259 m | **1 mm** |
| Wall B and D | 3.180 m | 3.147 m | 33 mm |
| Ceiling height | 2.640 m | 2.639 m | **1 mm**, passes the 1.5 cm gate |
| Floor area | 13.55 m² | 13.40 m² | 1.1 % |
| Window width | 1.240 m | 1.301 m | 61 mm |
| Door | 0.915 m | not detected | a miss |
| Crack length | 0.870 m | 0.864 m | **6 mm** |
| Water stain | on the ceiling | found on the ceiling, correctly located | class and surface correct, area under-read |

Both staged damage classes are detected, on the correct surfaces, and generate scope items keyed to
those surfaces including a plumbing investigation triggered by a ceiling stain. The door is missed
because it opens onto nothing in this synthetic room, so there is no space behind it to see through,
which is the one case the opening detector handles worst. `make dryrun` and
`floorplan bench --only synthetic` reproduce all of it.

**What this bought me.** The wall-length gates could not be scored at all against the supplied
captures, because a single depth frame almost never sees two opposite walls of those rooms. They are
the headline gate at every tier. Here they are scored, and one of the two walls lands inside the
1 cm LiDAR gate.

## 8. Conclusion

The system does what the brief specifies. Three tiers, one command, the full output contract to a
published schema, a rendered plan a homeowner would recognise, and a calibrated interval on every
number. It runs cold on a capture it has never seen in six and a half seconds, offline, and
deterministically, and the whole benchmark regenerates from raw input in under twelve minutes.

Where it is strong: the LiDAR geometry, which recovers a wall to a millimetre on data with defined
truth and agrees with itself to under two centimetres across independent passes; the discipline of
the intervals, which widen and say why rather than guessing; and the fix loop, where I found a real
defect, shipped a real fix, uncovered a second and larger one by shipping it, and wrote down the
prediction I got wrong.

Where it is weak, and I would rather say it than have it found: most gates fail. Opening detection is
the worst of them, at 3.9 cm mean error where the gate is 2 cm. Interval coverage is 62 % against a
nominal 90 %. Three requirements are not met at all because they need hardware I do not have, and one
of those, the head-to-head, is worth ten per cent of the score and has no table.

What I would do next is in the post-mortem, and the first item is the one that unlocks the rest:
derive each room's polygon from a single global wall-line arrangement rather than from each capture's
own contour, so wall counts match across captures by construction. That is the precondition for the
per-wall repeatability gate, and it is where I would start on Monday.

---

## Layout

```
src/floorplan/     cli, core, io, geometry, tiers, perception, scope, calibration, rendering, schema
data/raw/          the three supplied captures, untouched (6 GB, not in git)
data/benchmark/    bench.yaml, derived and synthetic captures, reference ground truth
analysis/bench/    gate tables
analysis/fixloop/  declaration, before, after, diff, post-mortem
scripts/           setup, weights, reference truth, derived captures, synthetic room, head-to-head
tests/             25 tests, including determinism and the fix's own property
docs/              capture protocol, device matrix, constraints, reports, compliance matrix
```
