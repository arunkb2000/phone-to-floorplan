# phone-to-floorplan — technical report

An iPhone capture in, a dimensioned whole-property floor plan out, with damage regions, concealed
damage flags, scope line items and a calibrated confidence interval on every number. One command per
capture, three input tiers, nothing calls our infrastructure.

---

## 1. Architecture

```
capture folder ─► io/           tier detection; loaders normalise everything to Frame
                                {rgb, K, depth?, conf?, pose?, t}
                     │
        ┌────────────┴────────────┬──────────────────────────┐
        ▼                         ▼                          ▼
   tiers/lidar.py            tiers/video.py             tiers/photo.py
   depth + poses             rgb only                   rgb only
        │                         └────────┬─────────────────┘
        │                                  ▼
        │                          tiers/mono.py
        │                          monocular metric depth per frame,
        │                          per-frame gravity + Manhattan,
        │                          rectangle fusion, chain stitch
        ▼
   geometry/drift.py   floor-anchored levelling, Manhattan yaw anchoring, loop closure
   geometry/cloud.py   one gravity-aligned cloud → ray-carved free space → watershed rooms
   geometry/layout.py  contour → rectilinear → edges snapped to measured wall planes → openings
        │
        └──────────────► core/room.py  RoomOut: polygon, walls, openings, ceiling, area
                              │
        damage/ ──────────────┤  open-vocabulary detection → masks → projection onto surface planes
        scope/  ──────────────┤  concealed-damage rules, scope catalogue keyed to surfaces
        calib/  ──────────────┤  split-conformal interval factors, per tier and per measurement kind
        render/ ──────────────┤  SVG + PNG plan
                              ▼
                    geometry/assemble.py → plan.json (schema 0.1.0)
```

One command: `floorplan run <capture> --out <dir>`. The tier is detected from the folder contents; a
Stray Scanner export gives LiDAR, a video file gives video, sub-folders of stills give photo.

### The bug that mattered most

Stray Scanner's `odometry.csv` documents an ARKit camera-to-world transform, and ARKit's camera axes
are OpenGL-style, so the textbook conversion to OpenCV is `diag(1, -1, -1)`. On the supplied captures
that conversion is wrong: the exported rotation is already in OpenCV axes. With the textbook flip the
vertical-normal points smear over three metres of height and nothing downstream works. Without it
they collapse into a single two-centimetre bin 1.47 m below the ARKit origin, which is a phone's
carry height. `scripts/check_convention.py` prints the comparison; it is four lines of table and it
is the difference between a working pipeline and a broken one.

| World map | Camera axes | Vertical-normal points | Sharpest 3 cm height bin | Floor z |
|---|---|---|---|---|
| ARKit y-up → z-up | OpenGL → OpenCV | 6.7 % | 0.103 | +0.09 m |
| ARKit y-up → z-up | **as exported** | **23.2 %** | **0.417** | **−1.47 m** |
| identity | OpenGL → OpenCV | 5.0 % | 0.035 | +2.76 m |
| identity | as exported | 32.0 % | 0.020 | +2.67 m |

---

## 2. Tier design and device matrix

Full matrix in `docs/capture/device_matrix.md`. In one paragraph each:

**LiDAR** (Pro devices). Every frame is back-projected with its own intrinsics and pose into one
gravity-aligned cloud. Free space is carved by 2-D ray casting — every return says the space in front
of it was empty — which is what opens a doorway the walker never stepped through. Rooms are a
watershed of that free space, so doorway necks become room boundaries. Each room's contour is forced
rectilinear in the Manhattan frame and then every edge is moved onto the wall plane the sensor
actually measured, which is what turns a 5 cm grid into a centimetre-level wall length. Rooms are
polygons, not rectangles, so alcoves survive.

**Video** (any iPhone 15+). No depth, no poses. A monocular metric-depth network supplies depth per
frame; gravity comes from the floor and ceiling planes in that frame and the Manhattan frame from its
wall normals. Rooms are cut at the protocol's palm-cover markers, or by geometry change detection if
the walker skipped them. Many frames per room, so the fusion averages down per-frame noise.

**Photo** (any iPhone 15+). The floor of the system: two to eight stills per room. Same core as
video with far fewer votes. A room's extent is measured only where a frame saw two opposite walls at
once; otherwise it is inferred and the output says so. The whole-property stitch chains rooms through
the doorway photos the protocol asks for, laying each room's entrance door against the previous
room's exit door and penalising overlap. Without those photos it falls back to folder order with
non-overlap enforced and drops the placement confidence. It never refuses input.

---

## 3. Drift handling

`--drift-correction off` is the ablation; it is never the shipped default. Three corrections, in
`geometry/drift.py`:

1. **Floor-anchored levelling and height datum.** Every frame that sees the floor says where z = 0
   is and which way is up. The running-median residual is removed. This kills the slow tilt that
   otherwise smears a 2.3 m ceiling across ten centimetres.
2. **Manhattan yaw anchoring.** Wall normals must agree with the property's dominant axes; the
   smoothed per-frame yaw residual is removed, so a long corridor stops bending.
3. **Loop closure.** When the walk ends within 1.5 m of where it started, as the protocol asks, the
   residual gap is distributed along the trajectory.

The supplied captures drift by 17 cm over a 54 m walk and 39 cm over a 100 m walk before correction.
The ablation table is in `analysis/bench/after/gates.md`.

---

## 4. Error budget

| Source | Scale | Where it is handled |
|---|---|---|
| LiDAR range noise | ~1 cm at 2 m, growing with range and incidence | confidence filter (high only), 5 m range cap, plane fitting over 10⁴–10⁵ points |
| ARKit pose drift | 17–39 cm end-to-end over the supplied walks | the three corrections above |
| Plane fit | sub-millimetre on 10⁵ points; the fit is not the limit | propagated into every wall's sigma |
| Room segmentation | **was the dominant error**; see the fix loop | the fix in §6 |
| Grid quantisation | 5 cm | removed by snapping edges to measured planes |
| Monocular depth scale | median ratio to LiDAR 0.95–1.06, but per-frame spread 0.34 to 3.13 | per-tier interval floor; the honest cap on the photo and video tiers |
| Occlusion | a wall behind a wardrobe is not measured | `coverage_fraction` per room, and `source: inferred` per wall |

The monocular row is the one to read twice. Averaged over a capture the network's scale is nearly
right; frame by frame it is not, and no amount of averaging fixes a scale that moves. That is why the
photo and video tiers carry a relative interval floor that no statistical argument is allowed to
undercut.

---

## 5. Calibration

Every measurement leaves the geometry with a propagated sigma that is honest about the error sources
we modelled and silent about the ones we did not. We correct that with split conformal prediction: on
a calibration split we compute `r = |error| / (1.645 · sigma)` for each measurement kind and take the
quantile of `r` that puts 90 % of calibration residuals inside. That scalar, per tier and per kind,
multiplies the sigma at output time (`floorplan/calibration/factors.json`). Coverage before and after is in
the gate tables. A tier floor sits underneath, so a thin-input tier can never claim LiDAR precision
no matter what the statistics say.

---

## 6. The fix loop

**Declared gate.** Repeatability at the LiDAR tier. Two passes over the same rooms sharing no frames
found 3 rooms and 2 rooms, paired footprints overlapped at IoU 0.41, and the worst wall disagreed by
281 cm against a 1 cm gate.

**Root cause, with evidence.** The watershed seeded rooms from a ladder of absolute clearance
thresholds, stopping at the first rung that yielded two cores. The two passes had near-identical free
space, 49.5 and 50.1 m², and still fell off different rungs, one at 0.65 m and one at 0.80 m. A
centimetre of carved free space decided how many rooms existed. Where the two passes did agree on a
room, its ceiling agreed to 0.3 cm, which is what said the fault was in the grouping and not in the
plane fitting.

**Shipped.** h-maxima seeding of the distance transform, which is scale-free and has no ladder to
fall off; a merge of regions that no wall separates, because walls are stable and free-space shape is
not; and hole filling so an unobserved patch cannot pinch the transform in two. The pre-fix behaviour
stays reachable as `--legacy`, which is how the before-run regenerates from shipped code.

**Result, and the second bug.** Ceiling agreement between passes went from up to 120.6 cm to 0.3, 0.5
and 1.7 cm on the correctly paired rooms; median IoU from 0.41 to 0.73. Shipping it made the ceiling
numbers legible for the first time and they were still wrong by a constant 30 cm, which exposed a
second defect in the same family: the per-frame height datum took the *modal* up-facing surface,
which in a furnished office is a desk at 0.75 m. The floor is now the lowest well-supported
horizontal surface a plausible carry height below the camera. Datum range across a capture: 0.83 m to
0.02 m.

**What we got wrong.** We predicted the passes would agree on room count; they went from 3 versus 2
to 4 versus 6. Both now produce seven candidate regions, so the seeding is stable and the
disagreement moved into the handling of regions too small to be rooms. The worst-case wall row also
regressed, 62 cm to 120 cm, because more resolved rooms means more short walls to disagree about.
Full accounting in `analysis/fixloop/POSTMORTEM.md`; runs regenerate with `make before` and
`make after`.

---

## 7. Known failure modes

**Mirrors.** LiDAR returns a phantom room behind the glass. Points that fall beyond an already
established wall plane are culled, and the room carries `mirror_suspected`. A large mirror on a wall
the pipeline has not yet established can still produce a phantom alcove.

**Glass.** Windows return nothing. A window therefore reads as a hole in the wall with no material
behind it, which is exactly how the opening detector finds it: material below the gap and none
through it. A full-height glass door is the ambiguous case and is classified by what lies beyond it.

**Wet-look and gloss floors.** Specular returns thin out the floor points. The floor plane is fitted
over the whole property, so a patch of gloss costs coverage, not accuracy.

**Low light.** The LiDAR tier is unaffected: it is an active sensor. The photo and video tiers score
each frame for exposure and blur and widen every interval in that room by half again, and say so in
the room's warnings.

**Ceilings that were never looked at.** `single_scan_floor_only` and `single_room` never point up.
The pipeline reports a 2.70 m prior with a ±49 cm interval and a warning, rather than a number that
looks like a measurement. This is the behaviour the brief's "intervals widen honestly" line asks for,
and it is worth more than a confident guess.

**Open-plan space.** The watershed needs a neck to cut at. A 40 m² space with no doorway between its
halves stays one room, correctly, but a homeowner might name it two.

**Non-Manhattan geometry.** Rooms are forced rectilinear in a single dominant frame. A genuinely
angled wall is approximated by steps and its `coverage_fraction` drops, which is the signal to
distrust it.

---

## 8. Disclosure

Pretrained models, all run locally, weights fetched by `scripts/fetch_weights.sh` with pinned
revisions and a SHA-256 manifest:

- **Depth Anything V2, metric indoor** (small and large), for the photo and video tiers.
- **OWLv2 base**, open-vocabulary detection, for damage regions and for vetoing phantom openings.

No API is called at run time. No data leaves the machine. The capture app is Stray Scanner, free on
the App Store; the photo and video tiers use the stock Camera app.
