# Hardware constraints, and the plan to close them

Three rows of the compliance matrix are marked **not met**. They share one cause, and it is not
time, skill or budget. It is that two pieces of hardware were not available:

1. **Physical access to the captured property.** The assessment supplied three Stray Scanner
   captures of an office. I have no idea where that office is and cannot walk into it.
2. **A LiDAR-class iPhone.** The device available is an **iPhone 15** (non-Pro). It has no LiDAR
   sensor, so it cannot produce a depth-and-pose capture at all, and no consumer scanning app can
   produce a LiDAR-tier scan on it either.

Everything the pipeline needed from a sensor, I took from the captures I was given. Everything
that needed a tape or a second scanner in a room, I could not do, and I said so rather than
substituting something easier under the same heading.

---

## What each constraint blocks

| Requirement | Blocked by | Why no workaround exists |
|---|---|---|
| Laser or tape ground truth on everything | no access to the property | A measurement of a room you cannot enter does not exist at any price. |
| One furnished room with staged damage spanning two classes | no access to the property | You cannot tape a stain decal to a wall you cannot reach. |
| Head-to-head against a consumer scanning app at the LiDAR tier | both | An app needs a phone *in the room*; it cannot ingest someone else's export. And an iPhone 15 could not produce the LiDAR scan even standing there. |

Two things follow that are worth saying out loud, because they are engineering decisions rather than
apologies:

**I moved the benchmark's weight onto the gates that need no external truth.** Repeatability, room
overlap, the drift ablation and determinism all test the system without a tape, and they are the ones
the brief calls the operational meaning of the thing working: same room in, same plan out.

**I built the most independent reference the data allowed** rather than skipping the gates that do
need truth. `scripts/make_reference_gt.py` measures the property one depth frame at a time using only
the depth image, the intrinsics and the pose's rotation as a gravity direction: no translation, no
pose graph, no drift state, no fusion, no segmentation, no pipeline module. It is a fair test of
everything the pipeline adds on top of the sensor, and it is honestly not independent of the sensor
itself. Its limits are stated in `docs/report/benchmark.md` and I do not quote it as if it were a
tape.

---

## Plan A: one borrowed Pro device, about 90 minutes

This closes all three rows. Borrow any **iPhone 15 Pro, 16 Pro or 17 Pro** (or an iPad Pro with
LiDAR) and pick **two rooms you can physically stand in** — your own room and kitchen are fine, and
the brief does not require the benchmark property.

### Before you start, 10 minutes
- Install **Stray Scanner** (free) and **magicplan** (free tier: two full-feature projects) on the
  borrowed device.
- Have a tape measure, masking tape, and two printed sheets: one with a brown tea-coloured blotch
  (the "water stain"), one with a jagged dark line (the "crack").

### Step 1, stage the damage, 10 minutes
Tape the two sheets in room one: the stain on the ceiling or a high wall, the crack on a different
wall. Measure each with the tape: width, height, distance from the floor, distance from the nearest
corner. Write them into `data/benchmark/ground_truth/` following `TAPE_TEMPLATE.yaml`.

*Closes: staged damage spanning two classes.*

### Step 2, tape the rooms, 20 minutes
For each of the two rooms, measure and record: all four wall lengths corner to corner at floor level,
ceiling height at three separate spots, and for every door and window the width jamb to jamb, the
height, and the distance from the left corner of its wall. Photograph your hand sketch.

Fill a copy of `data/benchmark/ground_truth/TAPE_TEMPLATE.yaml` per room and add a row to
`data/benchmark/bench.yaml` pointing at it.

*Closes: tape ground truth, and it converts every currently-unscorable wall-length gate into a
scored one at all three tiers.*

### Step 3, capture each room three times, 25 minutes
Follow `docs/capture/protocol.md` literally, once per tier:
- **LiDAR:** Stray Scanner, walk the perimeter, look at the ceiling in every room.
- **Video:** stock Camera, portrait, 1080p30, palm-cover between rooms.
- **Photos:** stock Camera, six stills per room, one folder per room.

Then capture room one a **second** time at the LiDAR tier, as a genuine repeatability pair rather
than the interleaved-frame split I had to use.

### Step 4, scan the same rooms with magicplan, 15 minutes
Start a project, scan both rooms with its LiDAR mode, export the plan, and note the app version from
its App Store page. Commit the export under `data/benchmark/app_exports/`.

*Closes: the head-to-head.*

### Step 5, run it, 10 minutes
```bash
uv run floorplan run <lidar capture> --out analysis/results/roomA
uv run python scripts/head_to_head.py analysis/results/roomA/plan.json \
    data/benchmark/app_exports/magicplan.yaml --out docs/report/head_to_head.md
make bench
make calibrate                 # now there is enough tape truth to fit the conformal factors
```

Then update three rows of `docs/compliance_matrix.md` from **not met** to **done**, and replace the
"see benchmark" cells in `docs/capture/device_matrix.md` with the real tape-referenced numbers.

---

## Plan B: no Pro device at all



Two of the three rows stay closed, but one improves and the largest scored component gets safer.

**Do this even if you have no Pro device**, because it is what the walk-in test will actually
exercise if the evaluators bring a non-Pro phone:

1. **Tape-measure one room you can stand in** and capture it at the **photo and video tiers** with
   your iPhone 15. That gives real tape ground truth for the two tiers that currently have none, and
   they are the two weakest. Roughly 30 minutes.
2. **Stage the two damage classes in that room** and measure them. The damage extents then have tape
   truth even though the benchmark property does not.
3. **Scan the same room with magicplan's non-LiDAR AR mode** and keep the export. This is *not* the
   head-to-head the brief asks for and must not be filed under that heading, but it is a legitimate
   extra table under a clearly different title: "photo tier against a non-LiDAR consumer app".
   Present it as the nearest available comparison and say plainly why it is not the requested one.

**What stays not met under Plan B:** the LiDAR-tier head-to-head, and tape truth on the supplied
benchmark property.

---

## What to say about this, in one paragraph

> Three requirements are marked not met. All three need physical access to the property the sample
> captures came from, or a LiDAR-class iPhone, and I had neither: the device available is an iPhone
> 15, which has no LiDAR sensor. Rather than substitute an easier comparison under the harder one's
> heading, I left those rows empty, moved the benchmark's weight onto the gates that need no external
> truth at all, and built the most independent reference the data allowed. The harness for the
> head-to-head and the template for tape truth are both committed, so each row is roughly ninety
> minutes of room time away from being filled.
