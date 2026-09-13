# Sample data

**The sample captures are not in this repository.** They are 912 MB, two of the video files are over
GitHub's 100 MB per-file limit, and they are not mine to redistribute. Everything the pipeline
produced from them **is** committed, under [analysis/](../analysis/), so every reported number can be
checked without downloading anything.

This page describes what the data is, where to put it, and how to run the code on it.

---

## 1. Where to put it

Copy the three capture folders into `data/raw/` so the tree looks like this:

```
data/raw/
├── single_room/
├── single_scan_floor_only/
└── single_scan_with_ceiling/
```

That path is what `data/benchmark/bench.yaml` and every script expects. Nothing else needs changing.

## 2. What each capture is

| Folder | Frames | Duration | Walk | What makes it interesting |
|---|---|---|---|---|
| `single_room` | 1,715 | 37 s | 14.5 m | Short walk through one area. The camera never looks up, so the ceiling height comes back as a flagged estimate rather than a measurement. |
| `single_scan_floor_only` | 5,251 | 115 s | 54 m | A full walk around the property with the camera pointed down. Same story on ceilings, over more rooms. |
| `single_scan_with_ceiling` | 9,745 | 215 s | 100 m | A full walk including the ceilings. This is the one most results come from. |

All three are of the same working office, captured with **Stray Scanner** on an iPhone Pro.

## 3. What is inside one capture folder

This is the format Stray Scanner exports. Nothing was renamed or converted.

```
single_scan_with_ceiling/
├── camera_matrix.csv     lens parameters
├── odometry.csv          where the phone was, for every frame
├── imu.csv               accelerometer and gyroscope
├── depth/                000000.png … 009744.png
├── confidence/           000000.png … 009744.png
└── rgb.mp4               the video, one frame per depth frame
```

### `camera_matrix.csv`

Three lines, nine numbers: the standard 3×3 camera matrix for the video resolution.

```
1601.0514, 0.0, 955.82275
0.0, 1601.0514, 717.7025
0.0, 0.0, 1.0
```

The first and middle numbers are the focal length in pixels; `955.8` and `717.7` are the centre of
the image, which tells you the video is 1920 × 1440.

### `odometry.csv`

One row per frame. The phone's own tracking, from ARKit.

```
timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y
65764.237587125, 000000, 0.00091, -0.00022, -0.00021, 0.7245, -0.6819, 0.0563, 0.0830, 1581.2, …
```

| Column | Meaning |
|---|---|
| `timestamp` | seconds since the phone booted |
| `frame` | matches the depth and confidence filenames |
| `x, y, z` | where the phone is, in metres, starting from wherever recording began |
| `qx, qy, qz, qw` | which way it is pointing, as a quaternion |
| `fx, fy, cx, cy` | lens parameters for that individual frame |

**Two things about this file cost me real time, so they are worth writing down.**

The world is **y-up**, not z-up: the second number is height. And although Apple's own camera
convention is the OpenGL one, the rotation stored here is already in OpenCV form. Applying the
textbook conversion between the two smears the floor across three metres of height and nothing
downstream works. Without it, the floor collapses into a single two-centimetre band 1.47 m below
where recording started, which is exactly where you would expect a phone being carried to be.

`scripts/check_convention.py` prints the comparison that settles it:

```bash
uv run python scripts/check_convention.py data/raw/single_room
```

### `depth/NNNNNN.png`

One per frame. 256 × 192, 16-bit greyscale. Each pixel is a distance in **millimetres**, so 2450
means 2.45 m away. Zero means the sensor got nothing back, which is normal for windows, mirrors and
anything beyond about five metres.

### `confidence/NNNNNN.png`

Same size, 8-bit, one value per depth pixel: `2` high, `1` medium, `0` low. The pipeline uses only
the high ones. On these captures that is around 98 % of pixels at close range, dropping off with
distance.

### `imu.csv`

`timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z` — acceleration and rotation rate, at a higher
rate than the frames. Not used by this pipeline; the position data in `odometry.csv` already has it
folded in.

### `rgb.mp4`

The video, 1920 × 1440 at about 46 frames per second, one frame per depth frame. Only the damage
detection reads it. All the geometry works from depth and position alone, which is why the floor plan
results reproduce even if the video is missing.

## 4. How to run the code on it

```bash
make setup      # Python and dependencies, about 3 minutes
make weights    # the pretrained model, about 2 minutes

uv run floorplan run data/raw/single_scan_with_ceiling --out results/
```

```
tier=lidar  rooms=5  openings=4  damage=7  6.5s -> results/plan.json
```

You get `results/plan.png`, `plan.svg`, `plan.json` and `timing.json`.

Other things you can do with it:

```bash
# every capture, scored, with the accuracy tables
make bench

# the same scan without position correction, for comparison
uv run floorplan run data/raw/single_scan_with_ceiling --out results/nodrift --drift-correction off

# rebuild every number in the reports from the raw captures, about 12 minutes
make reproduce
```

## 5. What is derived from it, and rebuilt rather than shipped

The assessment needs the same rooms captured three ways, and needs two separate passes over the same
rooms. Both are built from these captures by script, so they are not committed either:

| Built by | What it makes | Why |
|---|---|---|
| `scripts/make_rgb_tiers.py` | photos and a video, from the video track | The supplied captures are depth scans only, but the assessment requires all three input types on the same rooms. These get JPEGs and an mp4 and nothing else, so they are no easier than a stranger's phone. |
| `scripts/split_capture.py` | two passes over the same rooms, sharing no frames | For the consistency check. Two genuine walks would be better; this is what the data allows. |
| `scripts/make_synthetic_room.py` | a room with exact known dimensions and staged damage | Needs no sample data at all, and is the only place the measurements can be checked against a known answer. |
| `scripts/make_reference_gt.py` | reference measurements taken one depth frame at a time | Used to score accuracy. Its limits are in [report/benchmark.md](report/benchmark.md). |

`make reproduce` runs all four and then the full benchmark.

## 6. If you do not have the sample data

Two things still work with no sample data at all:

```bash
make test                                    # 25 tests, none need the captures
uv run python scripts/make_synthetic_room.py # builds the test room with known dimensions
uv run floorplan run data/benchmark/captures/synth_room --out results/
```

And every result is already committed: accuracy tables in
[analysis/bench/after/gates.md](../analysis/bench/after/gates.md), the improvement round in
[analysis/fixloop/](../analysis/fixloop/), and the floor plans themselves as PNG and JSON in each run
folder.
