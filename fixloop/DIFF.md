# Fix loop: before and after

`before` = tag `fixloop-before` (9dca162d6), `after` = tag `fixloop-after` (9dca162d6). Both runs regenerate with `make before` and `make after`.

The declaration, written before the fix existed, is in [DECLARATION.md](DECLARATION.md).

## The gate that was declared

Repeatability, LiDAR tier: two passes over the same rooms sharing no frames.

| Metric | Before | Predicted | After | Gate | Verdict |
|---|---|---|---|---|---|
| Rooms found, pass A vs pass B | 3 vs 3 | 3 vs 3 | 4 vs 6 | must agree | FAIL |
| Median paired-room footprint IoU | 0.60 | >= 0.85 | 0.73 | - | short |
| Worst ceiling difference between passes | 120.6 cm | <= 1 cm | 58.0 cm | <= 1 cm | FAIL |
| Median wall difference between passes | - | <= 3 cm | 29.7 cm | - | short |
| Worst wall difference between passes | 62.1 cm | <= 15 cm (gate not expected to pass) | 119.6 cm | <= 1 cm | FAIL, as predicted |

## Everything else, so the fix cannot hide a regression

| Capture | Rooms before | Rooms after | Footprint before | Footprint after | Openings before | Openings after | Overlap before | Overlap after | Seconds before | Seconds after |
|---|---|---|---|---|---|---|---|---|---|---|
| flatA_lidar | 2 | 5 | 57.22 | 56.67 | 2 | 4 | 0.1 % | 0.1 % | 5.8 | 6.5 |
| flatA_lidar_driftoff | 2 | 6 | 57.00 | 55.29 | 1 | 6 | 0.2 % | 0.4 % | 4.8 | 5.5 |
| flatA_lidar_repa | 3 | 4 | 55.53 | 53.08 | 3 | 4 | 0.0 % | 0.4 % | 3.1 | 4.1 |
| flatA_lidar_repb | 3 | 6 | 44.93 | 55.25 | 1 | 5 | 0.0 % | 0.1 % | 3.0 | 4.6 |
| flatA_photo | 3 | 3 | 21.34 | 21.34 | 2 | 2 | 0.0 % | 0.0 % | 1.0 | 1.2 |
| flatA_video | 2 | 2 | 7.04 | 7.04 | 4 | 4 | 0.0 % | 0.0 % | 4.5 | 4.6 |
| flatB_lidar | 2 | 6 | 41.51 | 43.44 | 2 | 6 | 0.0 % | 1.2 % | 3.2 | 3.5 |
| roomC_lidar | 2 | 3 | 20.57 | 17.80 | 1 | 1 | 0.7 % | 1.3 % | 1.0 | 1.0 |

## Code diff

```

```

The whole fix is in the room-seeding half of `floorplan/geometry/cloud.py`. The pre-fix seeding is kept as `_threshold_cascade_seeds` and is still reachable with `floorplan run ... --room-seeds cascade`, so the before-run is reproducible from the after-run's code.

```diff

```
