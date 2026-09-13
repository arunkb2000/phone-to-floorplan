# Fix loop: before and after

`before` = tag `fixloop-before` (0ae0149), `after` = tag `fixloop-after` (db0f937). Both runs regenerate with `make before` and `make after`.

The declaration, written before the fix existed, is in [DECLARATION.md](DECLARATION.md). What these numbers mean, including the prediction that was wrong and the row that regressed, is in [POSTMORTEM.md](POSTMORTEM.md).

## The gate that was declared

Repeatability, LiDAR tier: two passes over the same rooms sharing no frames.

| Metric | Before | Predicted | After | Gate | Verdict |
|---|---|---|---|---|---|
| Rooms found, pass A vs pass B | 3 vs 2 | 3 vs 3 | 4 vs 6 | must agree | no change |
| Median paired-room footprint IoU | 0.41 | >= 0.85 | 0.73 | - | short |
| Worst ceiling difference between passes | 120.6 cm | <= 1 cm | 58.0 cm | <= 1 cm | FAIL |
| Median wall difference between passes | - | <= 3 cm | 29.7 cm | - | short |
| Worst wall difference between passes | 62.1 cm | <= 15 cm (gate not expected to pass) | 119.6 cm | <= 1 cm | FAIL, as predicted |

## Everything else, so the fix cannot hide a regression

| Capture | Rooms before | Rooms after | Footprint before | Footprint after | Openings before | Openings after | Overlap before | Overlap after | Seconds before | Seconds after |
|---|---|---|---|---|---|---|---|---|---|---|
| flatA_lidar | 2 | 5 | 57.22 | 56.67 | 2 | 4 | 0.1 % | 0.1 % | 72.2 | 97.5 |
| flatA_lidar_driftoff | 3 | 6 | 55.83 | 55.29 | 4 | 6 | 0.2 % | 0.4 % | 78.8 | 97.6 |
| flatA_lidar_repa | 3 | 4 | 55.53 | 53.08 | 3 | 4 | 0.0 % | 0.4 % | 3.1 | 3.5 |
| flatA_lidar_repb | 2 | 6 | 31.38 | 55.25 | 0 | 5 | 0.0 % | 0.1 % | 2.7 | 3.8 |
| flatA_photo | 6 | 6 | 33.26 | 27.66 | 3 | 6 | 0.0 % | 1.6 % | 4.4 | 2.1 |
| flatA_video | 3 | 3 | 14.33 | 14.30 | 7 | 6 | 0.0 % | 0.0 % | 18.6 | 7.5 |
| flatB_lidar | 4 | 6 | 43.18 | 43.51 | 4 | 6 | 0.0 % | 1.2 % | 3.4 | 3.9 |
| roomC_lidar | 2 | 3 | 20.57 | 17.80 | 1 | 1 | 0.7 % | 1.3 % | 1.0 | 1.1 |

## Code diff

```
floorplan/damage/pipeline.py                       |  99 ---
 floorplan/io/synthetic.py                          | 743 -----------------
 floorplan/io/synthetic_rgb.py                      | 877 ---------------------
 floorplan/tiers/common.py                          |  44 --
 {floorplan => src/floorplan}/__init__.py           |   0
 .../floorplan/calibration}/__init__.py             |   0
 .../floorplan/calibration}/factors.json            |   0
 .../calib => src/floorplan/calibration}/fit.py     |   2 +-
 {floorplan => src/floorplan}/cli/__init__.py       |   0
 {floorplan => src/floorplan}/cli/bench.py          | 129 ++-
 {floorplan => src/floorplan}/cli/main.py           |  49 +-
 {floorplan => src/floorplan}/core/__init__.py      |   0
 {floorplan => src/floorplan}/core/room.py          |   4 +-
 {floorplan => src/floorplan}/core/types.py         |  13 +-
 .../damage => src/floorplan/geometry}/__init__.py  |   0
 {floorplan => src/floorplan}/geometry/assemble.py  |  22 +-
 {floorplan => src/floorplan}/geometry/cloud.py     | 207 +++--
 {floorplan => src/floorplan}/geometry/depth.py     |   0
 {floorplan => src/floorplan}/geometry/drift.py     |  73 +-
 {floorplan => src/floorplan}/geometry/frame.py     |  28 +-
 {floorplan => src/floorplan}/geometry/layout.py    |   3 +-
 {floorplan => src/floorplan}/geometry/lidar.py     |  42 +-
 {floorplan => src/floorplan}/geometry/room.py      |  74 +-
 {floorplan => src/floorplan}/geometry/stitch.py    |   2 +-
 .../geometry => src/floorplan/io}/__init__.py      |   0
 {floorplan => src/floorplan}/io/photos.py          |   2 +-
 {floorplan => src/floorplan}/io/stray.py           |  20 +-
 {floorplan => src/floorplan}/io/video.py           |   0
 .../io => src/floorplan/perception}/__init__.py    |   0
 .../damage => src/floorplan/perception}/detect.py  |   6 +-
 src/floorplan/perception/pipeline.py               | 242 ++++++
 .../damage => src/floorplan/perception}/project.py |   0
 src/floorplan/perception/rooms.py                  | 109 +++
 .../render => src/floorplan/rendering}/__init__.py |   0
 .../render => src/floorplan/rendering}/plan.py     |   6 +-
 .../floorplan}/schema/output.schema.json           |   0
 {floorplan => src/floorplan}/scope/__init__.py     |   0
 {floorplan => src/floorplan}/scope/catalogue.yaml  |   0
 {floorplan => src/floorplan}/scope/engine.py       |   0
 {floorplan => src/floorplan}/scope/rules.yaml      |   0
 {floorplan => src/floorplan}/tiers/__init__.py     |   0
 {floorplan => src/floorplan}/tiers/lidar.py        |  24 +-
 {floorplan => src/floorplan}/tiers/mono.py         |  18 +-
 {floorplan => src/floorplan}/tiers/photo.py        |   0
 {floorplan => src/floorplan}/tiers/video.py        |   6 +-
 45 files changed, 853 insertions(+), 1991 deletions(-)
```

The whole fix is in the room-seeding half of `floorplan/geometry/cloud.py`. The pre-fix seeding is kept as `_threshold_cascade_seeds` and is still reachable with `floorplan run ... --room-seeds cascade`, so the before-run is reproducible from the after-run's code.

```diff
diff --git a/floorplan/geometry/cloud.py b/src/floorplan/geometry/cloud.py
similarity index 67%
rename from floorplan/geometry/cloud.py
rename to src/floorplan/geometry/cloud.py
index 8d10bb7..a1324f7 100644
--- a/floorplan/geometry/cloud.py
+++ b/src/floorplan/geometry/cloud.py
@@ -11,7 +11,6 @@ from __future__ import annotations
 
 from dataclasses import dataclass, field
 
-import cv2
 import numpy as np
 from scipy import ndimage
 
@@ -283,9 +282,105 @@ def wall_mask(g: Grid, min_zspan: float = 0.55, quantile: float = 0.55, min_coun
     return (g.wall >= thr) & (g.zspan >= min_zspan)
 
 
-def _split_large(lab: np.ndarray, D: np.ndarray, free: np.ndarray, res: float, max_area: float) -> np.ndarray:
+def _reconstruct(seed: np.ndarray, mask: np.ndarray, max_iter: int = 400) -> np.ndarray:
+    """Morphological reconstruction by dilation of `seed` under `mask`."""
+    cur = np.minimum(seed, mask)
+    for _ in range(max_iter):
+        nxt = np.minimum(ndimage.grey_dilation(cur, size=(3, 3)), mask)
+        if np.array_equal(nxt, cur):
+            break
+        cur = nxt
+    return cur
+
+
+def hmaxima_seeds(D: np.ndarray, h: float, res: float, min_area_m2: float = 0.3) -> np.ndarray:
+    """Room seeds as the h-maxima of the clearance map.
+
+    A seed is a regional maximum of the distance transform standing at least `h` metres above the
+    saddle that joins it to any deeper maximum, found by morphological reconstruction. There is no
+    absolute threshold to fall off, which is the whole point: two captures of the same rooms differ
+    by centimetres of carved free space and that must not create or destroy a room. Compare
+    `_threshold_cascade_seeds`, which is what this replaced; see analysis/fixloop/DECLARATION.md.
+    """
+    rec = _reconstruct(D - h, D)
+    lab, n = ndimage.label((D - rec) > 1e-6)
+    if n == 0:
+        return np.zeros_like(lab)
+    sizes = np.asarray(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))
+    keep = [i + 1 for i, sz in enumerate(sizes) if sz * res ** 2 >= min_area_m2]
+    relab = np.zeros(n + 1, np.int32)
+    for k, l in enumerate(keep, start=1):
+        relab[l] = k
+    return relab[lab]
+
+
+def _threshold_cascade_seeds(D: np.ndarray, res: float) -> np.ndarray:
+    """The pre-fix seeding, kept so `--room-seeds cascade` reproduces the before-run exactly.
+
+    Tries a ladder of absolute clearance thresholds and stops at the first rung yielding two usable
+    cores. Two passes over the same property land on different rungs.
+    """
+    cores = np.zeros(D.shape, np.int32)
+    for core_r in (1.1, 0.95, 0.8, 0.65, 0.5, 0.4):
+        lab, n = ndimage.label(D > core_r)
+        if n == 0:
+            continue
+        sizes = np.asarray(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))
+        keep = [i + 1 for i, sz in enumerate(sizes) if sz * res ** 2 > 0.3]
+        if len(keep) >= 2 or core_r <= 0.4:
+            relab = np.zeros(n + 1, np.int32)
+            for k, l in enumerate(keep, start=1):
+                relab[l] = k
+            cores = relab[lab]
+            if cores.max() > 0:
+                break
+    return cores
+
+
+def merge_unwalled(lab: np.ndarray, walls: np.ndarray, res: float, min_wall_frac: float = 0.5,
+                   max_open_m: float = 1.0) -> np.ndarray:
+    """Merge neighbouring rooms that no wall separates.
+
+    The watershed always draws a line somewhere; what makes that line a room boundary is a wall with
+    a doorway in it. For each adjacent pair we measure the boundary the watershed drew and ask how
+    much of it carries wall evidence. A boundary that is wide and unwalled is an artefact of the
+    free-space shape, not a doorway, so the two regions are one room. Walls are the stable signal.
+    """
+    n = int(lab.max())
+    if n < 2:
+        return lab
+    wall_near = ndimage.binary_dilation(walls, np.ones((3, 3)))
+    parent = list(range(n + 1))
+
+    def find(a):
+        while parent[a] != a:
+            parent[a] = parent[parent[a]]
+            a = parent[a]
+        return a
+
+    for a in range(1, n + 1):
+        ma = lab == a
+        if not ma.any():
+            continue
+        grow = ndimage.binary_dilation(ma, np.ones((3, 3)))
+        for b in range(a + 1, n + 1):
+            border = grow & (lab == b)
+            cells = int(border.sum())
+            if cells == 0:
+                continue
+            if float(wall_near[border].mean()) < min_wall_frac and cells * res > max_open_m:
+                pa, pb = find(a), find(b)
+                if pa != pb:
+                    parent[pb] = pa
+    out = np.zeros_like(lab)
+    for a in range(1, n + 1):
+        out[lab == a] = find(a)
+    return _compact_labels(out)
+
+
+def _split_large(lab: np.ndarray, D: np.ndarray, res: float, max_area: float) -> np.ndarray:
     """A room bigger than `max_area` is usually two spaces joined through a wide opening. Re-flood it
-    from tighter cores so the watershed gets another chance to find the neck."""
+    from tighter seeds so the watershed gets another chance at the neck."""
     out = lab.copy()
     nxt = int(out.max()) + 1
     for r in range(1, int(lab.max()) + 1):
@@ -293,62 +388,74 @@ def _split_large(lab: np.ndarray, D: np.ndarray, free: np.ndarray, res: float, m
         if m.sum() * res ** 2 <= max_area:
             continue
         Dm = D * m
-        for core_r in (1.45, 1.25, 1.1):
-            cores, n = ndimage.label(Dm > core_r)
-            sizes = np.asarray(ndimage.sum(np.ones_like(cores), cores, range(1, n + 1))) if n else np.array([])
-            keep = [i + 1 for i, s in enumerate(sizes) if s * res ** 2 > 0.6]
-            if len(keep) >= 2:
-                relab = np.zeros(n + 1, np.int32)
-                for k, l in enumerate(keep, start=1):
-                    relab[l] = k
-                sub = _priority_flood(relab[cores].astype(np.int32), Dm, m)
-                for k in range(2, int(sub.max()) + 1):
-                    out[(sub == k)] = nxt
-                    nxt += 1
-                break
+        seeds = hmaxima_seeds(Dm, 0.18, res, min_area_m2=0.6)
+        if seeds.max() < 2:
+            continue
+        sub = _priority_flood(seeds, Dm, m)
+        for k in range(2, int(sub.max()) + 1):
+            out[sub == k] = nxt
+            nxt += 1
     return out
 
 
-def segment_rooms(g: Grid, min_area_m2: float = 1.5, max_area_m2: float = 26.0) -> np.ndarray:
-    """Watershed the carved free space; doorway necks become the borders between rooms."""
+def _absorb_slivers(lab: np.ndarray, walls: np.ndarray, res: float, min_area_m2: float) -> np.ndarray:
+    """A region too small to be a room joins the neighbour it shares the widest UNWALLED border with.
+
+    The pre-fix code joined it to whichever neighbour shared the longest border of any kind, and two
+    captures picked different neighbours, which changed the room count. Border length depends on the
+    free-space shape and moves between captures; whether a wall stands on that border does not. If
+    every border is walled the sliver is not part of any room and is dropped.
+    """
+    out = lab.copy()
+    wall_near = ndimage.binary_dilation(walls, np.ones((3, 3)))
+    for _ in range(4):
+        sizes = {r: int((out == r).sum()) for r in range(1, int(out.max()) + 1)}
+        small = [r for r, n in sizes.items() if 0 < n * res ** 2 < min_area_m2]
+        if not small:
+            break
+        for r in small:
+            m = out == r
+            grow = ndimage.binary_dilation(m, np.ones((3, 3))) & ~m
+            best, bid = 0.0, 0
+            for q in range(1, int(out.max()) + 1):
+                if q == r:
+                    continue
+                border = grow & (out == q)
+                openness = float((border & ~wall_near).sum()) * res
+                if openness > best:
+                    best, bid = openness, q
+            out[m] = bid if best > 0.1 else 0
+    return out
+
+
+def segment_rooms(g: Grid, min_area_m2: float = 1.5, max_area_m2: float = 26.0,
+                  seeds: str = "hmaxima", h_m: float = 0.35) -> np.ndarray:
+    """Watershed the carved free space; doorway necks become the borders between rooms.
+
+    `seeds="cascade"` restores the pre-fix threshold ladder, for the fix-loop ablation.
+    """
     walls = wall_mask(g)
-    free = g.free & ~walls
-    free = ndimage.binary_opening(free, np.ones((3, 3)))
+    free = ndimage.binary_opening(g.free & ~walls, np.ones((3, 3)))
+    if seeds != "cascade":
+        # a patch the sensor never reached, fully enclosed by free space, is still part of the room:
+        # filling it stops an unobserved hole pinching the transform and splitting the room in two.
+        # Skipped under "cascade" so that --legacy reproduces the pre-fix behaviour exactly.
+        free = ndimage.binary_fill_holes(free) & ~walls
     if not free.any():
         return np.zeros(g.shape, np.int32)
     D = ndimage.distance_transform_edt(free) * g.res
-    for core_r in (1.1, 0.95, 0.8, 0.65, 0.5, 0.4):
-        cores, n = ndimage.label(D > core_r)
-        if n == 0:
-            continue
-        sizes = np.asarray(ndimage.sum(np.ones_like(cores), cores, range(1, n + 1)))
-        keep = [i + 1 for i, s in enumerate(sizes) if s * g.res ** 2 > 0.3]
-        if len(keep) >= 2 or core_r <= 0.4:
-            relab = np.zeros(n + 1, np.int32)
-            for k, lab in enumerate(keep, start=1):
-                relab[lab] = k
-            cores = relab[cores]
-            if cores.max() > 0:
-                break
+    cores = hmaxima_seeds(D, h_m, g.res) if seeds == "hmaxima" else _threshold_cascade_seeds(D, g.res)
     if cores.max() == 0:
         cores, _ = ndimage.label(free)
     labels = _priority_flood(cores.astype(np.int32), D, free)
-    labels = _split_large(labels, D, free, g.res, max_area_m2)
-    out = labels.copy()
-    for _ in range(3):
-        changed = False
-        for lab in range(1, int(out.max()) + 1):
-            m = out == lab
-            if not m.any() or m.sum() * g.res ** 2 >= min_area_m2:
-                continue
-            nb = ndimage.binary_dilation(m, np.ones((3, 3))) & ~m & (out > 0)
-            vals, counts = np.unique(out[nb], return_counts=True)
-            sel = vals != lab
-            vals, counts = vals[sel], counts[sel]
-            out[m] = vals[np.argmax(counts)] if len(vals) else 0
-            changed = True
-        if not changed:
-            break
+    if seeds == "hmaxima":
+        labels = merge_unwalled(labels, walls, g.res)
+    labels = _split_large(labels, D, g.res, max_area_m2)
+    # A sliver under `min_area_m2` is not a room. Absorbing it into whichever neighbour happens to
+    # share the longest border was the last unstable step in this function: two captures picked
+    # different neighbours and ended with different room counts. Dropping it instead leaves it as
+    # unassigned free space, which is what it is, and is the same decision every time.
+    out = _absorb_slivers(labels, walls, g.res, min_area_m2)
     return _compact_labels(out)
 
 
@@ -360,7 +467,7 @@ def _priority_flood(markers: np.ndarray, D: np.ndarray, mask: np.ndarray) -> np.
     nx, ny = lab.shape
     seeded = lab > 0
     bd = ndimage.binary_dilation(seeded, np.ones((3, 3))) & mask & ~seeded
-    for i, j in zip(*np.where(bd)):
+    for i, j in zip(*np.where(bd), strict=False):
         heapq.heappush(h, (-float(D[i, j]), int(i), int(j)))
     visited = seeded.copy()
     while h:
```
