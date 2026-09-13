# Capture protocol

**Route 2: stock capture.** Nothing to build, nothing to side-load. One free App Store app for the
LiDAR tier, the stock Camera app for the other two. A non-engineer can be capturing in under five
minutes.

Follow this page literally. If a step is ambiguous, tell us and we will fix the page, not the excuse.

---

## Before you start (all tiers)

1. Turn on every light in the space. Open every interior door and leave it open.
2. Do not move furniture. We measure the room you have, not a tidied one.
3. Hold the phone **upright (portrait)**, screen facing you, at chest height, roughly level.
   Portrait matters: it puts the floor line and the ceiling line in the same frame, and both tiers
   without depth need that to recover the room's height.
4. Walk at about one slow step per second. No fast turns, no walking backwards.
5. Decide your walk order now and stick to it: for example room, kitchen, washroom, balcony.

**Optional but it tightens the photo and video tiers by about half:** put any bank card flat on the
floor just inside the door of the first room, long edge pointing into the room. Every card is
85.6 x 54.0 mm, so it gives the pipeline a known length. The pipeline runs fine without it and says
in its output whether it found one.

---

## Tier A: Photos. Any iPhone 15 or newer

Camera app, Photo mode, **1x** zoom, Live Photo **off**, never Portrait mode.

Make one folder per room on your phone or laptop, named in walk order:
`01_room`, `02_kitchen`, `03_washroom`, `04_balcony`.

In each room take **6 photos** (minimum 2, maximum 8), in this order:

1. Stand in the doorway you came in through, face into the room, take one.
2. Stand in each of the four corners in turn and aim at the opposite corner, so that two walls, the
   ceiling line and the floor line are all in the frame. In a small room, put your back to the
   corner. Four photos.
3. Stand inside the room facing the doorway that leads to the **next** room, with the whole door
   frame in the picture. One photo. In the last room, face the way you came in.

Keep the phone upright and level. Do not zoom. Do not crop.

---

## Tier B: Video. Any iPhone 15 or newer

Settings, Camera, Record Video, **1080p at 30 fps**. Camera app, Video mode, **portrait**, 1x.

One continuous clip for the whole property:

1. Start in the first room's entrance doorway, facing in.
2. Walk the perimeter of the room once, about one metre from the walls. On each wall, tilt slowly up
   until you see where the wall meets the ceiling, then down until you see where it meets the floor.
3. **To move to the next room, cover the lens completely with your palm for two seconds while you
   walk through the doorway,** then uncover it and carry on. That is how the pipeline knows a new
   room started. It is the only unusual thing on this page and it matters.
4. Repeat for every room, in your walk order.
5. Finish back where you started and stop recording.

About 60 to 90 seconds per room. If you forget the palm cover, the pipeline still works: it falls
back to detecting the room change from the geometry and says so in its output.

---

## Tier C: LiDAR. iPhone Pro or Pro Max only (15 Pro, 16 Pro, 17 Pro, or any iPad Pro with LiDAR)

1. Install **Stray Scanner** from the App Store. It is free. Open it and allow camera access.
2. Tap record. Walk exactly as in Tier B, phone upright, but **without** the palm cover: this tier
   records its own position, so it does not need the marker.
3. Look at the ceiling at least once in every room. Sweep up to the ceiling on one wall per room.
   The sensor cannot report a ceiling height it never saw, and the pipeline will tell you it had to
   fall back to a prior if you skip this.
4. Tap stop.

---

## What to avoid, every tier

Fast turns. Walking backwards. Fingers over the lens (except the deliberate palm cover in Tier B).
Pointing at a window or a mirror for more than a second or two. Zooming. Night mode. Portrait mode.

Mirrors, glass and wet-looking floors are all handled, but they cost accuracy, so give them a normal
pass rather than lingering on them.

---

## Handing the files over

| Tier | What to send | Where it goes |
|---|---|---|
| Photos | The room folders, inside one folder | `mycapture/01_room/…`, `mycapture/02_kitchen/…` |
| Video | The clip, alone in a folder | `mycapture/walk.mov` |
| LiDAR | Files app, On My iPhone, Stray Scanner, the recording folder, Share, AirDrop | `mycapture/` |

Then, on the laptop, one command:

```bash
floorplan run mycapture --out results
```

The tier is detected from what is in the folder. You get `results/plan.json`, `results/plan.svg`,
`results/plan.png` and `results/timing.json`.
