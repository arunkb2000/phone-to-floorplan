# Capture protocol — one page

Follow this page exactly. Setup takes 5 minutes. Each room takes about 2 minutes per tier.

## 1. Phone settings (once)
- Settings → Camera → **Formats → Most Compatible**.
- Settings → Camera → **Record Video → 1080p at 30 fps**.
- In the Camera app turn **Live Photo off** (the concentric-circles icon), use **1× zoom**, never Portrait mode.
- Turn on every light. Open every door and leave it open. Do not move furniture.
- Walk the spaces in one fixed order and give each a number: `01_room`, `02_kitchen`, `03_washroom`, `04_balcony` (use your own names after the number).

## 2. Photos tier (any iPhone 15 or newer)
For each space, in your fixed order, take **6 photos** (minimum 2, maximum 8), phone upright (portrait), held at chest height:
1. Photo 1: stand in the doorway you entered through, face into the space.
2. Photos 2–5: stand in each corner, aim at the opposite corner so two walls, the ceiling line and the floor line are all in frame. Step back to the wall if the room is small.
3. Last photo: from inside the space, face the doorway that leads to the **next** space. The whole door frame must be in the frame.

Put each space's photos in its own folder named as in step 1. All folders go inside one capture folder, for example `photo_MR1/01_room/…`.

## 3. Video tier (any iPhone 15 or newer)
One continuous clip for the whole property, phone **upright (portrait)** so floor and ceiling stay in frame, chest height, about one step per second:
1. Start in the first space, standing in its entrance doorway facing in.
2. Walk the perimeter once, about 1 m from the walls. On each wall, tilt slowly up to where the wall meets the ceiling, then down to where it meets the floor.
3. To move to the next space: **cover the lens fully with your palm for 2 seconds while you walk through the doorway**, then uncover and continue. This is how the pipeline knows a new space started.
4. Finish back in the first space. Stop recording. Typical total: 1 to 2 minutes per space.

Put the clip alone in a capture folder, for example `video_MR1/walk.mov`.

## 4. LiDAR tier (iPhone Pro / Pro Max only)
1. Install **Stray Scanner** (App Store, free). Allow camera access.
2. Tap record and walk exactly as in section 3, phone upright (portrait). Skip the palm-cover step; poses are recorded.
3. Tap stop. Files app → On My iPhone → Stray Scanner → the recording folder → Share → AirDrop to the laptop. Put the folder in a capture folder, for example `lidar_MR1/`.

## 5. Avoid (all tiers)
Fast turns, walking backwards, pointing at a window or mirror for more than a second, zooming, Portrait mode, Night mode, fingers on the lens (except the deliberate palm cover in the video tier).

## 6. Handing over and running
AirDrop the capture folder to the laptop, then:

```
floorplan run path/to/capture_folder --out results/
```

Outputs: `results/plan.json`, `results/plan.svg`, `results/plan.png`, `results/timing.json`. The tier is detected from the folder contents.
