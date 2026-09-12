# Capture protocol (one page) — DRAFT v0

Follow this page exactly. It takes about 10 minutes to set up and 1 to 2 minutes per room to capture.

## Before you start (all tiers)
1. Turn on every light. Open every interior door and leave it open. Do not move furniture.
2. Put one bank card (any credit/debit card) flat on the floor just inside the entrance of each room, long edge toward the door. Leave it there for the whole capture.
3. Hold the phone at chest height, in **portrait** for LiDAR and photos, **landscape** for video. Move slowly: about one step per second.

## Tier A — Photos (any iPhone 15 or newer, Camera app)
- Camera app, Photo mode, **1×** zoom, Live Photo **off**, no Portrait mode.
- Make one folder per room, named in the order you walk: `01_living`, `02_hall`, `03_bedroom`, ...
- In each room take **6 photos** (minimum 2, maximum 8):
  - Photo 1: standing in the doorway you came in through, facing into the room, bank card in view.
  - Photos 2–5: one from each corner, aiming at the opposite corner so two walls and the ceiling line are in frame.
  - Last photo: standing inside the room, facing the doorway that leads to the next room, with the whole door frame in frame.
- Keep the phone level. Do not zoom. Do not crop.

## Tier B — Video (any iPhone 15 or newer, Camera app)
- Settings → Camera → Record Video → **1080p at 30 fps**. Landscape.
- One continuous clip for the whole property. Start at the entrance, bank card in view for 2 seconds.
- In each room walk the perimeter once about 1 m from the walls, then slowly tilt up to show the ceiling edge and down to show the floor edge on every wall.
- Walk through each doorway slowly, keeping the door frame in frame while you pass.
- Finish where you started. Total: about 60 to 90 seconds per room.

## Tier C — LiDAR (iPhone Pro models only)
- Install **Stray Scanner** from the App Store (free). Open it, allow camera access.
- Tap record. Walk exactly as in Tier B (perimeter, tilt up and down per wall, slow doorways, finish where you started). Portrait orientation.
- Tap stop. The recording appears in the app's list.

## Avoid (all tiers)
- Fast turns, walking backwards, covering the camera, pointing at a window or mirror for more than a second, zooming, night mode.

## Handing over the files
- **Photos:** AirDrop or copy the room folders into one folder, e.g. `mycapture/`.
- **Video:** AirDrop the clip; put it alone in a folder, e.g. `mycapture/walk.mov`.
- **LiDAR:** Files app → On My iPhone → Stray Scanner → select the recording folder → Share → AirDrop to the laptop. Put it in `mycapture/`.
- Then run: `floorplan run mycapture --out results`

_Draft notes (removed in the final page): export path in Stray Scanner to be confirmed on-device in week 1; card step to be user-tested; 1× vs 0.5× to be decided by the photo-tier benchmark._
