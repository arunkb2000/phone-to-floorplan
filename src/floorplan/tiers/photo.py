"""Photo tier: one folder of 2-8 stills per room, no depth, no poses."""
from __future__ import annotations

from floorplan.io.photos import load_photo_capture
from floorplan.tiers.mono import chain_stitch, rooms_from_grouped_frames


def run_photo(capture_dir: str, *, depth_size: str, device: str, use_cache: bool, log: dict):
    groups = load_photo_capture(capture_dir)
    if not groups:
        raise ValueError(f"{capture_dir}: expected one sub-folder of photos per room")
    log["n_frames"] = sum(len(v) for v in groups.values())
    rooms = rooms_from_grouped_frames(groups, "photo", depth_size=depth_size, device=device,
                                      use_cache=use_cache, log=log)
    return rooms, chain_stitch(rooms)
