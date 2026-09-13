from __future__ import annotations

from floorplan.io.photos import load_photo_capture
from floorplan.tiers.common import rooms_from_frames


def run_photo(capture_dir: str, *, depth_size: str, device: str, use_cache: bool, log: dict):
    rooms = load_photo_capture(capture_dir)
    log["n_rooms"] = len(rooms)
    log["n_frames"] = sum(len(v) for v in rooms.values())
    return rooms_from_frames(rooms, "photo", depth_size=depth_size, device=device, use_cache=use_cache, log=log)
