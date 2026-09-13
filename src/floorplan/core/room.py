"""The tier-independent room representation that the JSON assembler consumes.

Every tier (photo, video, LiDAR) produces a list of these. A rectangle is just the 4-edge case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from floorplan.core.types import Measurement


@dataclass
class RoomOut:
    key: str
    label: str = ""
    polygon: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # world metres, closed ring
    wall_ids: list = field(default_factory=list)                            # letters or indices, per edge
    wall_len: list = field(default_factory=list)                            # Measurement per edge
    openings: list = field(default_factory=list)                            # (edge_index, Opening)
    ceiling: Measurement | None = None
    ceiling_spread: float = 0.0
    area: Measurement | None = None
    damage: list = field(default_factory=list)
    n_frames: int = 0
    coverage: float = 0.0          # fraction of wall length actually measured against the sensor
    exposure: float = 1.0
    mirror_suspected: bool = False
    warnings: list = field(default_factory=list)
    placement_confidence: float = 0.9
    surfaces_extra: dict = field(default_factory=dict)
    label_confidence: float = 0.3
    fixtures: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)     # source Frames, for the damage pass
    geoms: list = field(default_factory=list)

    def polygon_area(self) -> float:
        p = np.asarray(self.polygon, float)
        if len(p) < 3:
            return 0.0
        return float(0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))
