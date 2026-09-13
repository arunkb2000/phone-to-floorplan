"""Shared data types. Every module talks in these; nothing else crosses module boundaries."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """One image with whatever sensor data came with it. Units: metres, pixels."""
    path: str
    rgb: np.ndarray                      # H x W x 3 uint8
    K: np.ndarray                        # 3 x 3 intrinsics for rgb resolution
    depth: np.ndarray | None = None   # H x W float32 metres (same res as rgb after resize) or None
    conf: np.ndarray | None = None    # H x W uint8 (0 low, 1 med, 2 high) or None
    pose: np.ndarray | None = None    # 4 x 4 camera-to-world, or None
    t: float = 0.0                       # seconds
    room_key: str = ""                   # e.g. "01_room"; empty if unknown
    role: str = ""                       # "entry" | "corner" | "exit" | "" (from protocol position)
    depth_source: str = ""               # "lidar" | "monocular" | "synthetic"
    frame_index: int = -1                # index in the source capture, for lazy RGB lookup


@dataclass
class Plane:
    normal: np.ndarray                   # unit, world frame (z up)
    d: float                             # n·x + d = 0
    kind: str                            # floor | ceiling | wall
    n_inliers: int = 0
    rms: float = 0.0
    points: np.ndarray | None = None  # inlier points (N x 3), kept for openings/damage


@dataclass
class Measurement:
    value: float
    sigma: float
    source: str = "measured"             # measured | inferred | prior
    method: str = ""
    n_support: int = 0
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    def to_json(self) -> dict:
        return {
            "value": round(float(self.value), 4),
            "ci_low": round(float(self.ci_low), 4),
            "ci_high": round(float(self.ci_high), 4),
            "sigma": round(float(self.sigma), 4),
            "source": self.source,
            "method": self.method,
            "n_support": int(self.n_support),
        }


@dataclass
class Opening:
    wall_id: str
    type: str                            # door | window | passage
    offset: Measurement                  # along wall from its start
    width: Measurement
    height: Measurement | None = None
    sill: Measurement | None = None
    confidence: float = 0.5
    connects_to: str = ""


@dataclass
class DamageRegion:
    surface_id: str
    cls: str
    cls_conf: float
    area_m2: Measurement
    bbox_uv_m: tuple                     # (u0, v0, u1, v1) in the surface's UV frame, metres
    polygon_uv_m: list = field(default_factory=list)
    evidence_frames: list = field(default_factory=list)


@dataclass
class RoomEstimate:
    """A room in its own gravity-aligned frame: x along wall 'A' direction, origin at the corner."""
    key: str                             # folder / segment key, e.g. "01_room"
    label: str                           # human label, e.g. "room", "kitchen"
    length_x: Measurement                # extent along x (walls A and C)
    length_y: Measurement                # extent along y (walls B and D)
    ceiling: Measurement
    openings: list = field(default_factory=list)      # Opening
    damage: list = field(default_factory=list)        # DamageRegion
    n_frames: int = 0
    coverage: float = 0.0                # fraction of the 4 walls actually observed
    exposure: float = 1.0                # 0..1 image quality score
    mirror_suspected: bool = False
    warnings: list = field(default_factory=list)
    ceiling_spread: float = 0.0
    planes: list = field(default_factory=list)        # Plane, kept for debug/render
    entry_wall: str = "C"                # which wall holds the entrance door (protocol: C)
    exit_wall: str = ""                  # wall with the door to the next room, if found
    exit_offset: float = float("nan")
    origin_xy: tuple = (0.0, 0.0)          # LiDAR: world coords of the room corner (wall D, wall C)
    placement: dict = field(default_factory=dict)  # global placement after stitching: {x, y, rot_k}

    @property
    def area(self) -> float:
        return self.length_x.value * self.length_y.value
