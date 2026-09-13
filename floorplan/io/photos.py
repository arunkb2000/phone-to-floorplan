"""Photo-tier loader: one capture folder containing one sub-folder per room of 2-8 stills.

No depth, no poses. Intrinsics from EXIF (35 mm-equivalent focal length) with a per-device fallback.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

try:  # HEIC support if the phone was not set to "Most Compatible"
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

from floorplan.core.types import Frame

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff"}
MAX_SIDE = 1024  # working resolution for geometry; damage uses the same frame

# 35 mm-equivalent focal length of the main camera when EXIF is missing (iPhone 13..17 main = 24-26 mm).
DEFAULT_F35 = 26.0


def _exif_f35(img: Image.Image) -> float | None:
    try:
        ex = img.getexif()
        ifd = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
        f35 = ifd.get(0xA405) or ex.get(0xA405)  # FocalLengthIn35mmFilm
        if f35:
            return float(f35)
    except Exception:
        return None
    return None


def intrinsics_from_f35(f35: float, w: int, h: int) -> np.ndarray:
    """Pinhole K from a 35 mm-equivalent focal length. f35 is defined on the 43.27 mm diagonal."""
    diag_px = float(np.hypot(w, h))
    f = f35 * diag_px / 43.27
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def load_image(path: str, max_side: int = MAX_SIDE) -> tuple[np.ndarray, np.ndarray, float | None]:
    img = Image.open(path)
    f35 = _exif_f35(img)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(round(w * s)), int(round(h * s))), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    K = intrinsics_from_f35(f35 or DEFAULT_F35, rgb.shape[1], rgb.shape[0])
    return rgb, K, f35


def room_key_sort(keys: Iterable[str]) -> list[str]:
    def k(s):
        m = re.match(r"^(\d+)", s)
        return (int(m.group(1)) if m else 10**6, s)
    return sorted(keys, key=k)


def label_from_key(key: str) -> str:
    return re.sub(r"^\d+[_\- ]*", "", key).replace("_", " ").strip() or key


def is_photo_capture(capture_dir: str) -> bool:
    subs = [d for d in os.listdir(capture_dir) if os.path.isdir(os.path.join(capture_dir, d)) and not d.startswith(".")]
    if not subs:
        return False
    return any(any(os.path.splitext(f)[1].lower() in IMG_EXT for f in os.listdir(os.path.join(capture_dir, d))) for d in subs)


def load_photo_capture(capture_dir: str) -> dict[str, list[Frame]]:
    """Returns {room_key: [Frame,...]} in walk order; frames in filename order with protocol roles."""
    rooms: dict[str, list[Frame]] = {}
    subs = [d for d in os.listdir(capture_dir) if os.path.isdir(os.path.join(capture_dir, d)) and not d.startswith(".")]
    for key in room_key_sort(subs):
        d = os.path.join(capture_dir, key)
        files = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT and not f.startswith("."))
        if not files:
            continue
        frames = []
        for i, f in enumerate(files):
            rgb, K, _ = load_image(os.path.join(d, f))
            role = "entry" if i == 0 else ("exit" if (i == len(files) - 1 and len(files) >= 2) else "corner")
            frames.append(Frame(path=os.path.join(d, f), rgb=rgb, K=K, t=float(i), room_key=key, role=role))
        rooms[key] = frames
    return rooms
