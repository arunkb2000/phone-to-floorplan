"""Classical damage detection: find what does not belong on a painted surface.

Zero-shot open-vocabulary detection turned out to be unreliable in both directions here. Asked for a
shower it found one at score 0.79 in an office that has none; shown an unmistakable brown ceiling
stain it scored 0.12, and a clear plaster crack scored nothing at all. A detector that both
hallucinates and misses is not a detector.

Interior damage has a simpler definition. A wall is a large, nearly uniform, nearly flat colour
field, and damage is a local departure from it: a stain is a region that has moved toward yellow-brown
and darkened with a soft edge; a crack is a thin, dark, elongated structure with a hard edge. Both are
found with a background model and two morphological operators, both are deterministic, and both
explain themselves. That is what ships.

The open-vocabulary model is kept as a second opinion for the classes that resist this treatment
(peeling paint, mould, holes), at a threshold high enough that it stays quiet.
"""
from __future__ import annotations

import cv2
import numpy as np

# a stain has to be this different from its surface before we call it damage
STAIN_DELTA_B = 7.0        # CIELab b*, positive is yellow
STAIN_DELTA_L = -4.0       # CIELab L*, negative is darker
STAIN_MIN_FRAC = 0.0016    # of the image
CRACK_MIN_LEN_PX = 60
CRACK_MAX_STROKE_RATIO = 0.16   # mean stroke width over length: a crack is thin for how long it is
CRACK_MIN_CONTRAST = 30.0
SURROUND_MAX_STD = 20.0         # a stain sits on a plain surface; furniture does not


def _background(lab: np.ndarray, ksize: int = 0) -> np.ndarray:
    """The surface as it would look undamaged: a heavy blur, which follows lighting but not damage."""
    h, w = lab.shape[:2]
    k = ksize or (max(h, w) // 6) | 1
    return cv2.GaussianBlur(lab, (k, k), 0)


def _components(mask: np.ndarray, min_area: int):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= min_area:
            out.append((labels == i, (x, y, x + bw, y + bh), int(area)))
    return out


def detect_stains(rgb: np.ndarray) -> list[dict]:
    """Regions that have moved toward yellow-brown and darkened relative to their own surface."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg = _background(lab)
    dL, da, db = (lab - bg)[..., 0], (lab - bg)[..., 1], (lab - bg)[..., 2]
    score_map = db + 0.5 * da
    mask = (score_map > STAIN_DELTA_B) & (dL < STAIN_DELTA_L)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8)).astype(bool)
    h, w = rgb.shape[:2]
    out = []
    for m, box, _a in _components(mask, int(STAIN_MIN_FRAC * h * w)):
        strength = float(np.mean(score_map[m]))
        # a soft edge separates a stain from a poster or a picture frame
        edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((9, 9), np.uint8)).astype(bool)
        softness = float(np.std(score_map[edge])) if edge.any() else 0.0
        # A stain sits on a plain painted surface. If what surrounds it is busy, it is furniture or a
        # poster and the brown thing in the middle of it is not damage. The annulus starts well clear
        # of the region, because a stain's own soft halo is not its surroundings.
        near = cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        far = cv2.dilate(m.astype(np.uint8), np.ones((85, 85), np.uint8)).astype(bool)
        ring = far & ~near
        surround = float(np.std(lab[..., 0][ring])) if ring.any() else 99.0
        if surround > SURROUND_MAX_STD:
            continue
        conf = float(np.clip(0.35 + 0.020 * strength + 0.010 * softness, 0.0, 0.95))
        out.append(dict(cls="water_stain", score=conf, box=box, mask=m,
                        why=f"b*+a*/2 up {strength:.1f}, L down, soft edge {softness:.1f}, "
                            f"plain surround {surround:.1f}"))
    return out


def _straightness(m: np.ndarray) -> float:
    """Mean distance of the component's pixels from its own best-fit line, in pixels.

    A wall-ceiling junction is a straight line and a crack is not, which is the cheapest way to stop
    the room's own edges being reported as damage.
    """
    ys, xs = np.nonzero(m)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
    return float(np.mean(d))


def detect_cracks(rgb: np.ndarray) -> list[dict]:
    """Thin, dark, elongated, wandering structures.

    Black-hat only responds to features narrower than its kernel, and a crack seen from half a metre
    is far wider in pixels than one seen from three metres, so the response is taken over several
    kernel sizes and the strongest kept.
    """
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bh = np.zeros_like(grey, np.uint8)
    for k in (13, 23, 37):
        bh = np.maximum(bh, cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT,
                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))))
    thr = max(CRACK_MIN_CONTRAST, float(np.percentile(bh, 99.5)))
    mask = cv2.morphologyEx((bh >= thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    h, w = grey.shape
    out = []
    for m, box, _area in _components(mask, 60):
        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        if len(pts) < 25:
            continue
        (_c), (w1, w2), _ang = cv2.minAreaRect(pts)
        length = max(w1, w2)
        # the bounding box of a wandering line is much wider than the line, so thickness is measured
        # as area over length, which is the mean stroke width
        stroke = max(len(pts) / max(length, 1.0), 1.0)
        aspect = length / stroke
        if length < CRACK_MIN_LEN_PX or stroke / max(length, 1.0) > CRACK_MAX_STROKE_RATIO or aspect < 3.5:
            continue
        width = stroke
        wander = _straightness(m)
        spans = (box[0] <= 2 and box[2] >= w - 3) or (box[1] <= 2 and box[3] >= h - 3)
        if spans or wander < max(1.6, 0.012 * length):
            continue                                   # a room edge, not a crack
        contrast = float(np.mean(bh[m]))
        conf = float(np.clip(0.30 + 0.004 * aspect + 0.004 * contrast + 0.02 * wander, 0.0, 0.95))
        out.append(dict(cls="crack", score=conf, box=box, mask=m,
                        why=f"black-hat {contrast:.0f}, {length:.0f}x{width:.0f}px, "
                            f"aspect {aspect:.1f}, wanders {wander:.1f}px"))
    return out


def _overlaps(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def detect(rgb: np.ndarray, min_score: float = 0.40) -> list[dict]:
    """Stains and cracks on one frame, strongest first.

    Stains win ties. The dark core of a stain also lights up the black-hat response, so a crack found
    inside a stain is the same damage counted twice under the wrong name, and the softer, coloured
    explanation is the right one.
    """
    stains = [d for d in detect_stains(rgb) if d["score"] >= min_score]
    cracks = [d for d in detect_cracks(rgb)
              if d["score"] >= min_score and not any(_overlaps(d["box"], s["box"]) for s in stains)]
    return sorted(stains + cracks, key=lambda d: -d["score"])[:12]
