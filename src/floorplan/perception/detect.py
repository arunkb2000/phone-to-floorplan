"""OWLv2 zero-shot damage / opening detection with GrabCut masks.

Boxes are returned in the ORIGINAL image pixel frame. OWLv2 pads the input to a
square (bottom/right) before resizing to 960x960; transformers'
``post_process_object_detection`` rescales by max(H, W) to undo that, and the
result is clipped to the image here as a guard.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

DAMAGE_QUERIES: dict[str, list[str]] = {
    "water_stain": ["a water stain on a wall", "a brown damp patch on the ceiling", "water damage stain"],
    "mold": ["black mold on a wall", "mould patches"],
    "crack": ["a crack in the wall", "a crack in plaster", "cracked paint line"],
    "peeling_paint": ["peeling paint", "flaking paint on a wall"],
    "hole": ["a hole in the wall", "impact damage in drywall"],
    "efflorescence": ["white salt deposits on a wall"],
}

OPENING_QUERIES: dict[str, list[str]] = {
    "door": ["a door", "an open doorway", "a door frame"],
    "window": ["a window", "a glass window"],
}

_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/floorplan/perception -> repo root


def _resolve_weights(weights_dir: str | Path) -> Path:
    p = Path(weights_dir)
    if p.is_absolute() or p.exists():
        return p
    alt = _REPO_ROOT / p
    return alt if alt.exists() else p


def weights_available(weights_dir: str | Path = "weights/owlv2_base") -> bool:
    p = _resolve_weights(weights_dir)
    if not (p / "config.json").exists():
        return False
    return any(p.glob("*.safetensors")) or any(p.glob("*.bin"))


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def box_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """Greedy NMS; returns kept indices sorted by descending score."""
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        if all(box_iou(boxes[i], boxes[j]) < iou_thr for j in keep):
            keep.append(int(i))
    return keep


def grabcut_mask(rgb: np.ndarray, box, iters: int = 3, max_side: int = 640, min_frac: float = 0.05) -> np.ndarray:
    """Foreground mask (HxW bool) for ``box`` via GrabCut on a downscaled copy.

    Falls back to the full box when GrabCut fails or keeps < ``min_frac`` of the box.
    """
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = sorted((max(0.0, x0), min(float(W), x1)))
    y0, y1 = sorted((max(0.0, y0), min(float(H), y1)))
    full = np.zeros((H, W), bool)
    bx0, by0 = int(np.floor(x0)), int(np.floor(y0))
    bx1, by1 = int(np.ceil(x1)), int(np.ceil(y1))
    full[by0:by1, bx0:bx1] = True
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        return full

    s = min(1.0, max_side / max(H, W))
    if s < 1.0:
        small = cv2.resize(rgb, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    h, w = small.shape[:2]
    sx0, sy0 = int(np.floor(x0 * s)), int(np.floor(y0 * s))
    sx1, sy1 = int(np.ceil(x1 * s)), int(np.ceil(y1 * s))
    # GrabCut needs some background outside the rect: keep a 1 px margin.
    sx0, sy0 = max(1, sx0), max(1, sy0)
    sx1, sy1 = min(w - 1, sx1), min(h - 1, sy1)
    rw, rh = sx1 - sx0, sy1 - sy0
    if rw < 2 or rh < 2 or rw * rh > 0.95 * w * h:
        return full

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    bgr = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    try:
        cv2.grabCut(bgr, mask, (sx0, sy0, rw, rh), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return full
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if fg.sum() < min_frac * rw * rh:
        return full
    if s < 1.0:
        fg = cv2.resize(fg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return fg & full


class DamageDetector:
    """OWLv2 zero-shot detector. The model is loaded lazily once and cached on the instance."""

    def __init__(self, weights_dir: str | Path = "weights/owlv2_base", device: str = "auto",
                 score_threshold: float = 0.18):
        self.weights_dir = _resolve_weights(weights_dir)
        self.device = resolve_device(device)
        self.score_threshold = float(score_threshold)
        self.opening_threshold = 0.2
        self.max_dets = 10
        self.nms_iou = 0.5
        self._model = None
        self._processor = None
        self.last_inference_s: float = float("nan")

    # ---------------------------------------------------------------- model
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if not weights_available(self.weights_dir):
            raise FileNotFoundError(f"OWLv2 weights not found in {self.weights_dir}")
        self._processor = Owlv2Processor.from_pretrained(str(self.weights_dir), local_files_only=True)
        model = Owlv2ForObjectDetection.from_pretrained(
            str(self.weights_dir), local_files_only=True, dtype=torch.float32
        )
        model.eval()
        try:
            model.to(self.device)
        except (RuntimeError, NotImplementedError):  # pragma: no cover - MPS unavailable
            self.device = "cpu"
            model.to("cpu")
        self._model = model

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def processor(self):
        self._load()
        return self._processor

    # ------------------------------------------------------------ inference
    def _run(self, rgb: np.ndarray, queries: dict[str, list[str]], threshold: float,
             max_dets: int) -> list[dict]:
        """Raw boxes (no masks): [{"cls", "score", "box"}] in original pixel coordinates."""
        import torch
        from PIL import Image

        self._load()
        rgb = np.ascontiguousarray(rgb)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        H, W = rgb.shape[:2]
        flat: list[str] = []
        owner: list[str] = []
        for cls, qs in queries.items():
            for q in qs:
                flat.append(q)
                owner.append(cls)

        t0 = time.perf_counter()
        inputs = self._processor(text=[flat], images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        outputs.logits = outputs.logits.float().cpu()
        outputs.pred_boxes = outputs.pred_boxes.float().cpu()
        # transformers v5 moved this off Owlv2Processor; the image processor keeps it (v4 and v5)
        # and rescales normalised boxes by max(H, W) to undo OWLv2's bottom/right square padding.
        post = getattr(self._processor, "image_processor", self._processor).post_process_object_detection
        res = post(outputs, threshold=threshold, target_sizes=[(H, W)])[0]
        self.last_inference_s = time.perf_counter() - t0

        boxes = res["boxes"].numpy().astype(np.float64)
        scores = res["scores"].numpy().astype(np.float64)
        labels = res["labels"].numpy().astype(int)
        if len(boxes) == 0:
            return []
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H)

        dets: list[dict] = []
        cls_of = np.array([owner[l] for l in labels])
        for cls in queries:
            idx = np.where(cls_of == cls)[0]
            if len(idx) == 0:
                continue
            keep = nms(boxes[idx], scores[idx], self.nms_iou)
            for k in keep:
                i = idx[k]
                b = boxes[i]
                if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                    continue
                dets.append({
                    "cls": cls,
                    "score": float(scores[i]),
                    "box": (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                })
        dets.sort(key=lambda d: -d["score"])
        return dets[:max_dets]

    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Damage detections with GrabCut masks:
        [{"cls": str, "score": float, "box": (x0, y0, x1, y1) px, "mask": HxW bool}]."""
        dets = self._run(rgb, DAMAGE_QUERIES, self.score_threshold, self.max_dets)
        for d in dets:
            d["mask"] = grabcut_mask(rgb, d["box"])
        return dets

    def run_queries(self, rgb: np.ndarray, queries: dict[str, list[str]], threshold: float) -> list[dict]:
        """Boxes for an arbitrary query set, no mask refinement. Used for room-type fixtures."""
        return self._run(rgb, queries, threshold, self.max_dets)

    def detect_openings(self, rgb: np.ndarray) -> list[dict]:
        """Door / window boxes (same dict shape, ``mask`` is the box) for vetoing phantom openings."""
        dets = self._run(rgb, OPENING_QUERIES, self.opening_threshold, self.max_dets)
        H, W = rgb.shape[:2]
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            m = np.zeros((H, W), bool)
            m[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
            d["mask"] = m
        return dets
