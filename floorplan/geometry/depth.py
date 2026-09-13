"""Monocular metric depth (Depth Anything V2, metric-indoor) with a deterministic on-disk cache.

Disclosure: pretrained weights from depth-anything/Depth-Anything-V2-Metric-Indoor-{Small,Large}-hf,
fetched by scripts/fetch_weights.sh. Runs locally (MPS / CUDA / CPU). No network at run time.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

_MODEL = None
_DEVICE = None


def _device(pref: str = "auto"):
    import torch
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def weights_dir(size: str = "small") -> str:
    root = os.environ.get("FLOORPLAN_WEIGHTS", os.path.join(os.getcwd(), "weights"))
    return os.path.join(root, f"da2_metric_indoor_{size}")


def _load(size: str, device: str):
    global _MODEL, _DEVICE
    if _MODEL is not None and _DEVICE == device:
        return _MODEL
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    wd = weights_dir(size)
    proc = AutoImageProcessor.from_pretrained(wd)
    model = AutoModelForDepthEstimation.from_pretrained(wd).to(device).eval()
    torch.manual_seed(0)
    _MODEL, _DEVICE = (proc, model), device
    return _MODEL


def _key(rgb: np.ndarray, size: str) -> str:
    h = hashlib.sha256()
    h.update(rgb.tobytes()[::97])  # sparse but deterministic sample of the pixels
    h.update(f"{rgb.shape}{size}".encode())
    return h.hexdigest()[:24]


def predict_depth(rgb: np.ndarray, *, size: str = "small", device: str = "auto",
                  cache_dir: str | None = "cache/depth", use_cache: bool = True) -> np.ndarray:
    """Metric depth in metres, same HxW as rgb. Deterministic replay from cache when present."""
    key = _key(rgb, size)
    if cache_dir and use_cache:
        p = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(p):
            return np.load(p)["depth"].astype(np.float32)
    import torch
    from PIL import Image
    dev = _device(device)
    proc, model = _load(size, dev)
    img = Image.fromarray(rgb)
    inputs = proc(images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs).predicted_depth  # 1 x h x w, metres
    d = torch.nn.functional.interpolate(out.unsqueeze(1), size=rgb.shape[:2], mode="bilinear",
                                        align_corners=False)[0, 0].float().cpu().numpy()
    d = np.clip(d, 0.05, 30.0).astype(np.float32)
    if cache_dir and use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(os.path.join(cache_dir, key + ".npz"), depth=d.astype(np.float16))
    return d
