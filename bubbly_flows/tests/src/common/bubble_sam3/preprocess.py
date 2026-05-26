from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image


def preprocess_gray(image_rgb: Image.Image, cfg: Dict[str, Any]) -> np.ndarray:
    gray = np.array(image_rgb.convert("L"), dtype=np.float32) / 255.0

    if cfg["preprocess"].get("invert_gray", False):
        gray = 1.0 - gray

    if not cfg["preprocess"].get("enable_contrast_norm", True):
        return gray

    method = cfg["preprocess"].get("contrast_method", "clahe")
    if method == "clahe":
        try:
            from skimage import exposure
        except ImportError as exc:
            raise RuntimeError("scikit-image is required for CLAHE preprocessing.") from exc
        gray = exposure.equalize_adapthist(
            gray,
            clip_limit=cfg["preprocess"].get("clahe_clip_limit", 0.03),
            kernel_size=cfg["preprocess"].get("clahe_kernel_size", 64),
        )
    elif method == "rescale":
        p_low, p_high = cfg["preprocess"].get("rescale_percentiles", [2, 98])
        lo, hi = np.percentile(gray, p_low), np.percentile(gray, p_high)
        if hi > lo:
            gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown contrast_method: {method}")

    gamma = cfg["preprocess"].get("gamma", 1.0)
    if gamma and gamma != 1.0:
        gray = np.clip(gray, 0.0, 1.0) ** gamma

    return gray
