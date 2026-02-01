from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Candidate:
    x: float
    y: float
    r: Optional[float] = None


def diameter_to_sigma(diameter_px: float) -> float:
    return diameter_px / (2.0 * math.sqrt(2.0))


def _sigma_or_default(value: Optional[float], default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and value <= 0:
        return default
    return float(value)


def detect_candidates(gray: np.ndarray, cfg: Dict[str, Any]) -> List[Candidate]:
    candidates: List[Candidate] = []
    method = cfg["candidates"]["method"]
    min_d = cfg["candidates"]["min_diameter_px"]
    max_d = cfg["candidates"]["max_diameter_px"]

    min_sigma = diameter_to_sigma(min_d)
    max_sigma = diameter_to_sigma(max_d)

    if method in ("log", "log+dog", "all"):
        candidates.extend(detect_log_blobs(gray, cfg, min_sigma, max_sigma))
    if method in ("dog", "log+dog", "all"):
        candidates.extend(detect_dog_blobs(gray, cfg, min_sigma, max_sigma))
    if method in ("hough", "all"):
        candidates.extend(detect_hough_circles(gray, cfg, min_sigma, max_sigma))

    if cfg["candidates"].get("dedup_centers", True):
        candidates = dedup_candidates(candidates, cfg["candidates"].get("dedup_radius_px", 4.0))

    max_cands = int(cfg["candidates"].get("max_candidates_per_image", 0))
    if max_cands > 0 and len(candidates) > max_cands:
        idx = np.linspace(0, len(candidates) - 1, max_cands, dtype=int)
        candidates = [candidates[i] for i in idx]

    return candidates


def detect_log_blobs(gray: np.ndarray, cfg: Dict[str, Any], min_sigma: float, max_sigma: float) -> List[Candidate]:
    try:
        from skimage.feature import blob_log
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for LoG blob detection.") from exc

    min_sigma = _sigma_or_default(cfg["candidates"].get("log_min_sigma"), min_sigma)
    max_sigma = _sigma_or_default(cfg["candidates"].get("log_max_sigma"), max_sigma)
    num_sigma = int(cfg["candidates"].get("log_num_sigma", 10))
    threshold = float(cfg["candidates"].get("log_threshold", 0.02))

    blobs = blob_log(gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    results: List[Candidate] = []
    for y, x, sigma in blobs:
        radius = math.sqrt(2.0) * sigma
        results.append(Candidate(float(x), float(y), float(radius)))
    return results


def detect_dog_blobs(gray: np.ndarray, cfg: Dict[str, Any], min_sigma: float, max_sigma: float) -> List[Candidate]:
    try:
        from skimage.feature import blob_dog
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for DoG blob detection.") from exc

    min_sigma = _sigma_or_default(cfg["candidates"].get("dog_min_sigma"), min_sigma)
    max_sigma = _sigma_or_default(cfg["candidates"].get("dog_max_sigma"), max_sigma)
    sigma_ratio = float(cfg["candidates"].get("dog_sigma_ratio", 1.6))
    threshold = float(cfg["candidates"].get("dog_threshold", 0.02))

    blobs = blob_dog(gray, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=threshold)
    results: List[Candidate] = []
    for y, x, sigma in blobs:
        radius = math.sqrt(2.0) * sigma
        results.append(Candidate(float(x), float(y), float(radius)))
    return results


def detect_hough_circles(gray: np.ndarray, cfg: Dict[str, Any], min_sigma: float, max_sigma: float) -> List[Candidate]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for Hough circle detection.") from exc

    min_r = cfg["candidates"].get("hough_minRadius")
    max_r = cfg["candidates"].get("hough_maxRadius")
    if min_r is None:
        min_r = int(round(math.sqrt(2.0) * min_sigma))
    if max_r is None:
        max_r = int(round(math.sqrt(2.0) * max_sigma))

    img8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    circles = cv2.HoughCircles(
        img8,
        cv2.HOUGH_GRADIENT,
        dp=float(cfg["candidates"].get("hough_dp", 1.2)),
        minDist=float(cfg["candidates"].get("hough_minDist", 8)),
        param1=float(cfg["candidates"].get("hough_param1", 100)),
        param2=float(cfg["candidates"].get("hough_param2", 30)),
        minRadius=int(min_r),
        maxRadius=int(max_r),
    )

    results: List[Candidate] = []
    if circles is not None:
        circles = np.squeeze(circles, axis=0)
        for x, y, r in circles:
            results.append(Candidate(float(x), float(y), float(r)))
    return results


def dedup_candidates(candidates: List[Candidate], radius: float) -> List[Candidate]:
    if radius <= 0:
        return candidates

    kept: List[Candidate] = []
    r2 = radius * radius
    cell_size = max(radius, 1.0)
    grid: Dict[Tuple[int, int], List[Candidate]] = {}

    for cand in candidates:
        cx = int(cand.x // cell_size)
        cy = int(cand.y // cell_size)
        keep = True
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                for other in grid.get((gx, gy), []):
                    dx = cand.x - other.x
                    dy = cand.y - other.y
                    if dx * dx + dy * dy <= r2:
                        keep = False
                        break
                if not keep:
                    break
            if not keep:
                break
        if keep:
            kept.append(cand)
            grid.setdefault((cx, cy), []).append(cand)
    return kept


def generate_grid_points(
    h: int, w: int, spacing: int, jitter: int, max_points: int, seed: int
) -> List[Tuple[float, float]]:
    if spacing <= 0:
        raise ValueError("grid_spacing_px must be > 0")

    ys = np.arange(spacing / 2.0, h, spacing)
    xs = np.arange(spacing / 2.0, w, spacing)
    points = [(float(x), float(y)) for y in ys for x in xs]

    if jitter > 0:
        rng = np.random.default_rng(seed)
        jitter_xy = rng.uniform(-jitter, jitter, size=(len(points), 2))
        points = [(x + jx, y + jy) for (x, y), (jx, jy) in zip(points, jitter_xy)]
        points = [(min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)) for x, y in points]

    if max_points > 0 and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[i] for i in idx]

    return points
