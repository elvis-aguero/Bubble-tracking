from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class Instance:
    mask: np.ndarray  # bool h x w (bbox-local)
    score: Optional[float]
    area: int
    bbox: Tuple[int, int, int, int]  # full-image coords (x0, y0, x1, y1)


def resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.array(mask_img) > 127


def fill_holes(mask: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    if not cfg["hole_fill"].get("enable_hole_fill", True):
        return mask

    method = cfg["hole_fill"].get("hole_fill_method", "binary_fill_holes")
    min_hole_area = int(cfg["hole_fill"].get("min_hole_area_px", 0))
    fill_only_ringlike = cfg["hole_fill"].get("fill_only_if_ringlike", False)
    ringlike_thresh = float(cfg["hole_fill"].get("ringlike_threshold", 0.15))

    if method == "binary_fill_holes":
        try:
            from scipy.ndimage import binary_fill_holes
            from skimage.morphology import remove_small_holes
        except ImportError as exc:
            raise RuntimeError("scipy and scikit-image are required for hole filling.") from exc
        if min_hole_area > 0:
            filled = remove_small_holes(mask, area_threshold=min_hole_area)
        else:
            filled = binary_fill_holes(mask)
        hole_area = int(filled.sum() - mask.sum())
        if fill_only_ringlike and mask.sum() > 0:
            if hole_area / float(mask.sum()) < ringlike_thresh:
                return mask
        return filled

    if method == "closing":
        try:
            from skimage.morphology import binary_closing, disk, remove_small_holes
        except ImportError as exc:
            raise RuntimeError("scikit-image is required for morphological closing.") from exc

        radius = int(cfg["hole_fill"].get("closing_radius", 2))
        closed = binary_closing(mask, disk(radius)) if radius > 0 else mask
        hole_area = int(closed.sum() - mask.sum())
        if fill_only_ringlike and mask.sum() > 0:
            if hole_area / float(mask.sum()) < ringlike_thresh:
                return mask
        if min_hole_area > 0:
            closed = remove_small_holes(closed, area_threshold=min_hole_area)
        return closed

    raise ValueError(f"Unknown hole_fill_method: {method}")


def maybe_convex_hull(mask: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    if not cfg["postprocess"].get("enable_convex_hull", False):
        return mask
    try:
        from skimage.morphology import convex_hull_image
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for convex hull computation.") from exc
    return convex_hull_image(mask)


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)


def touches_border(
    mask: np.ndarray, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, int], border_px: int
) -> bool:
    if border_px <= 0:
        return False
    h, w = image_shape
    x0, y0, x1, y1 = bbox
    if x0 >= border_px and y0 >= border_px and x1 <= w - border_px and y1 <= h - border_px:
        return False

    if x0 < border_px:
        cols = min(border_px, x1) - x0
        if cols > 0 and mask[:, :cols].any():
            return True
    if x1 > w - border_px:
        start = max(0, (w - border_px) - x0)
        if start < mask.shape[1] and mask[:, start:].any():
            return True
    if y0 < border_px:
        rows = min(border_px, y1) - y0
        if rows > 0 and mask[:rows, :].any():
            return True
    if y1 > h - border_px:
        start = max(0, (h - border_px) - y0)
        if start < mask.shape[0] and mask[start:, :].any():
            return True

    return False


def mask_perimeter(mask: np.ndarray) -> float:
    try:
        from skimage.measure import perimeter
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for perimeter calculation.") from exc
    return float(perimeter(mask, neighborhood=8))


def mask_solidity(mask: np.ndarray) -> float:
    try:
        from skimage.morphology import convex_hull_image
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for solidity calculation.") from exc
    hull = convex_hull_image(mask)
    area = float(mask.sum())
    hull_area = float(hull.sum())
    if hull_area == 0:
        return 0.0
    return area / hull_area


def mask_iou(
    mask_a: np.ndarray,
    bbox_a: Tuple[int, int, int, int],
    mask_b: np.ndarray,
    bbox_b: Tuple[int, int, int, int],
) -> float:
    inter = mask_intersection_area(mask_a, bbox_a, mask_b, bbox_b)
    if inter == 0:
        return 0.0
    union = mask_a.sum() + mask_b.sum() - inter
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_intersection_area(
    mask_a: np.ndarray,
    bbox_a: Tuple[int, int, int, int],
    mask_b: np.ndarray,
    bbox_b: Tuple[int, int, int, int],
) -> int:
    x0 = max(bbox_a[0], bbox_b[0])
    y0 = max(bbox_a[1], bbox_b[1])
    x1 = min(bbox_a[2], bbox_b[2])
    y1 = min(bbox_a[3], bbox_b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    ax0 = x0 - bbox_a[0]
    ay0 = y0 - bbox_a[1]
    ax1 = x1 - bbox_a[0]
    ay1 = y1 - bbox_a[1]
    bx0 = x0 - bbox_b[0]
    by0 = y0 - bbox_b[1]
    bx1 = x1 - bbox_b[0]
    by1 = y1 - bbox_b[1]
    return int(
        np.logical_and(mask_a[ay0:ay1, ax0:ax1], mask_b[by0:by1, bx0:bx1]).sum()
    )


def mask_containment(
    mask_a: np.ndarray,
    bbox_a: Tuple[int, int, int, int],
    mask_b: np.ndarray,
    bbox_b: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    inter = mask_intersection_area(mask_a, bbox_a, mask_b, bbox_b)
    if inter == 0:
        return 0.0, 0.0
    area_a = mask_a.sum()
    area_b = mask_b.sum()
    if area_a == 0 or area_b == 0:
        return 0.0, 0.0
    return float(inter) / float(area_a), float(inter) / float(area_b)


def consolidate_instances(instances: List[Instance], cfg: Dict[str, Any], image_shape: Tuple[int, int]) -> List[Instance]:
    h, w = image_shape
    image_area = h * w

    filtered: List[Instance] = []
    min_area = int(cfg["postprocess"].get("min_area_px", 0))
    max_area_fraction = float(cfg["postprocess"].get("max_area_fraction", 1.0))
    border_px = int(cfg["postprocess"].get("border_exclusion_px", 0))
    circularity_min = float(cfg["postprocess"].get("circularity_min", 0.0))
    solidity_min = float(cfg["postprocess"].get("solidity_min", 0.0))

    for inst in instances:
        if inst.area < min_area:
            continue
        if max_area_fraction > 0 and inst.area > int(image_area * max_area_fraction):
            continue
        if touches_border(inst.mask, inst.bbox, image_shape, border_px):
            continue
        if circularity_min > 0:
            perim = mask_perimeter(inst.mask)
            if perim <= 0:
                continue
            circ = 4.0 * math.pi * inst.area / (perim * perim)
            if circ < circularity_min:
                continue
        if solidity_min > 0:
            sol = mask_solidity(inst.mask)
            if sol < solidity_min:
                continue
        filtered.append(inst)

    if not cfg["postprocess"].get("enable_consolidation", True) or len(filtered) <= 1:
        return filtered

    iou_thresh = float(cfg["postprocess"].get("iou_dedup_thresh", 0.5))
    containment_thresh = float(cfg["postprocess"].get("containment_thresh", 0.9))

    def score_key(inst: Instance) -> float:
        if inst.score is not None:
            return float(inst.score)
        return float(inst.area)

    sorted_instances = sorted(filtered, key=score_key, reverse=True)
    kept: List[Instance] = []
    replacement_count = 0

    for inst in sorted_instances:
        drop = False
        to_remove: List[int] = []
        for idx, kept_inst in enumerate(kept):
            iou = mask_iou(inst.mask, inst.bbox, kept_inst.mask, kept_inst.bbox)
            if iou >= iou_thresh:
                drop = True
                break
            containment_a_in_b, containment_b_in_a = mask_containment(
                inst.mask, inst.bbox, kept_inst.mask, kept_inst.bbox
            )
            if containment_a_in_b >= containment_thresh:
                drop = True
                break
            if containment_b_in_a >= containment_thresh:
                to_remove.append(idx)
        if drop:
            continue
        if to_remove:
            for idx in reversed(to_remove):
                kept.pop(idx)
            replacement_count += len(to_remove)
        kept.append(inst)

    logging.getLogger(__name__).info(
        "Consolidation replacements due to containment: %d", replacement_count
    )
    return kept
