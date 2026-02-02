#!/usr/bin/env python3
"""
Standalone black-hat + DoG/LoG tiny-bubble mask generator.

This is independent of the FRST/SAM3 pipeline and only outputs the black-hat masks.

Usage:
  python bubbly_flows/tests/blackhat_mask.py \
    --input bubbly_flows/tests/img6001.png \
    --output result_blackhat.png
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:
    raise RuntimeError("OpenCV is required for black-hat detection. Install with: pip install opencv-python") from exc

from bubble_sam3.config import load_config
from bubble_sam3.outputs import build_rgba_cutout, build_rgba_overlay, ensure_output_dir
from bubble_sam3.postprocess import Instance, mask_bbox


def _sigma_ratio(min_sigma: float, max_sigma: float, num_sigma: int) -> float:
    if num_sigma <= 1 or min_sigma <= 0 or max_sigma <= 0 or max_sigma <= min_sigma:
        return 1.6
    return float((max_sigma / min_sigma) ** (1.0 / float(num_sigma - 1)))


def _detect_blobs(
    blackhat_f: np.ndarray,
    method: str,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    overlap: float,
) -> np.ndarray:
    try:
        from skimage.feature import blob_dog, blob_log
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for LoG/DoG blob detection.") from exc

    method = str(method or "dog").lower()
    if method == "log":
        return blob_log(
            blackhat_f,
            min_sigma=float(min_sigma),
            max_sigma=float(max_sigma),
            num_sigma=int(num_sigma),
            threshold=float(threshold),
            overlap=float(overlap),
        )
    sigma_ratio = _sigma_ratio(float(min_sigma), float(max_sigma), int(num_sigma))
    return blob_dog(
        blackhat_f,
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        sigma_ratio=float(sigma_ratio),
        threshold=float(threshold),
        overlap=float(overlap),
    )


def _mask_from_center(
    blackhat_u8: np.ndarray,
    center_xy: Tuple[float, float],
    patch_radius: int,
    patch_percentile: float,
    patch_use_otsu: bool,
) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    h, w = blackhat_u8.shape
    cx = int(round(center_xy[0]))
    cy = int(round(center_xy[1]))
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None, (0, 0)
    pr = max(1, int(patch_radius))
    x0 = max(0, cx - pr)
    x1 = min(w, cx + pr + 1)
    y0 = max(0, cy - pr)
    y1 = min(h, cy + pr + 1)
    patch = blackhat_u8[y0:y1, x0:x1]
    if patch.size == 0:
        return None, (0, 0)

    if patch_use_otsu:
        otsu_thresh, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = float(otsu_thresh)
    else:
        thresh = float(np.percentile(patch, patch_percentile))

    bw = patch >= thresh
    cy_l = cy - y0
    cx_l = cx - x0
    if cy_l < 0 or cx_l < 0 or cy_l >= bw.shape[0] or cx_l >= bw.shape[1]:
        return None, (0, 0)
    if not bw[cy_l, cx_l]:
        return None, (0, 0)

    bw_u8 = bw.astype(np.uint8)
    num, labels = cv2.connectedComponents(bw_u8, connectivity=8)
    if num <= 1:
        return None, (0, 0)
    lbl = labels[cy_l, cx_l]
    if lbl == 0:
        return None, (0, 0)
    comp = labels == lbl
    return comp, (x0, y0)


def _watershed_split_component(
    comp_mask: np.ndarray,
    gray_crop: np.ndarray,
    fg_thresh: float,
) -> List[np.ndarray]:
    comp_u8 = (comp_mask.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
    max_val = float(dist.max())
    if max_val <= 0:
        return [comp_mask]
    _, sure_fg = cv2.threshold(dist, max_val * float(fg_thresh), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 1:
        return [comp_mask]
    markers = markers + 1
    markers[comp_u8 == 0] = 0
    color = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    max_marker = int(markers.max())
    if max_marker <= 1:
        return [comp_mask]
    masks: List[np.ndarray] = []
    for marker_id in range(2, max_marker + 1):
        submask = markers == marker_id
        if np.any(submask):
            masks.append(submask)
    return masks or [comp_mask]


def build_blackhat_instances(
    gray_u8: np.ndarray,
    radius: int,
    percentile: float,
    area_min: int,
    area_max: int,
    enable_watershed: bool,
    watershed_min_area: int,
    watershed_fg_thresh: float,
    blob_method: str,
    blob_min_sigma: float,
    blob_max_sigma: float,
    blob_num_sigma: int,
    blob_threshold: float,
    blob_overlap: float,
    patch_radius: int,
    patch_percentile: float,
    patch_use_otsu: bool,
) -> List[Instance]:
    radius = max(1, int(radius))
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel)
    blackhat_f = blackhat.astype(np.float32) / 255.0

    blobs = _detect_blobs(
        blackhat_f,
        method=blob_method,
        min_sigma=blob_min_sigma,
        max_sigma=blob_max_sigma,
        num_sigma=blob_num_sigma,
        threshold=blob_threshold,
        overlap=blob_overlap,
    )

    instances: List[Instance] = []
    for blob in blobs:
        if len(blob) < 3:
            continue
        y, x, sigma = float(blob[0]), float(blob[1]), float(blob[2])
        pr = max(int(patch_radius), int(round(3.0 * sigma)))
        comp_mask, origin = _mask_from_center(
            blackhat,
            (x, y),
            patch_radius=pr,
            patch_percentile=patch_percentile,
            patch_use_otsu=patch_use_otsu,
        )
        if comp_mask is None:
            continue
        ox, oy = origin
        area = int(comp_mask.sum())
        if area < area_min or area > area_max:
            continue
        if enable_watershed and area >= watershed_min_area:
            submasks = _watershed_split_component(
                comp_mask,
                gray_u8[oy : oy + comp_mask.shape[0], ox : ox + comp_mask.shape[1]],
                fg_thresh=watershed_fg_thresh,
            )
        else:
            submasks = [comp_mask]
        for submask in submasks:
            if not np.any(submask):
                continue
            sub_area = int(submask.sum())
            if sub_area < area_min or sub_area > area_max:
                continue
            bbox_local = mask_bbox(submask)
            if bbox_local == (0, 0, 0, 0):
                continue
            lx0, ly0, lx1, ly1 = bbox_local
            crop = submask[ly0:ly1, lx0:lx1]
            crop_area = int(crop.sum())
            if crop_area <= 0:
                continue
            instances.append(
                Instance(
                    mask=crop,
                    score=None,
                    area=crop_area,
                    bbox=(ox + lx0, oy + ly0, ox + lx1, oy + ly1),
                )
            )

    if instances:
        return instances

    thresh = float(np.percentile(blackhat, percentile))
    bw = blackhat >= thresh
    bw_u8 = (bw.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw_u8, connectivity=8)
    instances = []
    for idx in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < area_min or area > area_max:
            continue
        comp_mask = labels[y : y + h, x : x + w] == idx
        bbox_local = mask_bbox(comp_mask)
        if bbox_local == (0, 0, 0, 0):
            continue
        lx0, ly0, lx1, ly1 = bbox_local
        crop = comp_mask[ly0:ly1, lx0:lx1]
        crop_area = int(crop.sum())
        if crop_area <= 0:
            continue
        instances.append(
            Instance(mask=crop, score=None, area=crop_area, bbox=(x + lx0, y + ly0, x + lx1, y + ly1))
        )
    return instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone black-hat blob masks")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output RGBA PNG path")
    parser.add_argument("--config", default=None, help="Optional JSON/JSONC config path")
    parser.add_argument(
        "--output_mode",
        choices=["cutout", "overlay"],
        default="overlay",
        help="Output rendering mode (default: overlay)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    bh_cfg: Dict[str, Any] = cfg.get("blackhat", {})

    image = Image.open(args.input).convert("RGB")
    image_rgb = np.array(image, dtype=np.uint8)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    instances = build_blackhat_instances(
        gray_u8=gray,
        radius=int(bh_cfg.get("radius", 5)),
        percentile=float(bh_cfg.get("percentile", 99.0)),
        area_min=int(bh_cfg.get("area_min", 5)),
        area_max=int(bh_cfg.get("area_max", 120)),
        enable_watershed=bool(bh_cfg.get("watershed", True)),
        watershed_min_area=int(
            bh_cfg.get("watershed_min_area", max(1, int(round(2.0 * np.pi * 5 ** 2))))
            or max(1, int(round(2.0 * np.pi * 5 ** 2)))
        ),
        watershed_fg_thresh=float(bh_cfg.get("watershed_fg_thresh", 0.5)),
        blob_method=str(bh_cfg.get("blob_method", "dog")),
        blob_min_sigma=float(bh_cfg.get("blob_min_sigma", 1.3)),
        blob_max_sigma=float(bh_cfg.get("blob_max_sigma", 2.8)),
        blob_num_sigma=int(bh_cfg.get("blob_num_sigma", 10)),
        blob_threshold=float(bh_cfg.get("blob_threshold", 0.01)),
        blob_overlap=float(bh_cfg.get("blob_overlap", 0.5)),
        patch_radius=int(bh_cfg.get("patch_radius", 6)),
        patch_percentile=float(bh_cfg.get("patch_percentile", 90.0)),
        patch_use_otsu=bool(bh_cfg.get("patch_use_otsu", False)),
    )

    if args.output_mode == "cutout":
        rgba = build_rgba_cutout(image_rgb, instances)
    else:
        rgba = build_rgba_overlay(
            image_rgb,
            instances,
            alpha=int(cfg["output"].get("overlay_alpha", 128)),
            colormap=cfg["output"].get("overlay_colormap", "tab20"),
        )

    ensure_output_dir(args.output)
    Image.fromarray(rgba, mode="RGBA").save(args.output)


if __name__ == "__main__":
    main()
