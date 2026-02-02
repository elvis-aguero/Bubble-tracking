#!/usr/bin/env python3
"""
Combine FRST bubble-center detection with SAM3 point prompting.

Workflow:
  1) Detect bubble centers with FRST (from classical_test.py).
  2) Use centers as positive points for SAM3 tracker segmentation.
  3) Add a classical black-hat + threshold + components tiny-mask branch.
  4) Optionally run PCS text prompting (e.g., "tiny bubbles") to add masks.
  5) Produce three outputs:
     - FRST (+ micro prompt) masks
     - Black-hat masks
     - Big-bubble prompt masks only
     - Consolidated masks from both pipelines


Environment Setup (copy-paste for this environment):
    export HF_HOME=/users/eaguerov/scratch/hf
    export CUDA_VISIBLE_DEVICES=0  # if needed
    interact -q gpu -g 1 -n 4 -t 04:00:00 -m 16g
    eval "$(mamba shell hook --shell bash)"

Usage (copy/paste):
  python bubbly_flows/tests/bubble_frst_sam3_mask.py \
    --input bubbly_flows/tests/img6001.png \
    --output result.png \
    --frst_text_prompt "tiny bubbles" \
    --big_text_prompt "bubbles"

Flag overview:
  Core:
    --input --output --config --device --seed
  SAM3:
    --sam_model --points_per_batch --multimask_output --allow_download
    --frst_backend --frst_text_prompt --big_text_prompt --big_backend --text_prompt --disable_pcs
  Pipeline:
    --enable_candidates/--disable_candidates
    --enable_tiling/--disable_tiling
    --enable_hole_fill/--disable_hole_fill
    --enable_consolidation/--disable_consolidation
    --output_mode overlay|cutout
  Output:
    --debug_dir --output_json --no_output_json --output_csv --include_rle
  FRST:
    --r_min --r_max --r_step --alpha --mag_percentile
    --peak_percentile --nms_size --border --max_peaks
  Black-hat:
    --enable_blackhat/--disable_blackhat
    --blackhat_radius --blackhat_percentile --blackhat_area_min --blackhat_area_max
    --blackhat_watershed/--blackhat_no_watershed
    --blackhat_watershed_min_area --blackhat_watershed_fg_thresh

Notes:
  - Default output is an overlay with per-instance colors.
  - output_mode=cutout produces an RGBA cutout (alpha = union of masks).
  - Output files are derived from --output:
      <stem>_frst.png, <stem>_blackhat.png, <stem>_big.png, and the combined result at --output.
  - Point prompts are tiled with tile_size=8*r_max and overlap>=2.5*r_max.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:
    raise RuntimeError("OpenCV is required for FRST detection. Install with: pip install opencv-python") from exc

from classical_test import frst_symmetry_map, pick_peaks
from bubble_sam3.backend import Sam3ConceptBackend
from bubble_sam3.config import apply_cli_overrides, ensure_hf_home, load_config
from bubble_sam3.outputs import (
    build_rgba_cutout,
    build_rgba_overlay,
    ensure_output_dir,
    resolve_output_paths,
    save_candidate_viz,
    save_instance_outputs,
)
from bubble_sam3.postprocess import (
    Instance,
    consolidate_instances,
    fill_holes,
    mask_bbox,
    maybe_convex_hull,
    resize_mask_to_shape,
)
from bubble_sam3.tiling import create_tiles, pad_image
from frst_point_backend_fb import FrstPointBackendFB
from frst_point_backend_hf import FrstPointBackendHF

try:
    import torch
except ImportError as exc:
    raise RuntimeError("torch is required for SAM3. Install with: pip install torch") from exc


def resolve_output_path(input_path: str, output_arg: Optional[str]) -> str:
    """Resolve output path relative to the input image by default."""
    in_path = Path(input_path).expanduser().resolve()
    base_dir = in_path.parent
    out_dir = base_dir / "output"

    if not output_arg:
        return str(out_dir / f"{in_path.stem}_frst_sam3_overlay.png")

    out = Path(output_arg).expanduser()
    if out.is_absolute():
        return str(out)
    if out.parent == Path("."):
        return str(out_dir / out.name)
    return str(base_dir / out)


def resolve_logs_dir(input_path: str) -> str:
    in_path = Path(input_path).expanduser().resolve()
    return str(in_path.parent / "logs")


def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bubble_frst_sam3_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized. Log file: %s", log_file)
    return log_file


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FRST + SAM3 bubble segmentation pipeline")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output RGBA PNG path. If omitted, defaults to <input_dir>/output/<input_stem>_frst_sam3_overlay.png. "
            "If a bare filename is provided, it is written into <input_dir>/output/."
        ),
    )
    parser.add_argument("--config", default=None, help="Optional JSON/JSONC config path")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--debug_dir", default=None, help="Optional directory to save debug PNGs")
    parser.add_argument("--output_json", default=None, help="Optional per-instance JSON output path")
    parser.add_argument("--no_output_json", action="store_true", help="Disable per-instance JSON output")
    parser.add_argument("--output_csv", default=None, help="Optional per-instance CSV output path")
    parser.add_argument("--include_rle", action="store_true", help="Include COCO RLE masks in JSON")
    parser.add_argument("--sam_model", default=None, help="Override SAM3 model name/id")
    parser.add_argument("--points_per_batch", type=int, default=None, help="Points per SAM3 tracker batch")
    parser.add_argument("--multimask_output", action="store_true", help="Enable multiple masks per point")
    parser.add_argument("--allow_download", action="store_true", help="Allow SAM3 model downloads if cache miss")
    parser.add_argument(
        "--frst_text_prompt",
        default="micro-sized bubbles",
        help="PCS text prompt for FRST pipeline (e.g., 'micro-sized bubbles')",
    )
    parser.add_argument(
        "--frst_backend",
        choices=["fb", "hf"],
        default="hf",
        help="Backend for FRST point prompts: hf=transformers (default), fb=facebookresearch",
    )
    parser.add_argument(
        "--big_text_prompt",
        default="bubbles",
        help="PCS text prompt for large bubbles only (e.g., 'bubbles')",
    )
    parser.add_argument(
        "--big_backend",
        choices=["fb", "hf"],
        default="fb",
        help="Backend for big-prompt pass: fb=facebookresearch (default), hf=transformers",
    )
    parser.add_argument(
        "--text_prompt",
        default=None,
        help="Deprecated alias for --frst_text_prompt",
    )
    parser.add_argument(
        "--disable_pcs",
        action="store_true",
        help="Disable PCS text prompting (only use point prompts)",
    )
    parser.add_argument("--enable_candidates", action="store_true", help="Force enable candidate detection")
    parser.add_argument("--disable_candidates", action="store_true", help="Force disable candidate detection")
    parser.add_argument("--enable_tiling", action="store_true", help="Force enable tiling")
    parser.add_argument("--disable_tiling", action="store_true", help="Force disable tiling")
    parser.add_argument("--enable_hole_fill", action="store_true", help="Force enable hole filling")
    parser.add_argument("--disable_hole_fill", action="store_true", help="Force disable hole filling")
    parser.add_argument("--enable_consolidation", action="store_true", help="Force enable consolidation")
    parser.add_argument("--disable_consolidation", action="store_true", help="Force disable consolidation")
    parser.add_argument(
        "--output_mode",
        choices=["cutout", "overlay"],
        default=None,
        help="Output rendering mode (default: overlay)",
    )
    # FRST parameters (mirrors classical_test.py defaults)
    parser.add_argument("--r_min", type=int, default=4, help="Min radius (px)")
    parser.add_argument("--r_max", type=int, default=25, help="Max radius (px)")
    parser.add_argument("--r_step", type=int, default=2, help="Radius step")
    parser.add_argument("--alpha", type=float, default=1.4, help="Radial strictness (higher => dot-like)")
    parser.add_argument("--mag_percentile", type=float, default=88.0, help="Ignore gradients below this percentile")
    parser.add_argument("--peak_percentile", type=float, default=99.0, help="Keep peaks above this percentile")
    parser.add_argument("--nms_size", type=int, default=7, help="NMS neighborhood size (odd)")
    parser.add_argument("--border", type=int, default=8, help="Exclude peaks within this border (px)")
    parser.add_argument("--max_peaks", type=int, default=2000, help="Max centers (0 = no cap)")
    # Classical black-hat tiny-mask branch
    parser.add_argument("--enable_blackhat", action="store_true", help="Enable black-hat tiny-mask branch")
    parser.add_argument("--disable_blackhat", action="store_true", help="Disable black-hat tiny-mask branch")
    parser.add_argument("--blackhat_radius", type=int, default=None, help="Black-hat structuring element radius (px)")
    parser.add_argument(
        "--blackhat_percentile",
        type=float,
        default=None,
        help="Percentile threshold on black-hat response (e.g., 99.5)",
    )
    parser.add_argument("--blackhat_area_min", type=int, default=None, help="Min component area (px)")
    parser.add_argument("--blackhat_area_max", type=int, default=None, help="Max component area (px)")
    parser.add_argument("--blackhat_watershed", action="store_true", help="Enable watershed split for black-hat blobs")
    parser.add_argument("--blackhat_no_watershed", action="store_true", help="Disable watershed split for black-hat blobs")
    parser.add_argument(
        "--blackhat_watershed_min_area",
        type=int,
        default=None,
        help="Min area (px) to trigger watershed splitting",
    )
    parser.add_argument(
        "--blackhat_watershed_fg_thresh",
        type=float,
        default=None,
        help="Foreground threshold fraction for distance transform (0-1)",
    )
    parser.add_argument(
        "--blackhat_method",
        choices=["adaptive", "morph"],
        default="adaptive",
        help="Black-hat detection method: adaptive (default) or morph (legacy)",
    )
    return parser.parse_args()


def frst_centers(gray_u8: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    symmetry = frst_symmetry_map(
        gray_u8=gray_u8,
        r_min=args.r_min,
        r_max=args.r_max,
        r_step=args.r_step,
        alpha=args.alpha,
        mag_percentile=args.mag_percentile,
    )
    centers = pick_peaks(
        S=symmetry,
        peak_percentile=args.peak_percentile,
        nms_size=args.nms_size,
        border=args.border,
        max_peaks=args.max_peaks,
    )
    return centers, symmetry


def build_instances_from_masks(
    masks: Sequence[np.ndarray],
    scores: Sequence[Optional[float]],
    image_shape: Tuple[int, int],
    cfg: Dict[str, Any],
) -> List[Instance]:
    h, w = image_shape
    instances: List[Instance] = []
    for mask, score in zip(masks, scores):
        if mask is None:
            continue
        mask = resize_mask_to_shape(mask, (h, w))
        mask = fill_holes(mask, cfg)
        mask = maybe_convex_hull(mask, cfg)
        bbox = mask_bbox(mask)
        if bbox == (0, 0, 0, 0):
            continue
        x0, y0, x1, y1 = bbox
        cropped = mask[y0:y1, x0:x1]
        area = int(cropped.sum())
        if area == 0:
            continue
        instances.append(Instance(mask=cropped, score=score, area=area, bbox=bbox))
    return instances


def build_tile_instances_from_masks(
    masks: Sequence[np.ndarray],
    scores: Sequence[Optional[float]],
    tile_shape: Tuple[int, int],
    tile_origin: Tuple[int, int],
    image_shape: Tuple[int, int],
    cfg: Dict[str, Any],
    area_limit: int,
) -> Tuple[List[Instance], int]:
    tile_h, tile_w = tile_shape
    x0, y0 = tile_origin
    h, w = image_shape
    instances: List[Instance] = []
    discarded = 0
    for mask, score in zip(masks, scores):
        if mask is None:
            continue
        mask = resize_mask_to_shape(mask, (tile_h, tile_w))
        area_raw = int(mask.sum())
        if area_limit > 0 and area_raw > area_limit:
            discarded += 1
            continue
        mask = fill_holes(mask, cfg)
        mask = maybe_convex_hull(mask, cfg)
        bbox_local = mask_bbox(mask)
        if bbox_local == (0, 0, 0, 0):
            continue
        lx0, ly0, lx1, ly1 = bbox_local
        gx0 = max(x0 + lx0, 0)
        gy0 = max(y0 + ly0, 0)
        gx1 = min(x0 + lx1, w)
        gy1 = min(y0 + ly1, h)
        if gx1 <= gx0 or gy1 <= gy0:
            continue
        off_x0 = gx0 - (x0 + lx0)
        off_y0 = gy0 - (y0 + ly0)
        off_x1 = off_x0 + (gx1 - gx0)
        off_y1 = off_y0 + (gy1 - gy0)
        cropped = mask[ly0 + off_y0 : ly0 + off_y1, lx0 + off_x0 : lx0 + off_x1]
        area = int(cropped.sum())
        if area == 0:
            continue
        instances.append(Instance(mask=cropped, score=score, area=area, bbox=(gx0, gy0, gx1, gy1)))
    return instances, discarded


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


def _blackhat_global_cc(
    blackhat_u8: np.ndarray,
    area_min: int,
    area_max: int,
    enable_watershed: bool,
    watershed_min_area: int,
    watershed_fg_thresh: float,
) -> List[Instance]:
    bw_u8 = blackhat_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw_u8, connectivity=8)
    instances: List[Instance] = []

    for idx in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < area_min or area > area_max:
            continue
        comp_mask = labels[y : y + h, x : x + w] == idx
        if enable_watershed and area >= watershed_min_area:
            submasks = _watershed_split_component(
                comp_mask,
                blackhat_u8[y : y + h, x : x + w],
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
                    bbox=(x + lx0, y + ly0, x + lx1, y + ly1),
                )
            )

    return instances



def build_legacy_morph_instances(
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
    fallback_global_cc: bool,
) -> List[Instance]:
    """Legacy: Morphological Black-Hat + LoG/DoG blobs."""
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
                blackhat[oy : oy + comp_mask.shape[0], ox : ox + comp_mask.shape[1]],
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

    if instances or not fallback_global_cc:
        return instances

    thresh = float(np.percentile(blackhat, percentile))
    bw = blackhat >= thresh
    bw_u8 = (bw.astype(np.uint8) * 255)
    return _blackhat_global_cc(
        bw_u8,
        area_min=area_min,
        area_max=area_max,
        enable_watershed=enable_watershed,
        watershed_min_area=watershed_min_area,
        watershed_fg_thresh=watershed_fg_thresh,
    )


def build_adaptive_instances(
    gray_raw: np.ndarray,
    area_min: int = 20,
    area_max: int = 350,
    circularity_min: float = 0.4,
    solidity_min: float = 0.75,
    intensity_max: float = 160.0,
    watershed_min_area: int = 50, # Min area to try splitting
    watershed_fg_thresh: float = 0.5,
    debug_dir: Optional[str] = None,
) -> List[Instance]:
    """
    Adaptive Threshold detection (ported from detect_bubbles.py).
    Standard "Black Hat" / Small Bubble pipeline v2.
    Includes Watershed splitting for fused bubbles.
    """
    # 1. Preprocessing
    blur = cv2.GaussianBlur(gray_raw, (3,3), 0)
    
    # 2. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 31, 7)
    
    # 3. Refinement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "adaptive_thresh.png"), thresh)
        cv2.imwrite(os.path.join(debug_dir, "adaptive_open.png"), opening)
    
    # 4. Extraction & Filtering
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    instances: List[Instance] = []
    h, w = gray_raw.shape
    
    for c in cnts:
        # Initial loose area check (must be at least min_area)
        raw_area = cv2.contourArea(c)
        # Note: Upper bound check done AFTER splitting, but we can skip huge artifacts early
        if raw_area < area_min: continue 
        
        # Get bounding rect to crop
        x, y, cw, ch = cv2.boundingRect(c)
        
        # Create mask for this contour (relative to crop)
        c_mask_crop = np.zeros((ch, cw), dtype=np.uint8)
        # Offset contour to 0,0
        c_shifted = c - [x, y]
        cv2.drawContours(c_mask_crop, [c_shifted], -1, 255, -1)
        c_mask_bool = c_mask_crop > 0
        
        # Check if we should split
        submasks = [c_mask_bool]
        if raw_area >= watershed_min_area:
            gray_crop = gray_raw[y:y+ch, x:x+cw]
            submasks = _watershed_split_component(
                c_mask_bool,
                gray_crop,
                fg_thresh=watershed_fg_thresh
            )
            
        for submask in submasks:
            if not np.any(submask): continue
            
            # --- Per-Instance Filtering ---
            area = int(submask.sum())
            if area < area_min or area > area_max: continue
            
            # Reconstruct contour for geometric checks (Circularity/Solidity)
            # Need to find contours on the submask
            sub_u8 = (submask.astype(np.uint8) * 255)
            # Find external contours on the small mask
            sub_cnts, _ = cv2.findContours(sub_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not sub_cnts: continue
            sub_c = sub_cnts[0] # Should be one main component
            
            perimeter = cv2.arcLength(sub_c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < circularity_min: continue

            hull = cv2.convexHull(sub_c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0: continue
            solidity = float(area) / hull_area
            if solidity < solidity_min: continue

            # Intensity Filter
            # Calculate mean intensity on the original image crop using the submask
            # gray_raw crop needed again if not available
            gray_crop_sub = gray_raw[y:y+ch, x:x+cw]
            mean_val = cv2.mean(gray_crop_sub, mask=sub_u8)[0]
            if mean_val > intensity_max: continue

            # Build Instance
            # Need global bbox for this sub-component
            lx, ly, lcw, lch = cv2.boundingRect(sub_c)
            
            # Global coordinates
            gx = x + lx
            gy = y + ly
            
            # Final crop for instance (tight fit to sub-component)
            final_crop = submask[ly:ly+lch, lx:lx+lcw]
            
            instances.append(Instance(
                mask=final_crop,
                score=None, 
                area=int(area),
                bbox=(gx, gy, gx+lcw, gy+lch)
            ))

    return instances


def build_object_point_prompts(
    centers_xy: Sequence[Tuple[float, float]],
    tile_w: int,
    tile_h: int,
    knn_k: int,
    hex_radius: float,
) -> Tuple[List[List[Tuple[float, float]]], List[List[int]]]:
    """Build per-object (pos/neg) point prompts for SAM3 tracker.

    For each center:
      - 1 positive point at the center
      - kNN neighbors as negative points (k=3 requested)
      - 6 negative points on a hexagon of radius 1.5*r_max (out-of-tile points are ignored)
    """
    if not centers_xy:
        return [], []

    pts = np.asarray(centers_xy, dtype=np.float32)
    n = int(pts.shape[0])
    k = max(0, min(int(knn_k), max(0, n - 1)))

    objects_points: List[List[Tuple[float, float]]] = []
    objects_labels: List[List[int]] = []

    for i in range(n):
        cx = float(pts[i, 0])
        cy = float(pts[i, 1])
        obj_pts: List[Tuple[float, float]] = [(cx, cy)]
        obj_lbls: List[int] = [1]

        if k > 0:
            diff = pts - pts[i]
            d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
            d2[i] = np.inf
            nbr_idx = np.argpartition(d2, kth=k - 1)[:k]
            for j in nbr_idx:
                obj_pts.append((float(pts[j, 0]), float(pts[j, 1])))
                obj_lbls.append(0)

        for t in range(6):
            ang = float(t) * (math.pi / 3.0)
            hx = cx + float(hex_radius) * math.cos(ang)
            hy = cy + float(hex_radius) * math.sin(ang)
            if 0.0 <= hx < float(tile_w) and 0.0 <= hy < float(tile_h):
                obj_pts.append((float(hx), float(hy)))
                obj_lbls.append(0)

        objects_points.append(obj_pts)
        objects_labels.append(obj_lbls)

    return objects_points, objects_labels


def build_instances_from_pcs(
    masks: Sequence[np.ndarray],
    scores: Sequence[Optional[float]],
    boxes: Sequence[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int],
) -> List[Instance]:
    h, w = image_shape
    instances: List[Instance] = []
    for mask, score, box in zip(masks, scores, boxes):
        if mask is None:
            continue
        if mask.shape != (h, w):
            mask = resize_mask_to_shape(mask, (h, w))
        x0, y0, x1, y1 = box
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = mask[y0:y1, x0:x1]
        area = int(crop.sum())
        if area <= 0:
            continue
        instances.append(Instance(mask=crop, score=score, area=area, bbox=(x0, y0, x1, y1)))
    return instances


def apply_convex_hull(instances: List[Instance]) -> List[Instance]:
    """Apply convex hull to each instance mask (bbox-local).

    If hull inflates area by more than 2x, discard the instance.
    """
    try:
        from skimage.morphology import convex_hull_image
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for convex hull computation.") from exc

    kept: List[Instance] = []
    for inst in instances:
        if inst.mask is None:
            continue
        before_area = int(inst.mask.sum())
        if before_area <= 0:
            continue
        hull = convex_hull_image(inst.mask)
        after_area = int(hull.sum())
        if after_area > 1.5 * before_area:
            continue
        inst.mask = hull
        inst.area = after_area
        kept.append(inst)
    return kept


def derive_output_paths(output_path: str) -> Tuple[str, str, str, str]:
    base = Path(output_path)
    if not base.suffix:
        base = base.with_suffix(".png")
    stem = base.stem
    suffix = base.suffix
    frst_path = str(base.with_name(f"{stem}_frst{suffix}"))
    blackhat_path = str(base.with_name(f"{stem}_blackhat{suffix}"))
    big_path = str(base.with_name(f"{stem}_big{suffix}"))
    combined_path = str(base)
    return frst_path, blackhat_path, big_path, combined_path


def main() -> None:
    args = parse_args()

    ensure_hf_home()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    if args.text_prompt:
        args.frst_text_prompt = args.text_prompt
    cfg["sam"]["pcs_text_prompt"] = args.frst_text_prompt
    cfg["sam"]["pcs_enable"] = bool(cfg["sam"].get("pcs_enable", True)) and not args.disable_pcs

    output_path = resolve_output_path(args.input, args.output)
    frst_out, blackhat_out, big_out, combined_out = derive_output_paths(output_path)
    json_path, csv_path = resolve_output_paths(combined_out, cfg)
    ensure_output_dir(frst_out)
    ensure_output_dir(blackhat_out)
    ensure_output_dir(big_out)
    ensure_output_dir(combined_out)
    if json_path:
        ensure_output_dir(json_path)
    if csv_path:
        ensure_output_dir(csv_path)

    log_dir = resolve_logs_dir(args.input)
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    if args.seed is None:
        args.seed = int(cfg.get("seed", 0))
    set_deterministic_seed(int(args.seed))

    image = Image.open(args.input).convert("RGB")
    image_rgb = np.array(image, dtype=np.uint8)
    h, w = image_rgb.shape[:2]

    gray_raw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray_raw)

    centers, symmetry = frst_centers(gray, args)
    points_xy = [(float(x), float(y)) for x, y in centers]
    logger.info("FRST detected %d centers", len(points_xy))

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        sym_vis = (np.clip(symmetry, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(sym_vis, mode="L").save(os.path.join(args.debug_dir, "frst_symmetry.png"))
        if points_xy:
            save_candidate_viz(image_rgb, points_xy, os.path.join(args.debug_dir, "frst_centers.png"))

    frst_backend = None
    if not args.disable_candidates:
        if args.frst_backend == "fb":
            frst_backend = FrstPointBackendFB(cfg)
        else:
            frst_backend = FrstPointBackendHF(cfg)
        logger.info("FRST backend: %s", args.frst_backend)
    tile_size = max(1, int(round(8.0 * float(args.r_max))))
    tile_overlap = max(0, int(round(2.5 * float(args.r_max))))
    tile_cfg = {"tiling": {"tile_size": tile_size, "tile_overlap": tile_overlap}}
    tiles, (pad_bottom, pad_right) = create_tiles(h, w, tile_cfg)
    padded_rgb = pad_image(image_rgb, pad_bottom, pad_right, mode="reflect")
    logger.info(
        "Point-tiling enabled: tile_size=%d, overlap=%d, tiles=%d",
        tile_size,
        tile_overlap,
        len(tiles),
    )
    logger.info(
        "Negative prompts enabled: knn_k=%d, hex_radius=%.2f px",
        3,
        1.5 * float(args.r_max),
    )

    area_limit = int(round(16.0 * float(args.r_max) * float(args.r_max)))
    total_discarded = 0
    total_masks = 0
    frst_instances: List[Instance] = []

    # Only run FRST/SAM3 if candidate points exist and not explicitly disabled
    # (Actually we always run it if enabled, just might have 0 points)
    if not args.disable_candidates:
        if points_xy:
            centers_arr = np.array(points_xy, dtype=np.float32)
            xs = centers_arr[:, 0]
            ys = centers_arr[:, 1]
        else:
            xs = np.array([], dtype=np.float32)
            ys = np.array([], dtype=np.float32)

        for idx, (x0, y0, x1, y1) in enumerate(tiles):
            in_tile = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            if not in_tile.any():
                continue
            tile_points = [(float(x - x0), float(y - y0)) for x, y in zip(xs[in_tile], ys[in_tile])]
            tile_rgb = padded_rgb[y0:y1, x0:x1]
            tile_pil = Image.fromarray(tile_rgb)
            obj_points, obj_labels = build_object_point_prompts(
                tile_points,
                tile_w=int(tile_rgb.shape[1]),
                tile_h=int(tile_rgb.shape[0]),
                knn_k=3,
                hex_radius=1.5 * float(args.r_max),
            )
            masks, scores = frst_backend.segment_tile(tile_pil, obj_points, obj_labels)
            total_masks += len(masks)
            tile_instances, discarded = build_tile_instances_from_masks(
                masks,
                scores,
                (tile_rgb.shape[0], tile_rgb.shape[1]),
                (x0, y0),
                (h, w),
                cfg,
                area_limit,
            )
            total_discarded += discarded
            frst_instances.extend(tile_instances)
            logger.info(
                "Tile %d/%d: points=%d masks=%d discarded_large=%d",
                idx + 1,
                len(tiles),
                len(tile_points),
                len(masks),
                discarded,
            )

        logger.info(
            "Point pass totals: masks=%d discarded_large=%d (limit=%d px)",
            total_masks,
            total_discarded,
            area_limit,
        )

    # --- Black-hat (Adaptive vs Morph) Selection ---
    blackhat_cfg = cfg.get("blackhat", {})
    blackhat_enable = bool(blackhat_cfg.get("enable", True))
    if args.enable_blackhat:
        blackhat_enable = True
    if args.disable_blackhat:
        blackhat_enable = False
    
    # Check method (cli override? not yet standard, assume adaptive default logic)
    # The user asked for "adaptive" to be the main/default.
    # We will use the args or cfg to determine.
    # We'll use the 'blackhat_method' key if present in config, default to 'adaptive'.
    # CLI arg not explicit in parse_args above, let's just stick to logic:
    # If blackhat enabled, run Adaptive unless specifically configured for morph?
    # User said "The default should be adaptive + FRST + SAM. Nothing else."
    
    blackhat_instances: List[Instance] = []
    blackhat_method = args.blackhat_method # Use CLI argument
    
    if blackhat_enable:
        if blackhat_method == "adaptive":
            logger.info("Running Adaptive Threshold (new Black-hat) detection...")
            # Use parameters from detect_bubbles.py or cfg
            # detect_bubbles params: area 20-350, circ 0.4, sol 0.75, inten 160
            
            bh_area_min = 20
            bh_area_max = 350
            if args.blackhat_area_min is not None: bh_area_min = args.blackhat_area_min
            if args.blackhat_area_max is not None: bh_area_max = args.blackhat_area_max
            
            # Additional params could be exposed via config
            # min_circularity, min_solidity, max_intensity
            
            blackhat_instances = build_adaptive_instances(
                gray_raw,
                area_min=bh_area_min,
                area_max=bh_area_max,
                debug_dir=args.debug_dir
            )
            logger.info(f"Adaptive Threshold found {len(blackhat_instances)} instances.")
            
        else:
            # Legacy Path
            logger.info("Running Legacy Morphological Black-hat detection...")
            bh_radius = args.blackhat_radius if args.blackhat_radius is not None else int(
                blackhat_cfg.get("radius", 4)
            )
            bh_percentile = (
                args.blackhat_percentile
                if args.blackhat_percentile is not None
                else float(blackhat_cfg.get("percentile", 99.5))
            )
            bh_area_min = args.blackhat_area_min if args.blackhat_area_min is not None else int(
                blackhat_cfg.get("area_min", 8)
            )
            bh_area_max = args.blackhat_area_max if args.blackhat_area_max is not None else int(
                blackhat_cfg.get("area_max", 120)
            )
            bh_watershed = bool(blackhat_cfg.get("watershed", False))
            if args.blackhat_watershed:
                bh_watershed = True
            if args.blackhat_no_watershed:
                bh_watershed = False

            expected_area = math.pi * float(max(1, bh_radius)) ** 2
            default_ws_min = int(round(2.0 * expected_area))
            bh_ws_min_area = args.blackhat_watershed_min_area if args.blackhat_watershed_min_area is not None else int(
                blackhat_cfg.get("watershed_min_area", default_ws_min) or default_ws_min
            )
            bh_ws_fg = (
                args.blackhat_watershed_fg_thresh
                if args.blackhat_watershed_fg_thresh is not None
                else float(blackhat_cfg.get("watershed_fg_thresh", 0.5))
            )
            bh_blob_method = str(blackhat_cfg.get("blob_method", "dog"))
            bh_blob_min_sigma = float(blackhat_cfg.get("blob_min_sigma", 1.3))
            bh_blob_max_sigma = float(blackhat_cfg.get("blob_max_sigma", 2.8))
            bh_blob_num_sigma = int(blackhat_cfg.get("blob_num_sigma", 10))
            bh_blob_threshold = float(blackhat_cfg.get("blob_threshold", 0.01))
            bh_blob_overlap = float(blackhat_cfg.get("blob_overlap", 0.5))
            bh_patch_radius = int(blackhat_cfg.get("patch_radius", 6))
            bh_patch_percentile = float(blackhat_cfg.get("patch_percentile", 90.0))
            bh_patch_otsu = bool(blackhat_cfg.get("patch_use_otsu", False))
            bh_fallback_cc = bool(blackhat_cfg.get("fallback_global_cc", False))

            blackhat_instances = build_legacy_morph_instances(
                gray_raw,
                radius=bh_radius,
                percentile=bh_percentile,
                area_min=bh_area_min,
                area_max=bh_area_max,
                enable_watershed=bh_watershed,
                watershed_min_area=bh_ws_min_area,
                watershed_fg_thresh=bh_ws_fg,
                blob_method=bh_blob_method,
                blob_min_sigma=bh_blob_min_sigma,
                blob_max_sigma=bh_blob_max_sigma,
                blob_num_sigma=bh_blob_num_sigma,
                blob_threshold=bh_blob_threshold,
                blob_overlap=bh_blob_overlap,
                patch_radius=bh_patch_radius,
                patch_percentile=bh_patch_percentile,
                patch_use_otsu=bh_patch_otsu,
                fallback_global_cc=bh_fallback_cc,
            )
            logger.info(
                "Legacy Black-hat masks: %d (radius=%d, pct=%.2f, area=[%d,%d], watershed=%s, blob=%s)",
                len(blackhat_instances),
                bh_radius,
                bh_percentile,
                bh_area_min,
                bh_area_max,
                "on" if bh_watershed else "off",
                bh_blob_method,
            )

    pcs_backend: Optional[Sam3ConceptBackend] = None
    if cfg["sam"].get("pcs_enable", True) and args.frst_text_prompt:
        pcs_backend = Sam3ConceptBackend(cfg["device"], cfg["sam"])
        pcs_threshold = float(cfg["sam"].get("pcs_threshold", 0.9))
        pcs_mask_threshold = float(cfg["sam"].get("pcs_mask_threshold", 0.9))
        pcs_masks, pcs_scores, pcs_boxes = pcs_backend.segment_by_text(
            image, args.frst_text_prompt, pcs_threshold, pcs_mask_threshold
        )
        logger.info("PCS produced %d micro-text masks", len(pcs_masks))
        frst_instances.extend(
            build_instances_from_pcs(pcs_masks, pcs_scores, pcs_boxes, (h, w))
        )

    big_instances: List[Instance] = []
    if cfg["sam"].get("pcs_enable", True) and args.big_text_prompt:
        if args.big_backend == "hf":
            from big_bubble_prompt_hf import run_big_prompt as run_big_prompt_hf
            big_instances = run_big_prompt_hf(image, args.big_text_prompt, cfg)
        else:
            from big_bubble_prompt_fb import run_big_prompt as run_big_prompt_fb
            big_instances = run_big_prompt_fb(image, args.big_text_prompt, cfg)
        logger.info("Big-prompt (%s) produced %d masks", args.big_backend, len(big_instances))

    frst_instances = apply_convex_hull(frst_instances)

    combined_instances = consolidate_instances(
        frst_instances + blackhat_instances + big_instances, cfg, (h, w)
    )
    logger.info(
        "Instances: FRST=%d, ADAPTIVE=%d, BIG=%d, COMBINED=%d",
        len(frst_instances),
        len(blackhat_instances),
        len(big_instances),
        len(combined_instances),
    )

    output_mode = str(cfg["output"].get("output_mode", "overlay")).lower()
    def _save_instances(instances: List[Instance], out_path: str, label: str) -> None:
        if output_mode == "cutout":
            rgba = build_rgba_cutout(image_rgb, instances)
            Image.fromarray(rgba, mode="RGBA").save(out_path)
            logger.info("Saved %s cutout to %s", label, out_path)
        else:
            overlay = build_rgba_overlay(
                image_rgb,
                instances,
                alpha=int(cfg["output"].get("overlay_alpha", 128)),
                colormap=cfg["output"].get("overlay_colormap", "tab20"),
            )
            Image.fromarray(overlay, mode="RGBA").save(out_path)
            logger.info("Saved %s overlay to %s", label, out_path)

    _save_instances(frst_instances, frst_out, "FRST")
    _save_instances(blackhat_instances, blackhat_out, "ADAPTIVE")
    _save_instances(big_instances, big_out, "BIG")
    _save_instances(combined_instances, combined_out, "COMBINED")

    save_instance_outputs(
        combined_instances,
        args.input,
        (h, w),
        json_path,
        csv_path,
        include_rle=bool(cfg["output"].get("include_rle", False)),
    )
    if json_path:
        logger.info("Saved JSON to %s", json_path)
    if csv_path:
        logger.info("Saved CSV to %s", csv_path)


if __name__ == "__main__":
    main()

