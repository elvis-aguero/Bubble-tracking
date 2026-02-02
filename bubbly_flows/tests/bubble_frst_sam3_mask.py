#!/usr/bin/env python3
"""
Combine FRST bubble-center detection with SAM3 point prompting.

Workflow:
  1) Detect bubble centers with FRST (from classical_test.py).
  2) Use centers as positive points for SAM3 tracker segmentation.
  3) Optionally run PCS text prompting (e.g., "tiny bubbles") to add masks.
  4) Produce three outputs:
     - FRST + micro prompt masks
     - Big-bubble prompt masks only
     - Consolidated masks from both pipelines

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
    --frst_text_prompt --big_text_prompt --big_backend --text_prompt --disable_pcs
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

Notes:
  - Default output is an overlay with per-instance colors.
  - output_mode=cutout produces an RGBA cutout (alpha = union of masks).
  - Output files are derived from --output:
      <stem>_frst.png, <stem>_big.png, and the combined result at --output.
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
from bubble_sam3.backend import Sam3ConceptBackend, Sam3PointBackend, segment_with_object_points
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
from big_bubble_prompt_fb import run_big_prompt as run_big_prompt_fb
from big_bubble_prompt_hf import run_big_prompt as run_big_prompt_hf

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
        if after_area > 2 * before_area:
            continue
        inst.mask = hull
        inst.area = after_area
        kept.append(inst)
    return kept


def derive_output_paths(output_path: str) -> Tuple[str, str, str]:
    base = Path(output_path)
    if not base.suffix:
        base = base.with_suffix(".png")
    stem = base.stem
    suffix = base.suffix
    frst_path = str(base.with_name(f"{stem}_frst{suffix}"))
    big_path = str(base.with_name(f"{stem}_big{suffix}"))
    combined_path = str(base)
    return frst_path, big_path, combined_path


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
    frst_out, big_out, combined_out = derive_output_paths(output_path)
    json_path, csv_path = resolve_output_paths(combined_out, cfg)
    ensure_output_dir(frst_out)
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

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    centers, symmetry = frst_centers(gray, args)
    points_xy = [(float(x), float(y)) for x, y in centers]
    logger.info("FRST detected %d centers", len(points_xy))

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        sym_vis = (np.clip(symmetry, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(sym_vis, mode="L").save(os.path.join(args.debug_dir, "frst_symmetry.png"))
        if points_xy:
            save_candidate_viz(image_rgb, points_xy, os.path.join(args.debug_dir, "frst_centers.png"))

    sam_backend = Sam3PointBackend(cfg["device"], cfg["sam"])
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
        masks, scores = segment_with_object_points(sam_backend, tile_pil, obj_points, obj_labels)
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

    pcs_backend: Optional[Sam3ConceptBackend] = None
    if cfg["sam"].get("pcs_enable", True) and args.frst_text_prompt:
        pcs_backend = Sam3ConceptBackend(cfg["device"], cfg["sam"])
        pcs_threshold = float(cfg["sam"].get("pcs_threshold", 0.6))
        pcs_mask_threshold = float(cfg["sam"].get("pcs_mask_threshold", 0.6))
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
            big_instances = run_big_prompt_hf(image, args.big_text_prompt, cfg)
        else:
            big_instances = run_big_prompt_fb(image, args.big_text_prompt, cfg)
        logger.info("Big-prompt (%s) produced %d masks", args.big_backend, len(big_instances))

    frst_instances = apply_convex_hull(frst_instances)

    combined_instances = consolidate_instances(
        frst_instances + big_instances, cfg, (h, w)
    )
    logger.info(
        "Instances: FRST=%d, BIG=%d, COMBINED=%d",
        len(frst_instances),
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
