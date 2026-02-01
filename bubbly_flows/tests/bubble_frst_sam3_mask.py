#!/usr/bin/env python3
"""
Combine FRST bubble-center detection with SAM3 point prompting.

Workflow:
  1) Detect bubble centers with FRST (from classical_test.py).
  2) Use centers as positive points for SAM3 tracker segmentation.
  3) Optionally run PCS text prompting (e.g., "tiny bubbles") to add masks.
  4) Consolidate masks and write an overlay output with per-bubble colors.
"""

from __future__ import annotations

import argparse
import logging
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
from bubble_sam3.backend import Sam3ConceptBackend, Sam3PointBackend, segment_with_points
from bubble_sam3.config import apply_cli_overrides, ensure_hf_home, load_config
from bubble_sam3.outputs import (
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
        "--text_prompt",
        default="tiny bubbles",
        help="PCS text prompt (e.g., 'tiny bubbles' or 'micro-sized bubbles')",
    )
    parser.add_argument(
        "--disable_pcs",
        action="store_true",
        help="Disable PCS text prompting (only use point prompts)",
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


def main() -> None:
    args = parse_args()

    ensure_hf_home()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg["sam"]["pcs_text_prompt"] = args.text_prompt
    cfg["sam"]["pcs_enable"] = bool(cfg["sam"].get("pcs_enable", True)) and not args.disable_pcs

    output_path = resolve_output_path(args.input, args.output)
    json_path, csv_path = resolve_output_paths(output_path, cfg)
    ensure_output_dir(output_path)
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
    masks, scores = segment_with_points(sam_backend, image, points_xy)
    logger.info("SAM3 produced %d point-prompted masks", len(masks))

    instances = build_instances_from_masks(masks, scores, (h, w), cfg)

    if cfg["sam"].get("pcs_enable", True) and args.text_prompt:
        pcs_backend = Sam3ConceptBackend(cfg["device"], cfg["sam"])
        pcs_threshold = float(cfg["sam"].get("pcs_threshold", 0.5))
        pcs_mask_threshold = float(cfg["sam"].get("pcs_mask_threshold", 0.5))
        pcs_masks, pcs_scores, pcs_boxes = pcs_backend.segment_by_text(
            image, args.text_prompt, pcs_threshold, pcs_mask_threshold
        )
        logger.info("PCS produced %d text-prompted masks", len(pcs_masks))

        pcs_instances: List[Instance] = []
        for mask, score, box in zip(pcs_masks, pcs_scores, pcs_boxes):
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
            pcs_instances.append(Instance(mask=crop, score=score, area=area, bbox=(x0, y0, x1, y1)))
        instances.extend(pcs_instances)

    logger.info("Total instances before consolidation: %d", len(instances))
    instances = consolidate_instances(instances, cfg, (h, w))
    logger.info("Total instances after consolidation: %d", len(instances))

    overlay = build_rgba_overlay(
        image_rgb,
        instances,
        alpha=int(cfg["output"].get("overlay_alpha", 128)),
        colormap=cfg["output"].get("overlay_colormap", "tab20"),
    )
    Image.fromarray(overlay, mode="RGBA").save(output_path)
    logger.info("Saved overlay to %s", output_path)

    save_instance_outputs(
        instances,
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
