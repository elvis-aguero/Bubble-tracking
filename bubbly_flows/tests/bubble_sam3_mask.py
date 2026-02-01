"""
Fully automatic SAM3-based bubble segmentation pipeline.

Structural decisions:
- Tiling is used instead of global upsampling so small (~10 px) bubbles occupy more pixels in the model input after SAM3's internal resize; this improves small-object visibility without global memory blowups.
- SAM3 resizes inputs to a fixed encoder size, so the apparent bubble scale depends on the tile size; smaller tiles effectively enlarge bubbles in the resized space and help the point-prompted model resolve rims.

Candidate points:
- One positive point is generated per candidate bubble center (LoG/DoG/Hough) and points are batched for SAM3 tracker inference to avoid OOM/latency spikes.

Consolidation:
- Masks are filtered by area, border exclusion, and optional circularity/solidity; then deduplicated using IoU/containment rules, dropping smaller or lower-score masks when they substantially overlap.

Known failure modes and tuning:
- Very faint rims or low contrast reduce candidate detection; increase contrast normalization, lower blob thresholds, or enable tiling to boost effective scale.
- Overlapping bubbles may merge into one mask; tighten IoU/containment thresholds, enable convex hull for stability, or increase tile overlap.
- Masks truncated at tile edges can be dropped by min_coverage_for_keep; raise tile overlap or reduce that threshold if too many masks are discarded.

Assumptions:
- Bubbles are approximately circular with darker rims on a light background; optional convex-hull conversion supports downstream area estimation under this morphology.

ENVIRONMENT SETUP (copy-paste for this environment):
    export HF_HOME=/users/eaguerov/scratch/hf
    export CUDA_VISIBLE_DEVICES=0  # if needed
    interact -q gpu -g 1 -n 4 -t 01:00:00 -m 16g
    eval "$(mamba shell hook --shell bash)"


USAGE:
    python bubble_sam3_mask.py --input <image> --output <output_png> [options]

Examples:
    # Basic usage with required arguments
    python bubble_sam3_mask.py --input image.jpg --output result.png

    # With custom configuration file
    python bubble_sam3_mask.py --input image.jpg --output result.png --config config.jsonc

    # Using GPU and enabling debug output
    python bubble_sam3_mask.py --input image.jpg --output result.png --device cuda --debug_dir ./debug

    # Override specific pipeline components
    python bubble_sam3_mask.py --input image.jpg --output result.png --disable_candidates --enable_tiling

    # Use overlay output mode instead of cutout
    python bubble_sam3_mask.py --input image.jpg --output result.png --output_mode overlay

Arguments:
    --input PATH              Input image path (required)
    --output PATH             Output RGBA PNG path (required)
    --config PATH             Optional JSON/JSONC config path (uses defaults if omitted)
    --device DEVICE           Compute device: 'cuda' or 'cpu' (overrides config)
    --seed SEED               Random seed for reproducibility (overrides config)
    --debug_dir DIR           Directory to save debug PNGs for visualization
    --output_json PATH        Optional per-instance JSON output path (defaults to output.png -> output.json)
    --no_output_json          Disable per-instance JSON output
    --output_csv PATH         Optional per-instance CSV output path
    --include_rle             Include COCO RLE masks in JSON (requires pycocotools)
    --sam_model NAME          Override SAM3 model name/id (default: facebook/sam3)
    --points_per_batch N      Points per SAM3 tracker batch (tune for GPU memory)
    --multimask_output        Enable multiple masks per point (default: false)
    --allow_download          Allow SAM3 model downloads if cache miss

Pipeline Control Flags:
    --enable_candidates       Force enable candidate point detection
    --disable_candidates      Force disable candidate point detection
    --enable_tiling           Force enable tiling strategy
    --disable_tiling          Force disable tiling strategy
    --enable_hole_fill        Force enable hole filling
    --disable_hole_fill       Force disable hole filling
    --enable_consolidation    Force enable mask consolidation/deduplication
    --disable_consolidation   Force disable mask consolidation/deduplication
    --output_mode MODE        Output rendering mode: 'cutout' (RGBA with alpha) or 'overlay' (colored masks)

Output:
    - RGBA PNG containing segmented bubbles
    - Per-instance JSON (area, centroid, bbox, radius) for downstream measurements
    - If --debug_dir specified, also saves intermediate debug images (tiles, candidates, masks, etc.)

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bubble_sam3.backend import Sam3ConceptBackend, Sam3PointBackend, TransformersMaskGenerator
from bubble_sam3.config import apply_cli_overrides, ensure_hf_home, load_config
from bubble_sam3.outputs import (
    build_rgba_cutout,
    build_rgba_overlay,
    encode_mask_rle,
    ensure_output_dir,
    resolve_output_paths,
    save_debug_outputs,
    save_instance_outputs,
)
from bubble_sam3.pipeline import run_pipeline

try:
    import torch
except ImportError as exc:
    raise RuntimeError("torch is required. Install with: pip install torch") from exc


def resolve_output_path(input_path: str, output_arg: str | None) -> str:
    """Resolve output path relative to the input image by default.

    Rules:
    - If --output is omitted: write to <input_dir>/output/<input_stem>_mask.png
    - If --output is relative with no directory component: write to <input_dir>/output/<output_arg>
    - If --output is relative with directories: interpret it relative to <input_dir>
    - If --output is absolute: use it as-is.
    """
    in_path = Path(input_path).expanduser().resolve()
    base_dir = in_path.parent
    out_dir = base_dir / "output"

    if not output_arg:
        return str(out_dir / f"{in_path.stem}_mask.png")

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
    """Configure logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"bubble_sam3_mask_{timestamp}.log")

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

    logger.info(f"Logging initialized. Log file: {log_file}")
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
    parser = argparse.ArgumentParser(description="SAM3 bubble segmentation pipeline")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output RGBA PNG path. If omitted, defaults to <input_dir>/output/<input_stem>_mask.png. "
            "If a bare filename is provided, it is written into <input_dir>/output/."
        ),
    )
    parser.add_argument("--config", default=None, help="Optional JSON/JSONC config path")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--debug_dir", default=None, help="Directory for debug PNGs")
    parser.add_argument("--output_json", default=None, help="Optional per-instance JSON output path")
    parser.add_argument("--no_output_json", action="store_true", help="Disable per-instance JSON output")
    parser.add_argument("--output_csv", default=None, help="Optional per-instance CSV output path")
    parser.add_argument("--include_rle", action="store_true", help="Include COCO RLE masks in JSON (pycocotools)")

    parser.add_argument("--sam_model", default=None, help="Override SAM3 model name/id")
    parser.add_argument("--points_per_batch", type=int, default=None, help="SAM3 points per batch")
    parser.add_argument("--multimask_output", action="store_true", help="Enable SAM3 multimask output per point")
    parser.add_argument("--allow_download", action="store_true", help="Allow SAM3 model downloads if cache miss")

    parser.add_argument("--enable_candidates", action="store_true", help="Enable candidate point detection")
    parser.add_argument("--disable_candidates", action="store_true", help="Disable candidate point detection")
    parser.add_argument("--enable_tiling", action="store_true", help="Enable tiling")
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling")
    parser.add_argument("--enable_hole_fill", action="store_true", help="Enable hole filling")
    parser.add_argument("--disable_hole_fill", action="store_true", help="Disable hole filling")
    parser.add_argument("--enable_consolidation", action="store_true", help="Enable consolidation")
    parser.add_argument("--disable_consolidation", action="store_true", help="Disable consolidation")
    parser.add_argument("--output_mode", choices=["cutout", "overlay"], default=None, help="Output mode")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_hf_home()

    output_path = resolve_output_path(args.input, args.output)
    log_dir = resolve_logs_dir(args.input)
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("SAM3 Bubble Segmentation Pipeline Started")
    logger.info("=" * 80)
    logger.info(f"Input image: {args.input}")
    logger.info(f"Output path: {output_path}")
    if args.config:
        logger.info(f"Config file: {args.config}")
    logger.info(f"Device: {args.device or 'default'}")

    if not os.path.exists(args.input):
        logger.error(f"Input not found: {args.input}")
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    logger.info("Loading configuration...")
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    logger.debug(f"Configuration: {json.dumps(cfg, indent=2, default=str)}")

    # Interpret debug_dir relative to the input image by default.
    if cfg["debug"].get("debug_dir"):
        dbg = Path(str(cfg["debug"]["debug_dir"])).expanduser()
        if not dbg.is_absolute():
            cfg["debug"]["debug_dir"] = str(Path(args.input).expanduser().resolve().parent / dbg)

    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
        cfg["device"] = "cpu"

    logger.info(f"Setting random seed to {cfg['seed']}")
    set_deterministic_seed(int(cfg["seed"]))

    logger.info(f"Loading image from {args.input}")
    image = Image.open(args.input).convert("RGB")
    logger.info(f"Image size: {image.size}")

    logger.info(f"Loading SAM3 tracker model on {device}...")
    sam_backend = Sam3PointBackend(device, cfg["sam"])
    logger.info("SAM3 tracker model loaded successfully.")
    pcs_backend: Sam3ConceptBackend | None = None
    if cfg["sam"].get("pcs_enable", True):
        logger.info(f"Loading SAM3 PCS model on {device}...")
        pcs_backend = Sam3ConceptBackend(device, cfg["sam"])
        logger.info("SAM3 PCS model loaded successfully.")
    fallback_generator: TransformersMaskGenerator | None = None

    logger.info("Running segmentation pipeline...")
    instances, debug, fallback_generator = run_pipeline(
        image, cfg, sam_backend, fallback_generator, pcs_backend
    )
    logger.info(f"Pipeline completed. Found {len(instances)} bubble instances.")

    image_rgb = np.array(image)

    logger.info(f"Building output with {len(instances)} masks...")
    output_mode = cfg["output"].get("output_mode", "cutout")
    if output_mode == "cutout":
        logger.info("Using cutout output mode (RGBA with alpha channel)")
        rgba = build_rgba_cutout(image_rgb, instances)
    elif output_mode == "overlay":
        logger.info("Using overlay output mode (colored mask overlay)")
        rgba = build_rgba_overlay(
            image_rgb,
            instances,
            alpha=int(cfg["output"].get("overlay_alpha", 128)),
            colormap=cfg["output"].get("overlay_colormap", "tab20"),
        )
    else:
        logger.error(f"Unknown output_mode: {output_mode}")
        raise ValueError(f"Unknown output_mode: {output_mode}")

    logger.info(f"Saving output to {output_path}")
    ensure_output_dir(output_path)
    Image.fromarray(rgba, mode="RGBA").save(output_path)
    logger.info("Output saved successfully.")

    json_path, csv_path = resolve_output_paths(output_path, cfg)
    ensure_output_dir(json_path)
    ensure_output_dir(csv_path)
    include_rle = bool(cfg["output"].get("include_rle", False))
    if include_rle and encode_mask_rle(np.zeros((1, 1), dtype=bool)) is None:
        logger.warning("include_rle requested but pycocotools is not available; skipping RLE export.")
        include_rle = False
    if json_path:
        logger.info(f"Saving instance JSON to {json_path}")
    if csv_path:
        logger.info(f"Saving instance CSV to {csv_path}")
    save_instance_outputs(
        instances,
        args.input,
        image_rgb.shape[:2],
        json_path,
        csv_path,
        include_rle,
    )

    logger.info("Saving debug outputs...")
    save_debug_outputs(debug, image_rgb, instances, cfg)

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
