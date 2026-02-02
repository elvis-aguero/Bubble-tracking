#!/usr/bin/env python3
"""
HF transformers SAM3 text-prompted segmentation for big bubbles.

This uses the transformers Sam3Model/Sam3Processor pipeline (same as bubble_sam3 backend).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from bubble_sam3.backend import Sam3ConceptBackend
from bubble_sam3.config import ensure_hf_home, load_config
from bubble_sam3.outputs import build_rgba_cutout, build_rgba_overlay, ensure_output_dir
from bubble_sam3.postprocess import Instance, resize_mask_to_shape


def build_instances_from_pcs(
    masks: List[np.ndarray],
    scores: List[Optional[float]],
    boxes: List[Tuple[int, int, int, int]],
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


def run_big_prompt(image: Image.Image, prompt: str, cfg: Dict[str, Any]) -> List[Instance]:
    backend = Sam3ConceptBackend(cfg["device"], cfg["sam"])
    pcs_threshold = float(cfg["sam"].get("pcs_threshold", 0.5))
    pcs_mask_threshold = float(cfg["sam"].get("pcs_mask_threshold", 0.5))
    masks, scores, boxes = backend.segment_by_text(image, prompt, pcs_threshold, pcs_mask_threshold)
    return build_instances_from_pcs(masks, scores, boxes, image.size[::-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF SAM3 big-bubble prompt")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output RGBA PNG path")
    parser.add_argument("--prompt", default="bubbles", help="Text prompt")
    parser.add_argument("--config", default=None, help="Optional JSON/JSONC config path")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device override")
    parser.add_argument("--allow_download", action="store_true", help="Allow SAM3 model downloads")
    parser.add_argument(
        "--output_mode",
        choices=["cutout", "overlay"],
        default="overlay",
        help="Output rendering mode (default: overlay)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_hf_home()
    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    if args.allow_download:
        cfg["sam"]["local_files_only"] = False

    image = Image.open(args.input).convert("RGB")
    instances = run_big_prompt(image, args.prompt, cfg)

    image_rgb = np.array(image, dtype=np.uint8)
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
