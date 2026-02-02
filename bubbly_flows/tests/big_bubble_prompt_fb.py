#!/usr/bin/env python3
"""
Facebookresearch SAM3 text-prompted segmentation for big bubbles.

This mirrors tests/test.py behavior (no cropping by default), using sam3's Sam3Processor.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as exc:
    raise RuntimeError("sam3 package is required for facebookresearch backend.") from exc

from bubble_sam3.config import ensure_hf_home, load_config
from bubble_sam3.outputs import build_rgba_cutout, build_rgba_overlay, ensure_output_dir
from bubble_sam3.postprocess import Instance, mask_bbox

_MODEL_CACHE = None
_PROCESSOR_CACHE: Dict[float, Any] = {}


def _get_processor(confidence_threshold: float):
    global _MODEL_CACHE, _PROCESSOR_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = build_sam3_image_model()
    if confidence_threshold not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[confidence_threshold] = Sam3Processor(
            _MODEL_CACHE, confidence_threshold=confidence_threshold
        )
    return _PROCESSOR_CACHE[confidence_threshold]


def _instances_from_masks(
    masks: Any, boxes: Any, scores: Any, image_shape: Tuple[int, int]
) -> List[Instance]:
    h, w = image_shape
    instances: List[Instance] = []
    for mask, box, score in zip(masks, boxes, scores):
        if mask is None:
            continue
        if hasattr(mask, "cpu"):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        mask_bool = mask_np.astype(bool)

        if box is not None and getattr(box, "shape", None) is not None:
            if hasattr(box, "cpu"):
                box_np = box.cpu().numpy()
            else:
                box_np = np.array(box)
            x0, y0, x1, y1 = [int(v) for v in box_np[:4]]
        else:
            x0, y0, x1, y1 = mask_bbox(mask_bool)
            if (x1 - x0) <= 0 or (y1 - y0) <= 0:
                continue

        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            continue

        crop = mask_bool[y0:y1, x0:x1]
        area = int(crop.sum())
        if area <= 0:
            continue
        score_val = float(score.cpu().item()) if hasattr(score, "cpu") else float(score)
        instances.append(Instance(mask=crop, score=score_val, area=area, bbox=(x0, y0, x1, y1)))
    return instances


def run_big_prompt(image: Image.Image, prompt: str, cfg: Dict[str, Any]) -> List[Instance]:
    confidence_threshold = float(
        cfg["sam"].get("pcs_threshold", cfg["sam"].get("confidence_threshold", 0.4))
    )
    processor = _get_processor(confidence_threshold)
    state = processor.set_image(image)
    out = processor.set_text_prompt(state=state, prompt=prompt)
    masks, boxes, scores = out.get("masks", []), out.get("boxes", []), out.get("scores", [])
    return _instances_from_masks(masks, boxes, scores, image.size[::-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="facebookresearch SAM3 big-bubble prompt")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output RGBA PNG path")
    parser.add_argument("--prompt", default="bubbles", help="Text prompt")
    parser.add_argument("--config", default=None, help="Optional JSON/JSONC config path")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device override")
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
