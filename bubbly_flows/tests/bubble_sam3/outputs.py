from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .postprocess import Instance

TAB20_COLORS: List[Tuple[int, int, int]] = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
]
CANDIDATE_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "log": (255, 80, 80, 255),
    "dog": (80, 255, 120, 255),
    "hough": (80, 120, 255, 255),
}


def ensure_output_dir(path: Optional[str]) -> None:
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def resolve_output_paths(output_path: str, cfg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    output = Path(output_path)
    json_path: Optional[str] = None
    csv_path: Optional[str] = None

    write_json = cfg["output"].get("write_json", True)
    write_csv = cfg["output"].get("write_csv", False) or cfg["output"].get("csv_path") is not None

    if write_json:
        json_path = cfg["output"].get("json_path") or str(output.with_suffix(".json"))
    if write_csv:
        csv_path = cfg["output"].get("csv_path") or str(output.with_suffix(".csv"))

    return json_path, csv_path


def build_rgba_cutout(image_rgb: np.ndarray, instances: List[Instance]) -> np.ndarray:
    h, w, _ = image_rgb.shape
    alpha = np.zeros((h, w), dtype=np.uint8)
    for inst in instances:
        x0, y0, x1, y1 = inst.bbox
        patch = alpha[y0:y1, x0:x1]
        patch[inst.mask] = 255
        alpha[y0:y1, x0:x1] = patch
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = alpha
    return rgba


def build_rgba_overlay(
    image_rgb: np.ndarray, instances: List[Instance], alpha: int, colormap: str
) -> np.ndarray:
    h, w, _ = image_rgb.shape
    out = image_rgb.astype(np.float32).copy()
    colors = TAB20_COLORS if colormap == "tab20" else TAB20_COLORS
    a = float(np.clip(alpha, 0, 255)) / 255.0
    for i, inst in enumerate(instances):
        color = np.array(colors[i % len(colors)], dtype=np.float32)
        x0, y0, x1, y1 = inst.bbox
        patch = out[y0:y1, x0:x1]
        patch[inst.mask] = (1.0 - a) * patch[inst.mask] + a * color
        out[y0:y1, x0:x1] = patch
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(out, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = 255
    return rgba


def mask_centroid(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return float(bbox[0]), float(bbox[1])
    return float(xs.mean() + bbox[0]), float(ys.mean() + bbox[1])


def materialize_full_mask(
    mask: np.ndarray, bbox: Tuple[int, int, int, int], shape: Tuple[int, int]
) -> np.ndarray:
    full = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = bbox
    full[y0:y1, x0:x1] = mask
    return full


def encode_mask_rle(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    try:
        from pycocotools import mask as mask_util
    except ImportError:
        return None
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def save_instance_outputs(
    instances: List[Instance],
    image_path: str,
    image_shape: Tuple[int, int],
    output_json: Optional[str],
    output_csv: Optional[str],
    include_rle: bool,
) -> None:
    if not output_json and not output_csv:
        return

    h, w = image_shape
    records: List[Dict[str, Any]] = []
    for idx, inst in enumerate(instances):
        cx, cy = mask_centroid(inst.mask, inst.bbox)
        record: Dict[str, Any] = {
            "id": idx,
            "score": float(inst.score) if inst.score is not None else None,
            "area_px": int(inst.area),
            "bbox_xyxy": [int(v) for v in inst.bbox],
            "centroid_xy": [cx, cy],
            "radius_equiv_px": float(math.sqrt(inst.area / math.pi)) if inst.area > 0 else 0.0,
        }
        if include_rle:
            full_mask = materialize_full_mask(inst.mask, inst.bbox, (h, w))
            rle = encode_mask_rle(full_mask)
            if rle is not None:
                record["mask_rle"] = rle
        records.append(record)

    if output_json:
        payload = {
            "image_path": image_path,
            "image_size": [w, h],
            "instances": records,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if output_csv:
        fieldnames = [
            "id",
            "score",
            "area_px",
            "bbox_x0",
            "bbox_y0",
            "bbox_x1",
            "bbox_y1",
            "centroid_x",
            "centroid_y",
            "radius_equiv_px",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                bbox = record["bbox_xyxy"]
                centroid = record["centroid_xy"]
                writer.writerow(
                    {
                        "id": record["id"],
                        "score": record["score"],
                        "area_px": record["area_px"],
                        "bbox_x0": bbox[0],
                        "bbox_y0": bbox[1],
                        "bbox_x1": bbox[2],
                        "bbox_y1": bbox[3],
                        "centroid_x": centroid[0],
                        "centroid_y": centroid[1],
                        "radius_equiv_px": record["radius_equiv_px"],
                    }
                )


def save_candidate_viz(image_rgb: np.ndarray, points: List[Tuple[float, float]], out_path: str) -> None:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x, y in points:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0))
    img.save(out_path)


def draw_points_on_rgba(
    rgba: np.ndarray, points_by_method: Dict[str, List[Tuple[float, float]]], radius: int = 3
) -> np.ndarray:
    img = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(img)
    r = max(1, int(radius))
    for method, points in points_by_method.items():
        color = CANDIDATE_COLORS.get(method, (255, 255, 0, 255))
        for x, y in points:
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=1)
    return np.array(img)


def save_tile_grid_viz(image_rgb: np.ndarray, tiles: List[Tuple[int, int, int, int]], out_path: str) -> None:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in tiles:
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0))
    img.save(out_path)


def save_hole_fill_viz(before: np.ndarray, after: np.ndarray, out_path: str) -> None:
    left = (before.astype(np.uint8) * 255)
    right = (after.astype(np.uint8) * 255)
    combo = np.concatenate([left, right], axis=1)
    Image.fromarray(combo, mode="L").save(out_path)


def save_debug_outputs(
    debug: Dict[str, Any], image_rgb: np.ndarray, instances: List[Instance], cfg: Dict[str, Any]
) -> None:
    import logging

    logger = logging.getLogger(__name__)
    debug_dir = cfg["debug"].get("debug_dir")
    if not debug_dir:
        logger.debug("Debug output disabled (no debug_dir specified)")
        return

    logger.info(f"Saving debug outputs to {debug_dir}")
    os.makedirs(debug_dir, exist_ok=True)

    if debug.get("candidate_points"):
        logger.debug(f"Saving candidate points visualization ({len(debug['candidate_points'])} points)")
        save_candidate_viz(image_rgb, debug["candidate_points"], os.path.join(debug_dir, "candidates.png"))
    if debug.get("tiles"):
        logger.debug(f"Saving tile grid visualization ({len(debug['tiles'])} tiles)")
        save_tile_grid_viz(image_rgb, debug["tiles"], os.path.join(debug_dir, "tiles.png"))
    if debug.get("hole_fill_example"):
        logger.debug("Saving hole fill before/after visualization")
        before, after = debug["hole_fill_example"]
        save_hole_fill_viz(before, after, os.path.join(debug_dir, "hole_fill_before_after.png"))
    if debug.get("candidate_points_by_method") and debug.get("pcs_instances"):
        logger.debug("Saving PCS overlay with candidate centers")
        pcs_overlay = build_rgba_overlay(
            image_rgb,
            debug["pcs_instances"],
            alpha=int(cfg["output"].get("overlay_alpha", 128)),
            colormap=cfg["output"].get("overlay_colormap", "tab20"),
        )
        overlay = draw_points_on_rgba(pcs_overlay, debug["candidate_points_by_method"])
        Image.fromarray(overlay, mode="RGBA").save(
            os.path.join(debug_dir, "pcs_candidates_overlay.png")
        )
    if instances:
        logger.debug(f"Saving consolidated overlay visualization ({len(instances)} masks)")
        overlay = build_rgba_overlay(
            image_rgb,
            instances,
            alpha=int(cfg["output"].get("overlay_alpha", 128)),
            colormap=cfg["output"].get("overlay_colormap", "tab20"),
        )
        Image.fromarray(overlay, mode="RGBA").save(os.path.join(debug_dir, "consolidated_overlay.png"))

    logger.info("Debug outputs saved successfully.")
