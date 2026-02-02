from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def ensure_hf_home() -> None:
    """Prefer an existing HF cache location to avoid re-downloading models."""
    if "HF_HOME" in os.environ:
        return
    hf_cache = os.path.expanduser("~/hf")
    if os.path.exists(hf_cache):
        os.environ["HF_HOME"] = hf_cache


DEFAULT_CONFIG: Dict[str, Any] = {
    "device": "cuda",
    "seed": 0,
    "sam": {
        "backend": "tracker",  # tracker
        "model_name": "facebook/sam3",
        "confidence_threshold": 0.4,
        "use_fp16": True,
        "local_files_only": True,
        "points_per_batch": 128,
        "multimask_output": False,
        "pcs_enable": True,
        "pcs_text_prompt": "bubbles",
        "pcs_threshold": 0.5,
        "pcs_mask_threshold": 0.5,
    },
    "preprocess": {
        "enable_contrast_norm": True,
        "contrast_method": "clahe",  # clahe | rescale | none
        "clahe_clip_limit": 0.03,
        "clahe_kernel_size": 64,
        "rescale_percentiles": [2, 98],
        "gamma": 1.0,
        "invert_gray": False,
    },
    "candidates": {
        "enable_candidates": True,
        "method": "log+dog",  # log | dog | hough | log+dog | all
        "min_diameter_px": 10,
        "max_diameter_px": 200,
        "log_min_sigma": None,
        "log_max_sigma": None,
        "log_num_sigma": 10,
        "log_threshold": 0.02,
        "dog_min_sigma": None,
        "dog_max_sigma": None,
        "dog_sigma_ratio": 1.6,
        "dog_threshold": 0.02,
        "hough_dp": 1.2,
        "hough_minDist": 8,
        "hough_param1": 100,
        "hough_param2": 30,
        "hough_minRadius": None,
        "hough_maxRadius": None,
        "max_candidates_per_image": 2000,  # per-tile cap
        "dedup_centers": True,
        "dedup_radius_px": 4.0,
        "fallback_on_empty_candidates": True,
    },
    "tiling": {
        "enable_tiling": True,
        "tile_size": 256,
        "tile_h": None,
        "tile_w": None,
        "tile_overlap": 64,
        "pad_mode": "reflect",  # reflect | constant | edge
        "min_coverage_for_keep": 0.5,
    },
    "hole_fill": {
        "enable_hole_fill": True,
        "hole_fill_method": "binary_fill_holes",  # binary_fill_holes | closing
        "closing_radius": 2,
        "min_hole_area_px": 0,
        "fill_only_if_ringlike": False,
        "ringlike_threshold": 0.15,
    },
    "postprocess": {
        "enable_consolidation": True,
        "iou_dedup_thresh": 0.5,
        "containment_thresh": 0.9,
        "min_area_px": 60,
        "max_area_fraction": 0.2,
        "border_exclusion_px": 1,
        "circularity_min": 0.0,
        "solidity_min": 0.0,
        "enable_convex_hull": False,
    },
    "fallback": {
        "mode": "grid",  # grid | transformers
        "grid_spacing_px": 24,
        "grid_jitter_px": 0,
        "max_points": 4096,
    },
    "blackhat": {
        "enable": True,
        "radius": 5,
        "percentile": 99.0,
        "area_min": 30,
        "area_max": 120,
        "watershed": True,
        "watershed_min_area": None,
        "watershed_fg_thresh": 0.5,
        "blob_method": "dog",  # dog | log
        "blob_min_sigma": 1.3,
        "blob_max_sigma": 2.8,
        "blob_num_sigma": 10,
        "blob_threshold": 0.05,
        "blob_overlap": 0.5,
        "patch_radius": 6,
        "patch_percentile": 90.0,
        "patch_use_otsu": False,
        "fallback_global_cc": False,
    },
    "output": {
        "output_mode": "overlay",  # cutout | overlay
        "overlay_alpha": 128,
        "overlay_colormap": "tab20",
        "write_json": True,
        "json_path": None,
        "write_csv": False,
        "csv_path": None,
        "include_rle": False,
    },
    "debug": {
        "debug_dir": None,
    },
}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = merge_dicts(base[key], val)
        else:
            base[key] = val
    return base


def strip_jsonc_comments(raw: str) -> str:
    """Remove // and /* */ comments from a JSONC string."""
    in_str = False
    escape = False
    out: List[str] = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt == "/":
                i += 2
                while i < len(raw) and raw[i] not in "\r\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < len(raw) and not (raw[i] == "*" and raw[i + 1] == "/"):
                    i += 1
                i += 2
                continue

        out.append(ch)
        i += 1
    return "".join(out)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        if path.lower().endswith(".jsonc"):
            raw = strip_jsonc_comments(raw)
        user_cfg = json.loads(raw)
        cfg = merge_dicts(cfg, user_cfg)
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    if args.device:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.debug_dir:
        cfg["debug"]["debug_dir"] = args.debug_dir
    if args.output_json:
        cfg["output"]["json_path"] = args.output_json
        cfg["output"]["write_json"] = True
    if args.output_csv:
        cfg["output"]["csv_path"] = args.output_csv
        cfg["output"]["write_csv"] = True
    if args.include_rle:
        cfg["output"]["include_rle"] = True
    if args.no_output_json:
        cfg["output"]["write_json"] = False

    if args.sam_model:
        cfg["sam"]["model_name"] = args.sam_model
    if args.points_per_batch is not None:
        cfg["sam"]["points_per_batch"] = args.points_per_batch
    if args.multimask_output:
        cfg["sam"]["multimask_output"] = True
    if args.allow_download:
        cfg["sam"]["local_files_only"] = False

    if args.enable_candidates:
        cfg["candidates"]["enable_candidates"] = True
    if args.disable_candidates:
        cfg["candidates"]["enable_candidates"] = False
    if args.enable_tiling:
        cfg["tiling"]["enable_tiling"] = True
    if args.disable_tiling:
        cfg["tiling"]["enable_tiling"] = False
    if args.enable_hole_fill:
        cfg["hole_fill"]["enable_hole_fill"] = True
    if args.disable_hole_fill:
        cfg["hole_fill"]["enable_hole_fill"] = False
    if args.enable_consolidation:
        cfg["postprocess"]["enable_consolidation"] = True
    if args.disable_consolidation:
        cfg["postprocess"]["enable_consolidation"] = False
    if args.output_mode:
        cfg["output"]["output_mode"] = args.output_mode

    return cfg
