from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .backend import (
    Sam3ConceptBackend,
    Sam3PointBackend,
    TransformersMaskGenerator,
    segment_everything,
    segment_with_points,
)
from .candidates import detect_candidates
from .postprocess import Instance, consolidate_instances, fill_holes, maybe_convex_hull, resize_mask_to_shape, mask_bbox
from .preprocess import preprocess_gray
from .tiling import create_tiles, pad_image


def run_pipeline(
    image: Image.Image,
    cfg: Dict[str, Any],
    sam_backend: Sam3PointBackend,
    fallback_generator: Optional[TransformersMaskGenerator],
    pcs_backend: Optional[Sam3ConceptBackend] = None,
) -> Tuple[List[Instance], Dict[str, Any], Optional[TransformersMaskGenerator]]:
    logger = logging.getLogger(__name__)
    debug: Dict[str, Any] = {
        "candidate_points": [],
        "tiles": [],
        "hole_fill_example": None,
    }

    image_rgb = np.array(image.convert("RGB"))
    gray = preprocess_gray(image, cfg)
    logger.info(f"Image preprocessed. Shape: {gray.shape}")

    h, w = gray.shape

    tiles: List[Tuple[int, int, int, int]] = [(0, 0, w, h)]
    pad_bottom = 0
    pad_right = 0
    if cfg["tiling"].get("enable_tiling", True):
        logger.info("Tiling enabled. Creating tile grid...")
        tiles, (pad_bottom, pad_right) = create_tiles(h, w, cfg)
        logger.info(f"Created {len(tiles)} tiles with padding bottom={pad_bottom}, right={pad_right}")
    else:
        logger.info("Tiling disabled. Processing full image.")

    debug["tiles"] = tiles

    pad_mode = cfg["tiling"].get("pad_mode", "reflect")
    padded_rgb = pad_image(image_rgb, pad_bottom, pad_right, pad_mode)
    padded_gray = pad_image(gray, pad_bottom, pad_right, pad_mode)

    instances: List[Instance] = []

    for tile_idx, (x0, y0, x1, y1) in enumerate(tiles):
        logger.debug(f"Processing tile {tile_idx + 1}/{len(tiles)}: ({x0}, {y0}) to ({x1}, {y1})")
        tile_rgb = padded_rgb[y0:y1, x0:x1]
        tile_gray = padded_gray[y0:y1, x0:x1]
        tile_h, tile_w = tile_gray.shape

        tile_points: List[Tuple[float, float]] = []
        if cfg["candidates"].get("enable_candidates", True):
            cands = detect_candidates(tile_gray, cfg)
            tile_points = [(c.x, c.y) for c in cands]
            logger.debug(f"  Tile {tile_idx + 1}: Found {len(cands)} candidate points")
            for c in cands:
                debug["candidate_points"].append((c.x + x0, c.y + y0))
        else:
            cands = []

        use_fallback_grid = False
        if cfg["candidates"].get("enable_candidates", True):
            if not tile_points and cfg["candidates"].get("fallback_on_empty_candidates", True):
                use_fallback_grid = True
        else:
            use_fallback_grid = True

        tile_pil = Image.fromarray(tile_rgb)

        masks: List[np.ndarray] = []
        scores: List[Optional[float]] = []

        if use_fallback_grid:
            masks, scores, fallback_generator = segment_everything(
                tile_pil, sam_backend, fallback_generator, cfg, cfg["seed"]
            )
        else:
            masks, scores = segment_with_points(sam_backend, tile_pil, tile_points)

        if not masks:
            continue

        for mask, score in zip(masks, scores):
            mask = resize_mask_to_shape(mask, (tile_h, tile_w))
            before = mask.copy()
            mask = fill_holes(mask, cfg)
            mask = maybe_convex_hull(mask, cfg)
            if debug["hole_fill_example"] is None:
                debug["hole_fill_example"] = (before, mask)

            y1_clip = min(y1, h)
            x1_clip = min(x1, w)
            tile_clip_h = y1_clip - y0
            tile_clip_w = x1_clip - x0
            if tile_clip_h <= 0 or tile_clip_w <= 0:
                continue
            mask_clip = mask[:tile_clip_h, :tile_clip_w]

            if cfg["tiling"].get("enable_tiling", True):
                coverage = mask_clip.sum() / max(1.0, mask.sum())
                if coverage < float(cfg["tiling"].get("min_coverage_for_keep", 0.0)):
                    continue

            local_bbox = mask_bbox(mask_clip)
            if local_bbox == (0, 0, 0, 0):
                continue
            bx0, by0, bx1, by1 = local_bbox
            cropped = mask_clip[by0:by1, bx0:bx1]
            area = int(cropped.sum())
            if area == 0:
                continue

            bbox = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
            instances.append(Instance(mask=cropped, score=score, area=area, bbox=bbox))

    pcs_enable = bool(cfg["sam"].get("pcs_enable", True))
    if pcs_enable:
        if pcs_backend is None:
            pcs_backend = Sam3ConceptBackend(cfg["device"], cfg["sam"])
        prompt = str(cfg["sam"].get("pcs_text_prompt", "bubbles"))
        pcs_threshold = float(cfg["sam"].get("pcs_threshold", 0.5))
        pcs_mask_threshold = float(cfg["sam"].get("pcs_mask_threshold", 0.5))

        pcs_masks, pcs_scores, pcs_boxes = pcs_backend.segment_by_text(
            image, prompt, pcs_threshold, pcs_mask_threshold
        )
        pcs_produced = len(pcs_masks)
        pcs_survived = 0
        pcs_added = 0

        for mask, score, box in zip(pcs_masks, pcs_scores, pcs_boxes):
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
            pcs_survived += 1
            pcs_added += 1
            instances.append(Instance(mask=crop, score=score, area=area, bbox=(x0, y0, x1, y1)))

        logger.info(
            "PCS produced %d masks; %d survived filtering; %d added before consolidation",
            pcs_produced,
            pcs_survived,
            pcs_added,
        )

    logger.info(f"Total instances before consolidation: {len(instances)}")
    consolidated = consolidate_instances(instances, cfg, (h, w))
    logger.info(f"Total instances after consolidation: {len(consolidated)}")
    return consolidated, debug, fallback_generator
