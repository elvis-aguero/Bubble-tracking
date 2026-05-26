from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def create_tiles(h: int, w: int, cfg: Dict[str, Any]) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
    tile_h = cfg["tiling"].get("tile_h") or cfg["tiling"].get("tile_size")
    tile_w = cfg["tiling"].get("tile_w") or cfg["tiling"].get("tile_size")
    tile_h = int(tile_h)
    tile_w = int(tile_w)

    overlap = cfg["tiling"].get("tile_overlap", 0)
    if isinstance(overlap, float) and 0 < overlap < 1:
        overlap_h = int(round(tile_h * overlap))
        overlap_w = int(round(tile_w * overlap))
    else:
        overlap_h = int(overlap)
        overlap_w = int(overlap)

    stride_h = max(tile_h - overlap_h, 1)
    stride_w = max(tile_w - overlap_w, 1)

    y_starts = list(range(0, max(h - tile_h, 0) + 1, stride_h))
    x_starts = list(range(0, max(w - tile_w, 0) + 1, stride_w))

    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]

    if y_starts[-1] + tile_h < h:
        y_starts.append(h - tile_h)
    if x_starts[-1] + tile_w < w:
        x_starts.append(w - tile_w)

    tiles: List[Tuple[int, int, int, int]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            tiles.append((x0, y0, x0 + tile_w, y0 + tile_h))

    pad_right = max(0, max(x1 for _, _, x1, _ in tiles) - w)
    pad_bottom = max(0, max(y1 for _, y0, _, y1 in tiles) - h)
    return tiles, (pad_bottom, pad_right)


def pad_image(arr: np.ndarray, pad_bottom: int, pad_right: int, mode: str) -> np.ndarray:
    if pad_bottom <= 0 and pad_right <= 0:
        return arr

    if arr.ndim == 2:
        pad_width = ((0, pad_bottom), (0, pad_right))
    else:
        pad_width = ((0, pad_bottom), (0, pad_right), (0, 0))

    return np.pad(arr, pad_width, mode=mode)
