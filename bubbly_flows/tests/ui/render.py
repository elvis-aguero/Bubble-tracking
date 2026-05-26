from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


def _load_pil():
    from PIL import Image

    return Image


def _instance_colors() -> Tuple[Tuple[int, int, int], ...]:
    return (
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
    )


def render_gold_overlay(source_image_path: Path, gold_mask_path: Path, output_path: Path) -> Path:
    Image = _load_pil()
    source = Image.open(source_image_path).convert("RGB")
    mask = Image.open(gold_mask_path)
    width, height = source.size
    overlay = source.copy()
    src_px = overlay.load()
    mask_px = mask.load()
    colors = _instance_colors()

    for y in range(height):
        for x in range(width):
            instance_id = int(mask_px[x, y])
            if instance_id <= 0:
                continue
            base = src_px[x, y]
            color = colors[(instance_id - 1) % len(colors)]
            src_px[x, y] = (
                int(0.55 * base[0] + 0.45 * color[0]),
                int(0.55 * base[1] + 0.45 * color[1]),
                int(0.55 * base[2] + 0.45 * color[2]),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path


def summarize_labelme(labelme_json_path: Path) -> Dict[str, int]:
    import json

    payload = json.loads(labelme_json_path.read_text(encoding="utf-8"))
    return {"instance_count": len(payload.get("shapes", []))}
