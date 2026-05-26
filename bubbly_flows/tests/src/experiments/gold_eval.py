from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional


VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


MaskRows = List[List[int]]
MaskWriter = Callable[[MaskRows, Path], None]


def _find_source_image(image_stem: str, image_name: Optional[str], image_search_dirs: Iterable[Path]) -> Optional[Path]:
    if image_name:
        exact_name = Path(image_name).name
        for search_dir in image_search_dirs:
            candidate = search_dir / exact_name
            if candidate.exists():
                return candidate

    for search_dir in image_search_dirs:
        for ext in VALID_EXTS:
            candidate = search_dir / f"{image_stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def _point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def rasterize_labelme_shapes(width: int, height: int, shapes: List[dict]) -> MaskRows:
    mask: MaskRows = [[0 for _ in range(width)] for _ in range(height)]
    instance_id = 1
    for shape in shapes:
        points = shape.get("points") or []
        if len(points) < 3:
            continue

        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        min_x = max(int(min(xs)), 0)
        max_x = min(int(max(xs)) + 1, width - 1)
        min_y = max(int(min(ys)), 0)
        max_y = min(int(max(ys)) + 1, height - 1)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if _point_in_polygon(x + 0.5, y + 0.5, points):
                    mask[y][x] = instance_id
        instance_id += 1
    return mask


def _default_mask_writer(mask_rows: MaskRows, destination: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write default evaluation masks") from exc

    height = len(mask_rows)
    width = len(mask_rows[0]) if height else 0
    image = Image.new("I;16", (width, height), color=0)
    for y, row in enumerate(mask_rows):
        for x, value in enumerate(row):
            image.putpixel((x, y), int(value))
    image.save(destination)


def prepare_gold_evaluation_set(
    gold_json_dir: Path,
    output_dir: Path,
    image_search_dirs: Iterable[Path],
    mask_writer: Optional[MaskWriter] = None,
) -> dict:
    gold_json_dir = gold_json_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    writer = mask_writer or _default_mask_writer
    dataset_name = gold_json_dir.parent.name
    items = []

    for json_path in sorted(gold_json_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        image_stem = json_path.stem
        image_name = payload.get("imagePath")
        image_path = _find_source_image(image_stem, image_name, image_search_dirs)
        if image_path is None:
            raise FileNotFoundError(f"Could not find source image for {json_path.name}")

        width = int(payload["imageWidth"])
        height = int(payload["imageHeight"])
        mask_rows = rasterize_labelme_shapes(width, height, payload.get("shapes", []))

        copied_image_path = images_dir / image_path.name
        shutil.copy2(image_path, copied_image_path)

        label_path = labels_dir / f"{image_stem}.tif"
        writer(mask_rows, label_path)

        items.append(
            {
                "stem": image_stem,
                "source_json": str(json_path.relative_to(gold_json_dir.parent.parent)),
                "source_image": str(image_path),
                "image": f"images/{copied_image_path.name}",
                "label": f"labels/{label_path.name}",
                "instance_count": max((max(row) for row in mask_rows), default=0),
            }
        )

    manifest = {
        "dataset_name": dataset_name,
        "dataset_root": str(output_dir),
        "source_dir": str(gold_json_dir),
        "count": len(items),
        "items": items,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
