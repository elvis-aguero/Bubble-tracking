#!/usr/bin/env python3
"""
utils.py

End-to-end helper for the bubble instance-seg workflow:

1) (Optional, default ON) CLAHE preprocessing on full frames
2) Edge-complete tiling into overlapping patches + patch_map.csv
3) (Optional, default ON) Auto-label patches by generating LabelMe/X-AnyLabeling .json polygon sidecars
4) Detect what label sidecars you have after X-AnyLabeling export (robust detection: .txt/.json/masks/alt-json)
5) Build an Ultralytics YOLOv8(-seg) dataset folder structure (train/val + bubbles.yaml)

Default behavior matches the "preprocessor.py / postprocess_..." scripts:
- Default source is current folder (.)
- CLAHE enabled by default (use --no-clahe to disable)
- Auto-labeling enabled by default (use --no-auto-label to disable)
- Default sensitivity set to 0.6
- Skips dark patches by default
- Writes CLAHE full frames by default (use --no-save-clahe to disable)

Typical workflow
----------------
A) Generate 640x640 patches with ~30% overlap (CLAHE + auto-label .json sidecars by default):
    python utils.py

B) (Optional) Create the YOLO classes file needed by the X-AnyLabeling "Export YOLO-seg" dialog:
    python utils.py --write-classes-txt patches/images/classes.txt --class-name bubble

C) After labeling/fixing/exporting in X-AnyLabeling, detect which sidecars exist:
    python utils.py --detect-sidecars --patch-out patches/images

D) Build an Ultralytics-ready dataset (hardlinks save space if same filesystem):
    python utils.py --build-dataset --patch-out patches/images --dataset-root bubbles_dataset --split 0.90 --link-mode hardlink --require-seg

Useful variants
---------------
- Disable auto-labeling:
    python utils.py --no-auto-label

- Disable CLAHE:
    python utils.py --no-clahe

- Disable skipping dark/vignette patches:
    python utils.py --no-skip-dark

- Change tile size/overlap:
    python utils.py --tile 640 --overlap 0.30

- Allow bbox-only YOLO labels (NOT recommended for instance segmentation):
    python utils.py --build-dataset --allow-bbox

Notes
-----
- For YOLOv8 segmentation, label .txt lines must be polygons:
    class x1 y1 x2 y2 x3 y3 ...
  (bbox-only labels: class xc yc w h) are excluded when --require-seg is on.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import random
import re
import shutil
import sys
from typing import Iterable, Optional, Tuple, List

import cv2
import numpy as np

# Auto-relaunch under repository-root virtualenv if available and not already active.
# This ensures `utils.py` uses the same venv that auxiliary scripts in this repo expect.
try:
    import os
    if not os.environ.get("VIRTUAL_ENV") and not os.environ.get("_UTILS_VENV_LAUNCHED"):
        _script_dir = Path(__file__).resolve().parent
        _repo_root = _script_dir.parent
        # Try a few common venv names; prefer the project's x-labeling env
        for _venv_name in ("x-labeling-env", ".venv", "venv"):
            _py = _repo_root / _venv_name / "bin" / "python"
            if _py.exists():
                os.environ["_UTILS_VENV_LAUNCHED"] = "1"
                os.execv(str(_py), [str(_py)] + sys.argv)
except Exception:
    # If anything goes wrong, continue without relaunching
    pass


VALID_EXT_DEFAULT = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


# ----------------------------
# 1. Bubble Detection Logic (auto-label)
# ----------------------------
def detect_bubbles_deterministic(
    img: np.ndarray,
    sensitivity: float = 0.6,
    min_area: int = 50,
) -> List[List[List[float]]]:
    """
    Super-conservative bubble detection.
    Prioritizes false negatives over false positives.
    Returns polygons as [[x,y], ...] lists (pixel coordinates).
    """
    if img is None:
        return []

    # Work on grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 1) Bilateral filter: strong denoise while keeping edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2) Strict adaptive thresholding
    # sensitivity: higher -> more aggressive; lower -> more strict
    c_val = 20 - (sensitivity * 15)  # ~5 (aggr) to ~20 (strict)
    binary = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        c_val,
    )

    # 3) Morphological opening to remove specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4) Geometry filtering
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_polygons: List[List[List[float]]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.6:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidity = float(area) / float(hull_area)
        if solidity < 0.85:
            continue

        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(c, epsilon, True)
        points = approx.reshape(-1, 2).tolist()
        if len(points) > 2:
            final_polygons.append(points)

    return final_polygons


def save_json_sidecar(image_path: Path, polygons: List[List[List[float]]], height: int, width: int) -> None:
    """
    Writes LabelMe/X-AnyLabeling compatible JSON next to image.
    """
    if not polygons:
        return

    data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }

    for pts in polygons:
        data["shapes"].append(
            {
                "label": "bubble",
                "points": pts,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )

    json_path = image_path.with_suffix(".json")
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ----------------------------
# 2. Helpers
# ----------------------------
def norm_ext_list(exts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        out.add(e if e.startswith(".") else f".{e}")
    return out


def edge_complete_positions(L: int, tile: int, stride: int) -> list[int]:
    if L < tile:
        return []
    last = L - tile
    xs = list(range(0, last + 1, stride))
    if xs and xs[-1] != last:
        xs.append(last)
    return xs


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)  # requires same filesystem
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ----------------------------
# 3. Patch Generation (+ optional auto-label)
# ----------------------------
def make_patches(
    src_dir: Path,
    patch_out: Path,
    patch_map_csv: Path,
    tile: int,
    overlap: float,
    skip_dark: bool,
    dark_mean_thresh: float,
    valid_ext: set[str],
    clahe_enable: bool,
    clahe_out: Path,
    save_clahe: bool,
    clahe_clip: float,
    clahe_grid: Tuple[int, int],
    auto_label: bool,
    sensitivity: float,
) -> None:
    if not (0.0 <= overlap < 1.0):
        raise ValueError("--overlap must be in [0, 1).")

    stride = max(1, int(round(tile * (1.0 - overlap))))

    patch_out.mkdir(parents=True, exist_ok=True)
    patch_map_csv.parent.mkdir(parents=True, exist_ok=True)
    if clahe_enable and save_clahe:
        clahe_out.mkdir(parents=True, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid) if clahe_enable else None

    rows: list[list[object]] = []
    n_images = 0
    n_skipped_small = 0
    n_patches = 0
    n_skipped_dark = 0
    n_auto_labeled_imgs = 0  # count patches that got a json sidecar

    print(
        "Patch generation starting"
        + (" (CLAHE ON)" if clahe_enable else " (CLAHE OFF)")
        + (" (Auto-label ON)" if auto_label else " (Auto-label OFF)")
        + f"\n- Source: {src_dir.resolve()}"
        + f"\n- Patch out: {patch_out.resolve()}"
    )

    for img_path in sorted(src_dir.glob("*")):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in valid_ext:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARNING: could not read {img_path}", file=sys.stderr)
            continue

        H, W = img.shape
        if H < tile or W < tile:
            n_skipped_small += 1
            continue

        n_images += 1

        if clahe is not None:
            img_proc = clahe.apply(img)
            if save_clahe:
                cv2.imwrite(str(clahe_out / img_path.name), img_proc)
        else:
            img_proc = img

        xs = edge_complete_positions(W, tile, stride)
        ys = edge_complete_positions(H, tile, stride)

        for y in ys:
            for x in xs:
                patch = img_proc[y : y + tile, x : x + tile]
                if patch.shape != (tile, tile):
                    continue

                if skip_dark and float(patch.mean()) < float(dark_mean_thresh):
                    n_skipped_dark += 1
                    continue

                out_name = f"{img_path.stem}__x{x:04d}_y{y:04d}.png"
                out_path = patch_out / out_name
                cv2.imwrite(str(out_path), patch)

                rows.append([out_name, img_path.name, x, y, tile, tile])
                n_patches += 1

                if auto_label:
                    polys = detect_bubbles_deterministic(patch, sensitivity=sensitivity)
                    if polys:
                        save_json_sidecar(out_path, polys, tile, tile)
                        n_auto_labeled_imgs += 1

    with open(patch_map_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch_file", "source_file", "x0", "y0", "w", "h"])
        w.writerows(rows)

    print("Patch generation complete.")
    print(f"- Images processed: {n_images}")
    print(f"- Skipped small:    {n_skipped_small}")
    print(f"- Patches written:  {n_patches}")
    if skip_dark:
        print(f"- Skipped dark:     {n_skipped_dark}")
    if auto_label:
        print(f"- JSON sidecars:    {n_auto_labeled_imgs}")


# ----------------------------
# 4. Sidecar detection (robust)
# ----------------------------
def detect_sidecars(patch_dir: Path, valid_ext: set[str]) -> None:
    """
    Robustly detect what labels exist next to patch images.
    Counts:
      - .txt (YOLO export)
      - .json (LabelMe / X-AnyLabeling)
      - mask image variants
      - alternate json variants
    """
    if not patch_dir.exists():
        print(f"ERROR: patch dir not found: {patch_dir}", file=sys.stderr)
        return

    patch_imgs = [p for p in sorted(patch_dir.glob("*")) if p.is_file() and p.suffix.lower() in valid_ext]
    if not patch_imgs:
        print(f"No patch images found in: {patch_dir}")
        return

    txt_hits = 0
    json_hits = 0
    mask_hits = 0
    other_json_hits = 0

    mask_suffixes = ["_mask", ".mask", "-mask", "_seg", "_segmask", "_annotation"]
    mask_exts = [".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"]

    # sample up to 5000 like prep_patches
    for img in patch_imgs[:5000]:
        stem = img.stem

        if (patch_dir / f"{stem}.txt").exists():
            txt_hits += 1
        if (patch_dir / f"{stem}.json").exists():
            json_hits += 1

        if (patch_dir / f"{stem}.label.json").exists() or (patch_dir / f"{stem}.annotations.json").exists():
            other_json_hits += 1

        found_mask = False
        for suf in mask_suffixes:
            for ext in mask_exts:
                if (patch_dir / f"{stem}{suf}{ext}").exists():
                    mask_hits += 1
                    found_mask = True
                    break
            if found_mask:
                break

    print("Sidecar detection summary (sampled up to 5000 patches):")
    print(f"- Patch images found:        {len(patch_imgs)}")
    print(f"- .txt next to patches:      {txt_hits}")
    print(f"- .json next to patches:     {json_hits}")
    print(f"- mask images detected:      {mask_hits}")
    print(f"- other JSON variants:       {other_json_hits}")
    print("")
    print("If you expect YOLO-seg export, you want many .txt files.")
    print("Still verify the .txt content is segmentation polygons (not bbox).")


# ----------------------------
# 5. YOLO label parsing + checks
# ----------------------------
def parse_yolo_line(line: str) -> Optional[Tuple[int, list[float]]]:
    line = line.strip()
    if not line:
        return None
    toks = re.split(r"\s+", line)
    try:
        cls = int(toks[0])
        nums = [float(x) for x in toks[1:]]
        return cls, nums
    except Exception:
        return None


def classify_yolo_txt(txt_path: Path) -> str:
    """
    Returns:
      - "seg" if at least one non-empty line looks like YOLO segmentation polygon
      - "bbox" if lines look like YOLO detection bbox (class + 4 numbers)
      - "empty" if no usable lines
      - "unknown" otherwise
    """
    try:
        lines = txt_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "unknown"

    usable = 0
    seg_like = 0
    bbox_like = 0

    for ln in lines:
        parsed = parse_yolo_line(ln)
        if parsed is None:
            continue
        _, nums = parsed
        usable += 1
        if len(nums) == 4:
            bbox_like += 1
        elif len(nums) > 4 and len(nums) % 2 == 0:
            seg_like += 1

    if usable == 0:
        return "empty"
    if seg_like > 0:
        return "seg"
    if bbox_like == usable:
        return "bbox"
    return "unknown"


def check_yolo_numeric_ranges(txt_path: Path) -> list[str]:
    """
    Basic checks: coords typically normalized [0,1]. Allow slight tolerance.
    """
    issues: list[str] = []
    try:
        lines = txt_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return [f"{txt_path.name}: could not read"]

    for i, ln in enumerate(lines, start=1):
        parsed = parse_yolo_line(ln)
        if parsed is None:
            continue
        cls, nums = parsed
        if cls < 0:
            issues.append(f"{txt_path.name}: line {i} has negative class id")
        for v in nums:
            if v < -0.01 or v > 1.01:
                issues.append(f"{txt_path.name}: line {i} has value {v:.3g} outside ~[0,1]")
                break
    return issues


# ----------------------------
# 6. Dataset builder (Ultralytics YOLO)
# ----------------------------
def build_ultralytics_dataset(
    patch_dir: Path,
    dataset_root: Path,
    split: float,
    seed: int,
    link_mode: str,
    class_names: list[str],
    valid_ext: set[str],
    require_seg: bool,
) -> None:
    """
    Expects labels stored as <image_stem>.txt next to images.
    Creates:
      dataset_root/images/train, dataset_root/images/val
      dataset_root/labels/train, dataset_root/labels/val
      dataset_root/bubbles.yaml
    """
    imgs = [p for p in sorted(patch_dir.glob("*")) if p.is_file() and p.suffix.lower() in valid_ext]
    if not imgs:
        raise RuntimeError(f"No images found in {patch_dir}")

    labeled: list[Tuple[Path, Path, str]] = []
    unlabeled = 0
    bad_kind = 0
    range_issues = 0

    for im in imgs:
        txt = patch_dir / f"{im.stem}.txt"
        if not txt.exists():
            unlabeled += 1
            continue

        kind = classify_yolo_txt(txt)
        if require_seg and kind != "seg":
            bad_kind += 1
            continue

        if check_yolo_numeric_ranges(txt):
            range_issues += 1

        labeled.append((im, txt, kind))

    if not labeled:
        raise RuntimeError(
            f"No labeled images found (or none passed checks). "
            f"Found {len(imgs)} images, {unlabeled} unlabeled, {bad_kind} wrong-kind."
        )

    print("Label scan:")
    print(f"- Images total:                 {len(imgs)}")
    print(f"- Labeled pairs found:          {len(labeled)}")
    print(f"- Unlabeled images skipped:     {unlabeled}")
    if require_seg:
        print(f"- Labels rejected (not seg):    {bad_kind}")
    print(f"- Labels w/ out-of-range nums:  {range_issues}  (warn-only)")

    rnd = random.Random(seed)
    rnd.shuffle(labeled)

    n_train = int(round(split * len(labeled)))
    train = labeled[:n_train]
    # Ensure val is not empty
    if n_train < len(labeled):
        val = labeled[n_train:]
    else:
        val = labeled[-max(1, len(labeled) // 10) :]

    img_train = dataset_root / "images" / "train"
    img_val = dataset_root / "images" / "val"
    lab_train = dataset_root / "labels" / "train"
    lab_val = dataset_root / "labels" / "val"
    for d in [img_train, img_val, lab_train, lab_val]:
        d.mkdir(parents=True, exist_ok=True)

    for im, txt, _ in train:
        safe_link_or_copy(im, img_train / im.name, link_mode)
        safe_link_or_copy(txt, lab_train / txt.name, link_mode)

    for im, txt, _ in val:
        safe_link_or_copy(im, img_val / im.name, link_mode)
        safe_link_or_copy(txt, lab_val / txt.name, link_mode)

    yaml_path = dataset_root / "bubbles.yaml"
    names_yaml = "[" + ", ".join([f"'{n}'" for n in class_names]) + "]"
    yaml_text = (
        f"path: {dataset_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {names_yaml}\n"
    )
    write_text(yaml_path, yaml_text)

    print("Dataset build complete:")
    print(f"- Root:      {dataset_root}")
    print(f"- Train:     {len(train)}")
    print(f"- Val:       {len(val)}")
    print(f"- YAML:      {yaml_path}")
    print(f"- Link mode: {link_mode}")
    if require_seg:
        print("- Require seg: True (bbox-only .txt were excluded)")


# ----------------------------
# 7. CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLAHE + patching + optional auto-label + sidecar detection + Ultralytics dataset builder"
    )

    # Valid extensions
    p.add_argument("--valid-ext", type=str, nargs="*", default=VALID_EXT_DEFAULT)

    # Modes
    p.add_argument("--detect-sidecars", action="store_true")
    p.add_argument("--build-dataset", action="store_true")

    # Patch params (defaults match preprocessor/postprocess)
    p.add_argument("--src", type=Path, default=Path("."), help="Source folder (default: .)")
    p.add_argument("--patch-out", type=Path, default=Path("patches/images"))
    p.add_argument("--patch-map", type=Path, default=Path("patches/patch_map.csv"))
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.30)
    p.add_argument("--skip-dark", action="store_true", default=True)
    p.add_argument("--no-skip-dark", dest="skip_dark", action="store_false")
    p.add_argument("--dark-mean-thresh", type=float, default=20.0)

    # CLAHE (defaults match preprocessor/postprocess: enabled by default)
    p.add_argument("--clahe", dest="clahe", action="store_true", default=True)
    p.add_argument("--no-clahe", dest="clahe", action="store_false")
    p.add_argument("--clahe-out", type=Path, default=Path("images_clahe"))
    p.add_argument("--save-clahe", dest="save_clahe", action="store_true", default=True)
    p.add_argument("--no-save-clahe", dest="save_clahe", action="store_false")
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-grid", type=int, nargs=2, default=(8, 8), metavar=("GX", "GY"))

    # Auto-label (defaults match preprocessor/postprocess: enabled by default)
    p.add_argument("--auto-label", dest="auto_label", action="store_true", default=True)
    p.add_argument("--no-auto-label", dest="auto_label", action="store_false")
    p.add_argument("--sensitivity", type=float, default=0.6)

    # Dataset build params (keep YOLO backbone folder generation)
    p.add_argument("--dataset-root", type=Path, default=Path("bubbles_dataset"))
    p.add_argument("--split", type=float, default=0.90, help="Train split fraction (rest goes to val).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--link-mode",
        choices=["copy", "symlink", "hardlink"],
        default="hardlink",
        help="How to populate dataset folders (hardlink saves space if same filesystem).",
    )
    p.add_argument("--class-name", type=str, default="bubble", help="Single-class name.")
    p.add_argument(
        "--require-seg",
        action="store_true",
        default=True,
        help="Only include labels that look like segmentation polygons (exclude bbox-only).",
    )
    p.add_argument("--allow-bbox", dest="require_seg", action="store_false", help="Allow bbox-only labels.")

    # Classes file helper
    p.add_argument(
        "--write-classes-txt",
        type=Path,
        default=None,
        help="If set, write a YOLO classes file (one class per line) and exit.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    valid_ext = norm_ext_list(args.valid_ext)

    # Helper: write classes.txt
    if args.write_classes_txt is not None:
        write_text(args.write_classes_txt, f"{args.class_name}\n")
        print(f"Wrote classes file: {args.write_classes_txt} (contents: {args.class_name})")
        return

    # Mode: detect sidecars
    if args.detect_sidecars:
        detect_sidecars(args.patch_out, valid_ext)
        return

    # Mode: build dataset
    if args.build_dataset:
        build_ultralytics_dataset(
            patch_dir=args.patch_out,
            dataset_root=args.dataset_root,
            split=args.split,
            seed=args.seed,
            link_mode=args.link_mode,
            class_names=[args.class_name],
            valid_ext=valid_ext,
            require_seg=args.require_seg,
        )
        return

    # Default: patching (+ optional auto-label)
    make_patches(
        src_dir=args.src,
        patch_out=args.patch_out,
        patch_map_csv=args.patch_map,
        tile=args.tile,
        overlap=args.overlap,
        skip_dark=args.skip_dark,
        dark_mean_thresh=args.dark_mean_thresh,
        valid_ext=valid_ext,
        clahe_enable=args.clahe,
        clahe_out=args.clahe_out,
        save_clahe=args.save_clahe,
        clahe_clip=args.clahe_clip,
        clahe_grid=tuple(args.clahe_grid),
        auto_label=args.auto_label,
        sensitivity=args.sensitivity,
    )


if __name__ == "__main__":
    main()

