#!/usr/bin/env python3
"""
prep_patches.py

End-to-end helper for the bubble instance-seg workflow.
Updates:
- Default source is current folder (.)
- CLAHE is enabled by default (use --no-clahe to disable)
- Auto-labeling is enabled by default (use --no-auto-label to disable)
- Default sensitivity set to 0.6

Workflow:
1) python prep_patches.py
   (Generates patches, applies CLAHE, and auto-generates JSON labels)

2) Open X-AnyLabeling -> Fix labels -> Export to YOLO.

3) python prep_patches.py --build-dataset
   (Compiles the final dataset)
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import random
import re
import shutil
import sys
import json
from typing import Iterable, Optional, Tuple, List

import cv2
import numpy as np
# Ensure scipy is installed: pip install scipy
from scipy import ndimage as ndi 

VALID_EXT_DEFAULT = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

# ----------------------------
# 1. Bubble Detection Logic
# ----------------------------
def detect_bubbles_deterministic(img: np.ndarray, sensitivity: float = 0.6, min_area: int = 50) -> List[List[List[float]]]:
    """
    Super-Conservative Bubble Detection.
    Prioritizes FALSE NEGATIVES over false positives.
    (Better to miss a bubble than label noise).
    """
    if img is None: return []
    
    # Work on grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 1. Bilateral Filter (The "Noise Killer")
    # Unlike Gaussian Blur, this keeps edges sharp but smothers texture noise.
    # d=9, sigmaColor=75, sigmaSpace=75 are standard strong denoisers.
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. Strict Adaptive Thresholding
    # We use a LARGE block size (keeps only big features) 
    # and a LARGE C constant (requires high contrast).
    # Sensitivity 0.6 -> C=12 (Very strict).
    c_val = 20 - (sensitivity * 15)  # Range: 5 (aggr) to 20 (strict)
    
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, c_val
    )

    # 3. Morphological "Opening" (Delete specks)
    # Erode (shrink) then Dilate (expand). 
    # This completely deletes thin noise or tiny dots.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Filter by Geometry (The "Shape Police")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_polygons = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area: continue
        
        # Calculate Perimeter
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue

        # METRIC: Circularity
        # A perfect circle has circularity ~1.0
        # A square is ~0.78
        # Jagged noise is usually < 0.5
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # STRICT RULE: Must be > 0.6 (Roughly circle-ish)
        if circularity < 0.6: 
            continue

        # METRIC: Convexity
        # Bubbles are "convex" (no large dents).
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        
        # STRICT RULE: Must be solid (> 0.85)
        # This removes "C" shapes or broken rings
        if solidity < 0.85:
            continue

        # If it survived, simplify and add it
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(c, epsilon, True)
        points = approx.reshape(-1, 2).tolist()
        
        if len(points) > 2:
            final_polygons.append(points)

    return final_polygons

def save_json_sidecar(image_path: Path, polygons: List[List[List[float]]], height: int, width: int) -> None:
    if not polygons:
        return

    data = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    for pts in polygons:
        shape = {
            "label": "bubble",
            "points": pts,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        data["shapes"].append(shape)

    json_path = image_path.with_suffix(".json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

# ----------------------------
# 2. Helpers
# ----------------------------
def norm_ext_list(exts: Iterable[str]) -> set[str]:
    out = set()
    for e in exts:
        e = e.strip().lower()
        if not e: continue
        out.add(e if e.startswith(".") else f".{e}")
    return out

def edge_complete_positions(L: int, tile: int, stride: int) -> list[int]:
    if L < tile: return []
    last = L - tile
    xs = list(range(0, last + 1, stride))
    if xs and xs[-1] != last: xs.append(last)
    return xs

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy": shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists(): dst.unlink()
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        if dst.exists(): dst.unlink()
        dst.hardlink_to(src)
    else: raise ValueError(f"Unknown mode: {mode}")

# ----------------------------
# 3. Patch Generation & Auto-Labeling
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
    sensitivity: float
) -> None:
    stride = max(1, int(round(tile * (1.0 - overlap))))

    patch_out.mkdir(parents=True, exist_ok=True)
    patch_map_csv.parent.mkdir(parents=True, exist_ok=True)
    if clahe_enable and save_clahe:
        clahe_out.mkdir(parents=True, exist_ok=True)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid) if clahe_enable else None

    rows = []
    n_images = 0
    n_patches = 0
    n_auto_labeled = 0

    print(f"Starting patch generation{' (Auto-Labeling ON)' if auto_label else ''}...")
    print(f"Source: {src_dir.resolve()}")

    for img_path in sorted(src_dir.glob("*")):
        if img_path.suffix.lower() not in valid_ext:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARNING: could not read {img_path}", file=sys.stderr)
            continue

        H, W = img.shape
        if H < tile or W < tile:
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
                patch = img_proc[y:y + tile, x:x + tile]
                if patch.shape != (tile, tile):
                    continue

                if skip_dark and float(patch.mean()) < float(dark_mean_thresh):
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
                        n_auto_labeled += 1

    with open(patch_map_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch_file", "source_file", "x0", "y0", "w", "h"])
        w.writerows(rows)

    print("Patch generation complete.")
    print(f"- Processed images: {n_images}")
    print(f"- Patches created:  {n_patches}")
    if auto_label:
        print(f"- JSON sidecars:    {n_auto_labeled}")

# ----------------------------
# 4. Dataset Builder & Utils
# ----------------------------
def parse_yolo_line(line: str) -> Optional[Tuple[int, list[float]]]:
    line = line.strip()
    if not line: return None
    toks = re.split(r"\s+", line)
    try:
        return int(toks[0]), [float(x) for x in toks[1:]]
    except:
        return None

def classify_yolo_txt(txt_path: Path) -> str:
    try:
        lines = txt_path.read_text(encoding="utf-8").splitlines()
    except:
        return "unknown"
    usable, seg_like, bbox_like = 0, 0, 0
    for ln in lines:
        parsed = parse_yolo_line(ln)
        if not parsed: continue
        usable += 1
        if len(parsed[1]) == 4: bbox_like += 1
        elif len(parsed[1]) > 4 and len(parsed[1]) % 2 == 0: seg_like += 1
    
    if usable == 0: return "empty"
    if seg_like > 0: return "seg"
    return "bbox" if bbox_like == usable else "unknown"

def check_yolo_numeric_ranges(txt_path: Path) -> list[str]:
    issues = []
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines, start=1):
        parsed = parse_yolo_line(ln)
        if parsed:
            for v in parsed[1]:
                if v < -0.01 or v > 1.01:
                    issues.append(f"{txt_path.name}: line {i} value {v:.3g}")
                    break
    return issues

def detect_sidecars(patch_dir: Path, valid_ext: set[str]) -> None:
    if not patch_dir.exists(): return
    patch_imgs = [p for p in sorted(patch_dir.glob("*")) if p.suffix.lower() in valid_ext]
    json_hits, txt_hits = 0, 0
    for img in patch_imgs[:5000]:
        stem = img.stem
        if (patch_dir / f"{stem}.json").exists(): json_hits += 1
        if (patch_dir / f"{stem}.txt").exists(): txt_hits += 1
    
    print(f"Sidecar Scan ({len(patch_imgs)} images):")
    print(f"- .json (X-AnyLabeling): {json_hits}")
    print(f"- .txt (YOLO Export):    {txt_hits}")

def build_ultralytics_dataset(
    patch_dir: Path, dataset_root: Path, split: float, seed: int, 
    link_mode: str, class_names: list[str], valid_ext: set[str], require_seg: bool
) -> None:
    imgs = [p for p in sorted(patch_dir.glob("*")) if p.suffix.lower() in valid_ext]
    labeled = []
    
    for im in imgs:
        txt = patch_dir / f"{im.stem}.txt"
        if not txt.exists(): continue
        kind = classify_yolo_txt(txt)
        if require_seg and kind != "seg": continue
        labeled.append((im, txt))

    if not labeled:
        raise RuntimeError("No valid labeled images found (Check if you ran Export YOLO).")

    random.Random(seed).shuffle(labeled)
    n_train = int(round(split * len(labeled)))
    train, val = labeled[:n_train], labeled[n_train:]

    for split_name, data in [("train", train), ("val", val)]:
        img_dest = dataset_root / "images" / split_name
        lbl_dest = dataset_root / "labels" / split_name
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        for im, txt in data:
            safe_link_or_copy(im, img_dest / im.name, link_mode)
            safe_link_or_copy(txt, lbl_dest / txt.name, link_mode)

    yaml_text = (
        f"path: {dataset_root.resolve()}\n"
        f"train: images/train\nval: images/val\n"
        f"nc: {len(class_names)}\nnames: {class_names}\n"
    )
    write_text(dataset_root / "bubbles.yaml", yaml_text)
    print(f"Dataset built at {dataset_root}")

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    
    # --- DEFAULTS UPDATED HERE ---
    p.add_argument("--src", type=Path, default=Path("."), help="Source folder (default: .)")
    p.add_argument("--patch-out", type=Path, default=Path("patches/images"))
    
    # Auto-Label (Default True)
    p.add_argument("--auto-label", dest="auto_label", action="store_true", default=True)
    p.add_argument("--no-auto-label", dest="auto_label", action="store_false")
    p.add_argument("--sensitivity", type=float, default=0.6)

    # CLAHE (Default True)
    p.add_argument("--clahe", dest="clahe", action="store_true", default=True)
    p.add_argument("--no-clahe", dest="clahe", action="store_false")

    # Modes
    p.add_argument("--detect-sidecars", action="store_true")
    p.add_argument("--build-dataset", action="store_true")
    p.add_argument("--dataset-root", type=Path, default=Path("bubbles_dataset"))
    p.add_argument("--write-classes-txt", type=Path)
    
    # Other Patch params
    p.add_argument("--patch-map", type=Path, default=Path("patches/patch_map.csv"))
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.30)
    p.add_argument("--skip-dark", action="store_true", default=True)
    p.add_argument("--no-skip-dark", dest="skip_dark", action="store_false")
    p.add_argument("--dark-mean-thresh", type=float, default=20.0)
    p.add_argument("--valid-ext", nargs="*", default=VALID_EXT_DEFAULT)
    
    # CLAHE options
    p.add_argument("--clahe-out", type=Path, default=Path("images_clahe"))
    p.add_argument("--save-clahe", action="store_true", default=True)
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-grid", type=int, nargs=2, default=(8, 8))
    
    # Dataset build options
    p.add_argument("--split", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--link-mode", choices=["copy", "symlink", "hardlink"], default="hardlink")
    p.add_argument("--class-name", type=str, default="bubble")
    p.add_argument("--require-seg", action="store_true", default=True)
    p.add_argument("--allow-bbox", dest="require_seg", action="store_false")

    args = p.parse_args()
    
    if args.write_classes_txt:
        write_text(args.write_classes_txt, f"{args.class_name}\n")
        return

    if args.detect_sidecars:
        detect_sidecars(args.patch_out, norm_ext_list(args.valid_ext))
        return

    if args.build_dataset:
        build_ultralytics_dataset(
            args.patch_out, args.dataset_root, args.split, args.seed,
            args.link_mode, [args.class_name], norm_ext_list(args.valid_ext), args.require_seg
        )
        return

    make_patches(
        args.src, args.patch_out, args.patch_map, args.tile, args.overlap,
        args.skip_dark, args.dark_mean_thresh, norm_ext_list(args.valid_ext),
        args.clahe, args.clahe_out, args.save_clahe, args.clahe_clip, 
        tuple(args.clahe_grid), args.auto_label, args.sensitivity
    )

if __name__ == "__main__":
    main()
