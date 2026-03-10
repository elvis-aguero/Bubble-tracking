#!/usr/bin/env python3
"""
train_yolov9.py — Fine-tune YOLOv9c-seg on bubble instance masks.

Converts our uint16 TIF instance masks to YOLO polygon format on-the-fly,
then fine-tunes from the pre-staged yolov9c-seg.pt weights.
(No bubble-specific YOLOv9 weights were published by Yang et al. 2025 or
Nizovtseva et al. 2024; we start from COCO-pretrained weights.)

Requires (not in bubbly-train-env by default — install once):
    pip install ultralytics tifffile

Interface matches the manage_bubbly.py training contract:
    --dataset PATH    root with images/ and labels/ subdirectories
    --name    STR     experiment name (used for checkpoint folder)
    --config  PATH    config JSON with training hyperparameters
    --save_root PATH  where to save checkpoints (passed by manage_bubbly.py)
"""

import argparse
import json
import sys
import shutil
from pathlib import Path

import numpy as np


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True, type=Path)
    p.add_argument("--name",      required=True, type=str)
    p.add_argument("--config",    required=True, type=Path)
    p.add_argument("--save_root", type=Path, default=None)
    return p.parse_args(argv)


def load_training_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)

    training = cfg.get("training", {})
    return {
        "epochs": int(training.get("epochs", 100)),
        "imgsz": int(training.get("imgsz", 1024)),
        "batch": int(training.get("batch", 4)),
        "val_fraction": float(training.get("val_fraction", 0.15)),
    }


# ── Mask → YOLO polygon conversion ───────────────────────────────────────────
def mask_to_yolo_lines(mask: np.ndarray, img_h: int, img_w: int) -> list:
    """
    Convert a uint16 instance mask to YOLO-seg label lines.
    Each line: "0 x1 y1 x2 y2 ..." (class=0 bubble, coords normalized).
    """
    import cv2

    lines = []
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]  # skip background (0)

    for iid in instance_ids:
        binary = (mask == iid).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 4:
            continue
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        coords = []
        for x, y in pts:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append("0 " + " ".join(coords))

    return lines


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    train_cfg = load_training_config(args.config)

    if args.save_root:
        save_dir = args.save_root / args.name
    else:
        save_dir = Path(__file__).resolve().parent.parent / "pipeline" / "models" / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"STARTING YOLOv9 TRAINING: {args.name}")
    print(f"Config:   {args.config}")
    print(f"Dataset:  {args.dataset}")
    print(f"Epochs:   {train_cfg['epochs']}")
    print(f"imgsz:    {train_cfg['imgsz']}")
    print(f"Batch:    {train_cfg['batch']}")
    print(f"Val frac: {train_cfg['val_fraction']}")
    print(f"Output:   {save_dir}")
    print("--------------------------------")

    try:
        import cv2
        import tifffile
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: {e}")
        print("  pip install ultralytics tifffile")
        sys.exit(1)

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {'GPU' if device == '0' else 'CPU'}")

    # ── Locate pre-staged weights ─────────────────────────────────────────────
    SCRATCH = Path.home() / "scratch" / "bubble-models"
    weights = SCRATCH / "yolo" / "yolov9c-seg.pt"
    if not weights.exists():
        print(f"ERROR: YOLOv9 weights not found at {weights}")
        print("  Run:  bash bubbly_flows/scripts/download_models.sh")
        sys.exit(1)
    print(f"Weights:  {weights}")

    # ── Convert dataset to YOLO format ────────────────────────────────────────
    images_dir = args.dataset / "images"
    labels_dir = args.dataset / "labels"

    image_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.tif")))
    if not image_paths:
        print("ERROR: No images found in images/")
        sys.exit(1)

    print(f"Converting {len(image_paths)} instance masks to YOLO polygon format ...")

    yolo_ds = save_dir / "_yolo_dataset"
    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        (yolo_ds / subdir).mkdir(parents=True, exist_ok=True)

    split_idx = max(1, int(len(image_paths) * (1 - train_cfg["val_fraction"])))
    splits = {
        "train": image_paths[:split_idx],
        "val":   image_paths[split_idx:],
    }

    total_instances = 0
    skipped = 0
    for split_name, paths in splits.items():
        for img_path in paths:
            lbl_path = labels_dir / (img_path.stem + ".tif")
            if not lbl_path.exists():
                skipped += 1
                continue

            # Copy image
            shutil.copy2(img_path, yolo_ds / "images" / split_name / img_path.name)

            # Convert mask
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
            mask = tifffile.imread(str(lbl_path)).astype(np.uint16)
            lines = mask_to_yolo_lines(mask, h, w)
            total_instances += len(lines)

            dst_lbl = yolo_ds / "labels" / split_name / (img_path.stem + ".txt")
            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))

    if skipped:
        print(f"  Skipped {skipped} images (no label or unreadable)")
    print(f"  Total bubble instances written: {total_instances}")

    # Write data.yaml
    data_yaml = yolo_ds / "bubbles.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {yolo_ds.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['bubble']\n")

    print(f"  YOLO dataset: {yolo_ds}")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print("Starting YOLOv9 training ...")
    model = YOLO(str(weights))
    model.train(
        data=str(data_yaml),
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        name=args.name,
        project=str(save_dir.parent),
        device=device,
        exist_ok=True,
        verbose=True,
    )

    best = save_dir.parent / args.name / "weights" / "best.pt"
    print("--------------------------------")
    print("TRAINING COMPLETE.")
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()
