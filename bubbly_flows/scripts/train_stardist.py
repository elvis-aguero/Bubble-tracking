#!/usr/bin/env python3
"""
train_stardist.py — Fine-tune StarDist 2D on bubble instance masks.

Loads the HZDR 2022 bubble-specific StarDist checkpoint as the starting point
(Hessenkemper et al. 2022, RODARE 2566; AP@0.5 ~0.91 on air-water flow).
Falls back to training from scratch if the HZDR weights are not found.

Requires (not in bubbly-train-env by default — install once):
    pip install stardist tensorflow csbdeep tifffile

Interface matches the manage_bubbly.py training contract:
    --dataset PATH    root with images/ and labels/ subdirectories
    --name    STR     experiment name (used for checkpoint folder)
    --epochs  INT     training epochs (default 50)
    --save_root PATH  where to save checkpoints (passed by manage_bubbly.py)
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True, type=Path)
    p.add_argument("--name",      required=True, type=str)
    p.add_argument("--epochs",    type=int, default=50)
    p.add_argument("--save_root", type=Path, default=None)
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data(images_dir: Path, labels_dir: Path):
    """Load image/label pairs. Returns (X, Y) as lists of numpy arrays."""
    try:
        import cv2
        import tifffile
        from stardist import fill_label_holes
        from csbdeep.utils import normalize
    except ImportError as e:
        print(f"ERROR: {e}")
        print("  pip install stardist tensorflow csbdeep tifffile")
        sys.exit(1)

    image_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.tif")))
    if not image_paths:
        print("ERROR: No images found in images/")
        sys.exit(1)

    X, Y = [], []
    skipped = 0
    for img_path in image_paths:
        lbl_path = labels_dir / (img_path.stem + ".tif")
        if not lbl_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            skipped += 1
            continue
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)

        lbl = tifffile.imread(str(lbl_path)).astype(np.uint16)
        lbl = fill_label_holes(lbl)

        X.append(img)
        Y.append(lbl)

    if skipped:
        print(f"  Skipped {skipped} images (no paired label or unreadable)")

    if not X:
        print("ERROR: No valid image/label pairs found.")
        sys.exit(1)

    # Normalize to [0, 1] using percentiles
    X = [normalize(x, 1, 99.8, axis=(0, 1)) for x in X]
    return X, Y


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.save_root:
        save_dir = args.save_root / args.name
    else:
        save_dir = Path(__file__).resolve().parent.parent / "microsam" / "models" / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"STARTING StarDist TRAINING: {args.name}")
    print(f"Dataset:  {args.dataset}")
    print(f"Epochs:   {args.epochs}")
    print(f"Output:   {save_dir}")
    print("--------------------------------")

    try:
        from stardist.models import StarDist2D, Config2D
    except ImportError as e:
        print(f"ERROR: {e}")
        print("  pip install stardist tensorflow csbdeep tifffile")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    images_dir = args.dataset / "images"
    labels_dir = args.dataset / "labels"
    print("Loading data ...")
    X, Y = load_data(images_dir, labels_dir)
    print(f"  Loaded {len(X)} pairs.")

    split_idx = max(1, int(len(X) * 0.9))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val,   Y_val   = X[split_idx:], Y[split_idx:]
    print(f"  Split: {len(X_train)} train / {len(X_val)} val")

    # ── Load HZDR pre-trained model ───────────────────────────────────────────
    SCRATCH = Path.home() / "scratch" / "bubble-models"
    hzdr_base = SCRATCH / "stardist" / "hzdr_2022" / "Models" / "SDmodel"
    hzdr_config  = hzdr_base / "stardist" / "config.json"
    hzdr_weights = hzdr_base / "stardist" / "weights_best.h5"

    if hzdr_config.exists() and hzdr_weights.exists():
        print(f"Loading HZDR 2022 pre-trained weights from:\n  {hzdr_weights}")

        with open(hzdr_config) as f:
            conf_dict = json.load(f)

        # Override training parameters; keep architecture unchanged
        conf_dict["train_epochs"]           = args.epochs
        conf_dict["train_steps_per_epoch"]  = max(10, len(X_train) // max(1, conf_dict.get("train_batch_size", 2)))
        conf_dict["use_gpu"]                = True

        # Strip checkpoint keys — Config2D does not accept them
        for key in ("train_checkpoint", "train_checkpoint_last", "train_checkpoint_epoch"):
            conf_dict.pop(key, None)

        conf = Config2D(**conf_dict)
        model = StarDist2D(conf, name=args.name, basedir=str(save_dir.parent))
        model.keras_model.load_weights(str(hzdr_weights))
        print("  HZDR weights loaded.")
    else:
        print(f"WARNING: HZDR weights not found at {hzdr_weights}")
        print("  Training from scratch with default StarDist 2D config.")
        conf = Config2D(
            n_rays=64,
            grid=(4, 8),
            n_channel_in=1,
            train_patch_size=(512, 512),
            train_batch_size=2,
            train_epochs=args.epochs,
            train_steps_per_epoch=max(10, len(X_train) // 2),
            use_gpu=True,
        )
        model = StarDist2D(conf, name=args.name, basedir=str(save_dir.parent))

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting StarDist training ...")
    model.train(X_train, Y_train, validation_data=(X_val, Y_val), augmenter=None)

    print("Optimizing NMS thresholds on validation set ...")
    model.optimize_thresholds(X_val, Y_val)

    print("--------------------------------")
    print("TRAINING COMPLETE.")
    print(f"Model saved to: {save_dir.parent / args.name}")


if __name__ == "__main__":
    main()
