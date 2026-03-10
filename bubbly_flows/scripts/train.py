import argparse
import json
import sys
import os
import glob
import numpy as np
import torch
from torch_em.transform.raw import normalize_percentile
from micro_sam.training import default_sam_loader, train_sam
from pathlib import Path


def raw_transform(raw):
    """Normalize image to [0, 255] as required by MicroSAM.

    torch_em's default standardize() produces z-score values outside [0, 255],
    which violates MicroSAM's expected input range. Percentile normalization is
    robust to outliers and consistent with how SAM was originally trained.
    """
    raw = normalize_percentile(raw.astype(np.float32), lower=1.0, upper=99.8)
    return (np.clip(raw, 0.0, 1.0) * 255.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path, help="Path to dataset root (must have images/ and labels/ subdirs)")
    parser.add_argument("--name", required=True, type=str, help="Experiment name")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to model config JSON (e.g. configs/microsam.json)")
    parser.add_argument("--save_root", type=Path, default=None,
                        help="Root directory for saving checkpoints. "
                             "Defaults to bubbly_flows/microsam/models/<name>/")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    t = cfg["training"]
    patch_shape  = t.get("patch_shape", 1024)
    epochs       = t.get("epochs", 100)
    batch_size   = t.get("batch_size", 1)
    num_workers  = t.get("num_workers", 4)
    val_fraction = t.get("val_fraction", 0.15)
    early_stop   = t.get("early_stopping_patience", 10)
    freeze       = t.get("freeze", ["image_encoder"])
    backbone     = t.get("model_type", cfg.get("backbone", "vit_b"))

    print(f"STARTING TRAINING: {args.name}")
    print(f"Config:      {args.config}")
    print(f"Dataset:     {args.dataset}")
    print(f"Epochs:      {epochs}")
    print(f"Patch shape: ({patch_shape}, {patch_shape})")
    print(f"Batch size:  {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Val fraction:{val_fraction}")
    print(f"Early stop:  {early_stop}")
    print(f"Freeze:      {freeze}")
    print(f"Backbone:    {backbone}")
    print("--------------------------------")

    # 1. Validation
    if not args.dataset.exists():
        print(f"Error: Dataset {args.dataset} not found.")
        sys.exit(1)
        
    # Check for images and labels
    train_image_paths = sorted(glob.glob(os.path.join(args.dataset, "images", "*")))
    train_label_paths = sorted(glob.glob(os.path.join(args.dataset, "labels", "*")))
    
    if len(train_image_paths) == 0:
        print("Error: No images found in dataset/images/")
        sys.exit(1)
        
    print(f"Found {len(train_image_paths)} images for training.")

    # 2. Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be extremely slow.")

    # 3. Setup Paths
    # We save models to microsam/models/<name>
    # The script is usually run from repo root or scripts dir, but let's be robust
    # If this script is in scripts/, we want REPO/microsam/models
    
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    if args.save_root:
        model_save_root = args.save_root / args.name
    else:
        model_save_root = repo_root / "microsam" / "models" / args.name
    model_save_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoints will be saved to: {model_save_root}")

    # 4. Train
    # Create DataLoaders using torch_em
    # micro_sam expects patches. We use a default shape of (512, 512).
    print("Creating DataLoaders...")
    
    # Simple validation split
    split_idx = max(1, int(len(train_image_paths) * (1 - val_fraction)))
    val_image_paths = train_image_paths[split_idx:]
    val_label_paths = train_label_paths[split_idx:]
    train_image_paths = train_image_paths[:split_idx]
    train_label_paths = train_label_paths[:split_idx]
    
    print(f"Split: {len(train_image_paths)} training, {len(val_image_paths)} validation.")

    # Use micro_sam's own loader — it applies the 4-channel label transform
    # required by the instance segmentation decoder (foreground + distance fields).
    ps = (patch_shape, patch_shape)
    train_loader = default_sam_loader(
        raw_paths=train_image_paths,
        raw_key=None,
        label_paths=train_label_paths,
        label_key=None,
        patch_shape=ps,
        with_segmentation_decoder=True,
        raw_transform=raw_transform,
        is_train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = default_sam_loader(
        raw_paths=val_image_paths,
        raw_key=None,
        label_paths=val_label_paths,
        label_key=None,
        patch_shape=ps,
        with_segmentation_decoder=True,
        raw_transform=raw_transform,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    
    print("Starting MicroSAM training...")
    
    try:
        train_sam(
            name=args.name,
            save_root=str(model_save_root),
            train_loader=train_loader,
            val_loader=val_loader,
            model_type=backbone,
            n_epochs=epochs,
            device=device,
            # Freeze the image encoder so gradients are not stored through ViT-B's
            # 1024x1024 attention maps. With batch_size=1 the encoder alone fills
            # the 24GB GPU. Fine-tuning only the mask decoder + segmentation decoder
            # is the standard approach for small microscopy datasets.
            freeze=freeze,
        )
    except Exception as e:
        print(f"CRITICAL TRAINING ERROR: {e}")
        sys.exit(1)

    print("--------------------------------")
    print("TRAINING COMPLETE.")
    print(f"Best model expected at: {model_save_root}/checkpoints/{args.name}/best.pt")

if __name__ == "__main__":
    main()
