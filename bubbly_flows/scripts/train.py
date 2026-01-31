import argparse
import sys
import os
import glob
import torch
import torch
import torch_em
from micro_sam.training.training import train_sam
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path, help="Path to dataset root (must have images/ and labels/ subdirs)")
    parser.add_argument("--name", required=True, type=str, help="Experiment name")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"STARTING TRAINING: {args.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs:  {args.epochs}")
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
    model_save_root = repo_root / "microsam" / "models" / args.name
    model_save_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoints will be saved to: {model_save_root}")

    # 4. Train
    # Create DataLoaders using torch_em
    # micro_sam expects patches. We use a default shape of (512, 512).
    print("Creating DataLoaders...")
    
    # Simple validation split (last 10%)
    split_idx = max(1, int(len(train_image_paths) * 0.9))
    val_image_paths = train_image_paths[split_idx:]
    val_label_paths = train_label_paths[split_idx:]
    train_image_paths = train_image_paths[:split_idx]
    train_label_paths = train_label_paths[:split_idx]
    
    print(f"Split: {len(train_image_paths)} training, {len(val_image_paths)} validation.")

    # We use torch_em to handle reading images and patching
    # raw_key=None, label_key=None implies standard image files (tif/png)
    train_loader = torch_em.default_segmentation_loader(
        raw_paths=train_image_paths,
        raw_key=None,
        label_paths=train_label_paths,
        label_key=None,
        batch_size=2,
        patch_shape=(512, 512),
        ndim=2,
        is_seg_dataset=True,
        num_workers=4,
        shuffle=True
    )
    
    val_loader = torch_em.default_segmentation_loader(
        raw_paths=val_image_paths,
        raw_key=None,
        label_paths=val_label_paths,
        label_key=None,
        batch_size=1,
        patch_shape=(512, 512),
        ndim=2,
        is_seg_dataset=True,
        num_workers=1,
        shuffle=False
    )
    
    print("Starting MicroSAM training...")
    
    try:
        train_sam(
            name=args.name,
            save_root=str(model_save_root),
            train_loader=train_loader,
            val_loader=val_loader,
            model_type="vit_b",
            n_epochs=args.epochs,
            batch_size=2,   # Conservative batch size for 3090/standard GPUs
            save_every_k_epochs=10,
            device=device
        )
    except Exception as e:
        print(f"CRITICAL TRAINING ERROR: {e}")
        sys.exit(1)

    print("--------------------------------")
    print("TRAINING COMPLETE.")
    print(f"Best model expected at: {model_save_root}/checkpoints/{args.name}/best.pt")

if __name__ == "__main__":
    main()
