import argparse
import sys
import os
import glob
import torch
from micro_sam.training import sam_trainer
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
    # micro_sam expects lists of file paths
    # We use a default VIT_B model configuration
    
    # Check if we need to download a checkpoint first or if micro_sam handles it
    # micro_sam handles it if we pass model_type
    
    print("Starting MicroSAM training...")
    
    # Check what functions are available if we hit issues again
    # print(f"DEBUG: sam_trainer attributes: {dir(sam_trainer)}")

    try:
        sam_trainer.train_sam(
            name=args.name,
            save_root=str(model_save_root),
            train_image_paths=train_image_paths,
            train_label_paths=train_label_paths,
            # We don't have a separate val set in this simple pipeline yet, 
            # usually we split, but for now we might leave val empty or use a subset.
            # Let's use the last 10% for validation if possible.
            val_image_paths=train_image_paths[-max(1, int(len(train_image_paths)*0.1)):],
            val_label_paths=train_label_paths[-max(1, int(len(train_label_paths)*0.1)):],
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
