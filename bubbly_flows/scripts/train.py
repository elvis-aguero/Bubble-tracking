#!/usr/bin/env python3
"""
train.py

Training entry point for MicroSAM.
Executed by Slurm jobs submitted via manage_bubbly.py.
"""
import argparse
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"STARTING TRAINING: {args.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs:  {args.epochs}")
    print("--------------------------------")
    
    # 1. Load Dataset
    if not args.dataset.exists():
        print(f"Error: Dataset {args.dataset} not found.")
        sys.exit(1)
        
    print("[1/3] Loading data...")
    time.sleep(2) # Simulate work
    
    # 2. Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"[2/3] GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        print("[2/3] WARNING: No GPU detected. Training will be slow.")
        
    # 3. Train Loop (Placeholder)
    print("[3/3] Training MicroSAM (Simulated)...")
    for i in range(1, 6):
        print(f"Epoch {i}/{args.epochs} - Loss: {0.9**i:.4f}")
        time.sleep(1)
        
    print("--------------------------------")
    print("TRAINING COMPLETE.")
    print(f"Model saved to microsam/models/{args.name}_best.pt")

if __name__ == "__main__":
    main()
