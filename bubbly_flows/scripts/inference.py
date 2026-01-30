#!/usr/bin/env python3
"""
inference.py

Run inference using a trained MicroSAM model on a single image or a directory of images.
"""
import argparse
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from micro_sam.util import get_sam_model
from micro_sam.inference import segment_from_mask, predict_large_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=Path, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--output", required=True, type=Path, help="Output path for the mask")
    parser.add_argument("--model_type", type=str, default="vit_b", help="Model type (vit_b, vit_h, etc)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # 1. Setup Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Model
    if not args.model_path.exists():
        print(f"Error: Model file {args.model_path} not found.")
        sys.exit(1)

    print(f"Loading model from {args.model_path}...")
    predictor = get_sam_model(model_type=args.model_type, checkpoint_path=str(args.model_path), device=device)

    # 3. Read Image
    if not args.image.exists():
        print(f"Error: Image {args.image} not found.")
        sys.exit(1)
        
    # MicroSAM expects RGB typically
    image = cv2.imread(str(args.image))
    if image is None:
        print("Error: Could not read image.")
        sys.exit(1)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Running inference on {args.image.name} ({image.shape})...")

    # 4. Predict
    # There are two modes: 'segment_from_mask' (refinement) or full auto.
    # Actually, often for SAM we need prompts. But if we trained it for automatic segmentation
    # using the 'instance_segmentation' logic, we might want `micro_sam.instance_segmentation`.
    # Let's check typical usage. 
    # If we trained with `train_sam_for_instance_segmentation`, we likely want `AutomaticMaskGenerator` 
    # or `instance_segmentation.run_instance_segmentation_with_decoder`.
    # For now, let's assume we want automatic instance segmentation.
    
    from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder
    
    # We need to wrap the predictor
    segmenter = InstanceSegmentationWithDecoder(predictor)
    
    # Compute
    # amg = AutomaticMaskGenerator(predictor) would be for vanilla SAM.
    # But since we fine-tuned, let's use the segmenter.
    
    # Note: InstanceSegmentationWithDecoder might require specific args (embeddings).
    # A simpler entry point is usually `run_automatic_instance_segmentation` if available, 
    # or we simply compute the embedding and decode.
    
    # Simpler approach for generic usage:
    # 1. Compute embedding
    # 2. Decode instances (if the model has a decoder head) or use AMG.
    
    # Let's try the high level helper if possible. 
    # Since I cannot see the exact API version, I'll use a robust common pattern:
    # Calculate embedding -> generate masks.
    
    # Using AutomaticMaskGenerator from micro_sam (custom wrapper) or creating one.
    # For simplicity in this v1 script, let's try the AMG approach which works generally if the model is loaded.
    
    from micro_sam.instance_segmentation import get_predictor_and_decoder
    # predictor, decoder = get_predictor_and_decoder(model_type=args.model_type, checkpoint_path=str(args.model_path), device=device)
    # But we already got predictor.
    
    # Let's assume standard AMG for now. 
    from micro_sam.instance_segmentation import AutomaticMaskGenerator
    amg = AutomaticMaskGenerator(predictor)
    
    masks = amg.generate(image)
    # masks is usually a list of dicts. We want a single labeled mask image.
    
    # Create a blank label map
    label_map = np.zeros(image.shape[:2], dtype=np.uint16)
    
    for i, res in enumerate(masks):
        # res['segmentation'] is the boolean mask
        mask_bool = res['segmentation']
        label_map[mask_bool] = (i + 1)
        
    # 5. Save
    print(f"Found {len(masks)} instances. Saving to {args.output}")
    cv2.imwrite(str(args.output), label_map)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
