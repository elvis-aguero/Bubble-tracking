#!/usr/bin/env python3
"""
inference.py

Run inference using a trained MicroSAM model on a single image.
The model must have been trained with with_segmentation_decoder=True (the default in train.py).
"""
import argparse
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from micro_sam.instance_segmentation import get_predictor_and_decoder, InstanceSegmentationWithDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=Path,
                        help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--image", required=True, type=Path,
                        help="Input image path")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output file path for the instance mask (.png or .tif)")
    parser.add_argument("--model_type", type=str, default="vit_b",
                        help="SAM encoder variant used during training (default: vit_b)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.model_path.exists():
        print(f"Error: checkpoint not found: {args.model_path}")
        sys.exit(1)
    if not args.image.exists():
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    # Load SAM predictor + UNETR decoder from the fine-tuned checkpoint
    print(f"Loading model from {args.model_path} ...")
    predictor, decoder = get_predictor_and_decoder(
        model_type=args.model_type,
        checkpoint_path=str(args.model_path),
        device=device,
    )
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)

    # Load image — normalise to uint8 RGB (SAM expects 0-255)
    raw = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if raw is None:
        print(f"Error: could not read image: {args.image}")
        sys.exit(1)
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.dtype != np.uint8:
        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    print(f"Running inference on {args.image.name} ({image_rgb.shape}) ...")

    segmenter.initialize(image_rgb)
    label_map = segmenter.generate(output_mode="instance_segmentation")

    n_instances = int(label_map.max())
    print(f"Found {n_instances} instances. Saving to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Raw label map (uint16, pixel value = instance ID) — used by evaluate.py
    cv2.imwrite(str(args.output), label_map.astype(np.uint16))

    # Overlay: original image + coloured instance masks
    vis_path = args.output.parent / (args.output.stem + "_vis.png")
    # Convert original to uint8 BGR for overlay
    base = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    rng = np.random.default_rng(42)
    for inst_id in range(1, n_instances + 1):
        color = rng.integers(60, 255, size=3).tolist()
        overlay[label_map == inst_id] = color
    vis = cv2.addWeighted(base, 0.5, overlay, 0.5, 0)
    cv2.imwrite(str(vis_path), vis)
    print(f"Visualisation saved to {vis_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
