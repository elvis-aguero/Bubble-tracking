import os
import sys
import json
import time
import torch
import inspect
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_util
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

"""
SAM3 Text-Prompted Segmentation Script with Center Cropping

This script uses SAM3 to segment objects in a cropped image based on a text prompt.
It outputs a PNG with transparent colored overlays and a COCO-format JSON with mask annotations.

ENVIRONMENT SETUP (copy-paste for this environment):
    export HF_HOME=/users/eaguerov/scratch/hf
    export CUDA_VISIBLE_DEVICES=0  # if needed

USAGE:
    python test2.py <image_path> <prompt> <crop>

PARAMETERS:
    image_path : Path to the input image
    prompt     : Text prompt for segmentation
    crop       : Crop ratio between 0 and 1 (fraction of original width/height to keep)
                 - 1.0 = full image (no cropping)
                 - 0.5 = crop to 50% width and height (25% of area)

EXAMPLES:
    python test2.py /path/to/image.jpg "laptop" 0.8
    python test2.py /users/eaguerov/scratch/sam3_test.jpg "dog" 0.5

OUTPUTS:
    - <image>_<prompt>_crop_<crop>_masks.png      : Cropped image with transparent color-coded mask overlays
    - <image>_<prompt>_crop_<crop>_masks.json     : COCO format with RLE-encoded masks, boxes, scores

REQUIREMENTS:
    - HF_HOME: Path to Hugging Face cache directory (default: ~/hf)
    - CUDA GPU: Required for efficient inference
    - Model: SAM3 will be cached in HF_HOME/hub/models--facebook--sam3/
"""

# Ensure Hugging Face uses the existing HF_HOME cache
if 'HF_HOME' not in os.environ:
    hf_cache = os.path.expanduser("~/hf")
    if os.path.exists(hf_cache):
        os.environ['HF_HOME'] = hf_cache

# Global model cache
_model_cache = None
_processor_cache = None


def get_model_and_processor():
    """Get cached model and processor to avoid re-downloading."""
    global _model_cache, _processor_cache
    if _model_cache is None:
        print("Loading model from cache...")
        _model_cache = build_sam3_image_model()
        _processor_cache = Sam3Processor(_model_cache, confidence_threshold=0.4)
        print("Model loaded successfully")
    return _model_cache, _processor_cache


def main(image_path, prompt, crop):
    """
    Process a cropped image with a text prompt using SAM3 and export masks in COCO format.
    
    Args:
        image_path: Path to the input image
        prompt: Text prompt for segmentation
        crop: Crop ratio between 0 and 1 (fraction of original width/height to keep)
    """
    # Validate crop parameter
    try:
        crop = float(crop)
        if not (0 < crop <= 1):
            raise ValueError("Crop must be between 0 and 1 (exclusive of 0, inclusive of 1)")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
    
    # Get cached model and processor
    _, processor = get_model_and_processor()
    
    print("Sam3Processor.__init__:", inspect.signature(Sam3Processor.__init__))
    print("set_image:", inspect.signature(processor.set_image))
    print("set_text_prompt:", inspect.signature(processor.set_text_prompt))

    # Load and process image
    original_image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = original_image.size
    
    # Crop image around center
    crop_width = int(orig_width * crop)
    crop_height = int(orig_height * crop)
    left = (orig_width - crop_width) // 2
    top = (orig_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    image = original_image.crop((left, top, right, bottom))
    img_width, img_height = image.size
    print(f"Cropped image from {orig_width}x{orig_height} to {img_width}x{img_height}")
    
    # Time the inference
    print(f"Running inference with prompt: '{prompt}'...")
    inference_start = time.perf_counter()
    
    state = processor.set_image(image)
    out = processor.set_text_prompt(state=state, prompt=prompt)
    
    inference_time = time.perf_counter() - inference_start
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
    print("instances:", len(masks))
    print("boxes:", getattr(boxes, "shape", None))
    print("scores:", getattr(scores, "shape", None))
    
    # Generate output paths
    base_dir = os.path.dirname(image_path) or "."
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    crop_str = f"{crop:.2f}".rstrip('0').rstrip('.')
    json_path = os.path.join(base_dir, f"{base_name}_{prompt}_crop_{crop_str}_masks.json")
    
    # Prepare COCO format annotations
    annotations = []
    
    # Create combined mask visualization with unique colors for each mask
    combined_mask = np.zeros((img_height, img_width, 4), dtype=np.uint8)  # RGBA
    
    # Start with the cropped image
    img_array = np.array(image)
    combined_mask[:, :, :3] = img_array
    combined_mask[:, :, 3] = 255  # Full opacity for background
    
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        mask_np = mask[0].cpu().numpy().astype(bool)  # HxW
        
        # Encode mask to RLE format (COCO standard)
        rle = mask_util.encode(np.asfortranarray(mask_np.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string for JSON serialization
        
        # Extract bounding box coordinates
        if box is not None and len(box.shape) > 0:
            box_np = box.cpu().numpy() if isinstance(box, torch.Tensor) else box
            bbox = [float(box_np[0]), float(box_np[1]), float(box_np[2] - box_np[0]), float(box_np[3] - box_np[1])]
        else:
            bbox = [0, 0, img_width, img_height]
        
        annotation = {
            "id": i,
            "image_id": 0,
            "category_id": 1,
            "category_name": prompt,
            "bbox": bbox,
            "area": float(mask_util.area(rle)),
            "segmentation": rle,
            "score": float(score.cpu().item() if isinstance(score, torch.Tensor) else score),
            "iscrowd": 0
        }
        annotations.append(annotation)
        
        # Assign unique color to each mask with 50% alpha
        color = plt.cm.tab20(i % 20)[:3]
        color = tuple(int(c * 255) for c in color)
        combined_mask[mask_np, :3] = color
        combined_mask[mask_np, 3] = 128  # 50% transparency for masked regions
    
    # Save combined mask PNG with transparency
    mask_png_path = os.path.join(base_dir, f"{base_name}_{prompt}_crop_{crop_str}_masks.png")
    mask_img = Image.fromarray(combined_mask, 'RGBA')
    mask_img.save(mask_png_path)
    print(f"saved masks PNG: {mask_png_path}")
    
    # Save COCO format JSON
    coco_data = {
        "info": {
            "prompt": prompt,
            "crop": crop,
            "image_path": image_path,
            "original_size": [orig_width, orig_height],
            "cropped_size": [img_width, img_height],
            "crop_box": [left, top, right, bottom]
        },
        "annotations": annotations
    }
    
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"saved COCO annotations: {json_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test2.py <image_path> <prompt> <crop>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    prompt = sys.argv[2]
    crop = sys.argv[3]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    main(image_path, prompt, crop)
