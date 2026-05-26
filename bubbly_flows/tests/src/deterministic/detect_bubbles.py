#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import json

def detect_bubbles(image_path, output_dir, debug=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Preprocessing: Flatten noise
    # Standard GaussianBlur to remove sensor noise before Adaptive Threshold
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    
    # 2. Adaptive Thresholding
    # Block Size 31: Covers largest expected bubble (~15px radius) with margin
    # C = 7: High sensitivity constant to separate bubble from background and shrink mask
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 31, 7)
    
    # 3. Refinement
    # Morph Open (3x3): Remove salt noise ("dust")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "debug_adaptive_thresh.png"), thresh)
        cv2.imwrite(os.path.join(output_dir, "debug_adaptive_open.png"), opening)
    
    # 4. Extraction & Filtering
    # Simple Contour detection (Adaptive usually separates well)
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        
        # Area Filter: 20 to 350
        # Increased Min Area to 20 (Radius ~2.5px) to remove salt noise.
        # Max Area 350 (Radius ~10.5px) per user constraint.
        if area < 20 or area > 350: continue
        
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        
        # Circularity Filter: > 0.4
        # Relaxed slightly to allow "less circular figures" per user request
        if circularity < 0.4: continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        
        # Solidity Filter: > 0.75
        # Relaxed slightly for irregular shapes
        if solidity < 0.75: continue

        # Intensity Filter: < 160
        # Ensure it's Dark
        c_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=c_mask)[0]
        if mean_val > 160: continue
        
        # TIGHTER CHECK: Use Equivalent Diameter
        # radius = sqrt(Area / Pi)
        # This creates a circle with the exact same area as the contour, 
        # avoiding the "background inclusion" of minEnclosingCircle.
        radius = np.sqrt(area / np.pi)
        
        # Calculate center moment
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        else:
            (cX, cY), _ = cv2.minEnclosingCircle(c)
            
        detections.append((cX, cY, radius))

    print(f"Detected {len(detections)} bubbles in {os.path.basename(image_path)}")

    # Visualization & Output
    result_img = img.copy()
    for (x, y, r) in detections:
        # Green circle for detected
        cv2.circle(result_img, (int(x), int(y)), int(r), (0, 255, 0), 2)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"detected_{base_name}.png")
    cv2.imwrite(out_path, result_img)
    
    json_path = os.path.join(output_dir, f"detected_{base_name}.json")
    with open(json_path, 'w') as f:
        json.dump(detections, f)
        
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    detect_bubbles(args.input, args.output, args.debug)
