import cv2
import numpy as np
import argparse
import sys
import os

def detect_bubbles(args):
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not open '{args.image}'")
        sys.exit(1)
    
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    output_image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. BLACK HAT FILTER
    # Extracts the "White Rims" from the "Grey Background"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.kernel_size, args.kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # 2. DEBUG: Verify the Black Hat input
    if args.debug:
        bh_filename = f"{base_name}_debug_blackhat.jpg"
        cv2.imwrite(bh_filename, blackhat)
        print(f"--> [DEBUG] Using Black Hat image for detection: {bh_filename}")
        
        # Visualize what edges Param1 will see
        edges = cv2.Canny(blackhat, args.param1 // 2, args.param1)
        cv2.imwrite(f"{base_name}_debug_edges.jpg", edges)

    # 3. HOUGH DETECTOR (DIRECTLY ON BLACK HAT)
    # We skip the binary threshold. We let Canny (inside Hough) handle the contrast.
    circles = cv2.HoughCircles(
        blackhat, 
        cv2.HOUGH_GRADIENT, 
        dp=args.dp, 
        minDist=args.min_dist, 
        param1=args.param1, 
        param2=args.param2, 
        minRadius=args.min_radius, 
        maxRadius=args.max_radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"--> Found {len(circles)} bubbles.")
        for (x, y, r) in circles:
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)
    else:
        print("--> No bubbles detected.")

    # Save Final
    out_file = args.save if args.save else f"{base_name}_output.jpg"
    cv2.imwrite(out_file, output_image)
    print(f"--> Result saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bubbles using Direct Black Hat.")
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-s", "--save", type=str)
    parser.add_argument("--debug", action="store_true")

    # Filter Params
    parser.add_argument("--kernel_size", type=int, default=15, help="Size of the Black Hat kernel (approx bubble size).")

    # Hough Params
    # Param1: High threshold for Canny. Since we use BlackHat, the contrast is good.
    # Try 30-50. If you get noise, raise it.
    parser.add_argument("--param1", type=int, default=40)
    
    # Param2: Accumulator threshold. 
    # Try 10-15. If you miss bubbles, lower it slightly.
    parser.add_argument("--param2", type=int, default=12)
    
    parser.add_argument("--dp", type=float, default=1.0)
    parser.add_argument("--min_dist", type=int, default=10)
    parser.add_argument("--min_radius", type=int, default=3)
    parser.add_argument("--max_radius", type=int, default=10)

    args = parser.parse_args()
    detect_bubbles(args)