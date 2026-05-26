#!/usr/bin/env python3
"""
Detect small bubble centers with ONE algorithm: Fast Radial Symmetry Transform (FRST).

Why FRST here:
- FRST is gradient-based and enhances locally circular / radially symmetric structures by voting
  toward (or away from) the center at a set of radii. This is well-suited for bubbles that look
  like rings (dark rim with bright interior) or dark dots, under varying illumination. :contentReference[oaicite:1]{index=1}
- We make it polarity-robust by voting in BOTH gradient directions (toward and away) inside FRST
  itself (still one algorithm).

Dependencies:
  pip install numpy opencv-python

Usage:
  python frst_bubble_centers.py --input in.png --output centers.png

Key knobs (highest ROI):
  --r_min / --r_max         radius range in pixels (bubble diameter ~ 2r)
  --mag_percentile          ignore weak gradients (noise/background); 90–98 typical :contentReference[oaicite:2]{index=2}
  --peak_percentile         how strict peak picking is (99–99.9 typical)
  --nms_size                local-max neighborhood (odd int); ~ (2*r_min+1) is a good start
"""

import argparse
import numpy as np
import cv2
from math import sqrt

def frst_symmetry_map(gray_u8: np.ndarray,
                      r_min: int,
                      r_max: int,
                      r_step: int,
                      alpha: float,
                      mag_percentile: float) -> np.ndarray:
    """
    Compute FRST symmetry map on an 8-bit grayscale image.
    Returns float32 map in [0, 1].
    """
    # Mild denoise helps gradients
    gray = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=1.0, sigmaY=1.0)

    # Scharr gradients (more accurate 3x3 derivatives than Sobel in OpenCV)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)


    # Threshold weak gradients to suppress background noise (percentile rule is common) :contentReference[oaicite:4]{index=4}
    thr = np.percentile(mag, mag_percentile)
    if thr <= 0:
        thr = 1e-6
    mask = mag > thr

    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros_like(gray, dtype=np.float32)

    gxv = gx[ys, xs]
    gyv = gy[ys, xs]
    mv = mag[ys, xs]

    # Unit gradient directions
    inv = 1.0 / (mv + 1e-12)
    dx = gxv * inv
    dy = gyv * inv

    h, w = gray.shape[:2]
    S = np.zeros((h, w), dtype=np.float32)

    radii = list(range(int(r_min), int(r_max) + 1, int(r_step)))
    for r in radii:
        O = np.zeros((h, w), dtype=np.float32)
        M = np.zeros((h, w), dtype=np.float32)

        # Polarity-robust FRST: vote both along +g and -g directions
        px = np.rint(xs + dx * r).astype(np.int32)
        py = np.rint(ys + dy * r).astype(np.int32)
        nx = np.rint(xs - dx * r).astype(np.int32)
        ny = np.rint(ys - dy * r).astype(np.int32)

        # Clip to image
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        nx = np.clip(nx, 0, w - 1)
        ny = np.clip(ny, 0, h - 1)

        # Accumulate orientation and magnitude projections
        # (Using +1 votes in both directions for polarity invariance)
        np.add.at(O, (py, px), 1.0)
        np.add.at(O, (ny, nx), 1.0)
        np.add.at(M, (py, px), mv)
        np.add.at(M, (ny, nx), mv)

        # Normalize per-radius
        Omax = float(O.max())
        Mmax = float(M.max())
        if Omax > 0:
            O /= Omax
        if Mmax > 0:
            M /= Mmax

        # Symmetry contribution: F_n = (O_n^alpha) * M_n  (standard FRST form) :contentReference[oaicite:5]{index=5}
        F = (O ** alpha) * M

        # Optional smoothing to spread influence (Gaussian is part of the original FRST) :contentReference[oaicite:6]{index=6}
        sigma = 0.3 * r
        if sigma >= 0.5:
            k = int(round(6 * sigma + 1)) | 1  # odd
            F = cv2.GaussianBlur(F, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)

        # Weight smaller radii a bit more to favor small bubbles
        S += (1.0 / min(1.0, float(sqrt(r)))) * F

    # Normalize to [0,1]
    Smax = float(S.max())
    if Smax > 0:
        S /= Smax
    return S.astype(np.float32)


def pick_peaks(S: np.ndarray,
               peak_percentile: float,
               nms_size: int,
               border: int,
               max_peaks: int) -> np.ndarray:
    """
    Non-maximum suppression (NMS) on symmetry map. Returns Nx2 array of (x,y).
    """
    h, w = S.shape
    nms_size = int(nms_size) | 1  # force odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_size, nms_size))
    dil = cv2.dilate(S, kernel)

    thr = np.percentile(S, peak_percentile)
    peaks = (S >= thr) & (S == dil)

    ys, xs = np.where(peaks)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # Sort by score desc
    scores = S[ys, xs]
    order = np.argsort(-scores)
    xs = xs[order]
    ys = ys[order]

    # Border exclusion
    keep = (xs >= border) & (xs < w - border) & (ys >= border) & (ys < h - border)
    xs = xs[keep]
    ys = ys[keep]

    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # Cap
    if max_peaks > 0 and xs.size > max_peaks:
        xs = xs[:max_peaks]
        ys = ys[:max_peaks]

    return np.stack([xs, ys], axis=1).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input image path")
    ap.add_argument("--output", required=True, help="Output image path (dots drawn)")
    ap.add_argument("--r_min", type=int, default=4, help="Min radius (px)")
    ap.add_argument("--r_max", type=int, default=25, help="Max radius (px)")
    ap.add_argument("--r_step", type=int, default=2, help="Radius step")
    ap.add_argument("--alpha", type=float, default=1.4, help="Radial strictness (higher => dot-like)")
    ap.add_argument("--mag_percentile", type=float, default=88.0, help="Ignore gradients below this percentile")
    ap.add_argument("--peak_percentile", type=float, default=99.0, help="Keep peaks above this percentile")
    ap.add_argument("--nms_size", type=int, default=7, help="NMS neighborhood size (odd)")
    ap.add_argument("--border", type=int, default=8, help="Exclude peaks within this border (px)")
    ap.add_argument("--max_peaks", type=int, default=2000, help="Max centers to draw (0 = no cap)")
    ap.add_argument("--dot_radius", type=int, default=2, help="Dot radius in output image")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    # Grayscale + CLAHE to reduce illumination issues (still preprocessing)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    S = frst_symmetry_map(
        gray_u8=gray,
        r_min=args.r_min,
        r_max=args.r_max,
        r_step=args.r_step,
        alpha=args.alpha,
        mag_percentile=args.mag_percentile
    )

    centers = pick_peaks(
        S=S,
        peak_percentile=args.peak_percentile,
        nms_size=args.nms_size,
        border=args.border,
        max_peaks=args.max_peaks
    )

    out = img.copy()
    for (x, y) in centers:
        cv2.circle(out, (int(x), int(y)), args.dot_radius, (0, 0, 255), thickness=-1)

    cv2.imwrite(args.output, out)
    print(f"Detected centers: {centers.shape[0]}")
    # Optional: save symmetry map for debugging
    sym_vis = (np.clip(S, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(args.output.rsplit('.', 1)[0] + "_symmetry.png", sym_vis)


if __name__ == "__main__":
    main()

