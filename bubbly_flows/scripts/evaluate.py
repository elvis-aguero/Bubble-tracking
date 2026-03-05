#!/usr/bin/env python3
"""
evaluate.py — compare predicted instance masks against gold masks and report metrics.

For each predicted mask, finds the matching ground-truth mask by filename stem,
runs Hungarian-matched instance comparison, and reports per-image and aggregate metrics.

Usage:
    python bubbly_flows/scripts/evaluate.py \
        --preds output/preds/ \
        --gts   bubbly_flows/microsam/datasets/<test-dataset>/labels/ \
        --iou_threshold 0.5 \
        [--output results.csv]

Output mask format (both preds and gts):
    uint16 TIF or PNG where pixel value = instance ID (0 = background, 1+ = bubble IDs).
    This is the format produced by inference.py and the export pipeline.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Matching ──────────────────────────────────────────────────────────────────

def match_instances(pred_mask: np.ndarray, gt_mask: np.ndarray,
                    iou_threshold: float = 0.5):
    """
    Match predicted bubbles to labeled bubbles using the Hungarian algorithm.

    Returns:
        matches  - list of (pred_id, gt_id, iou) for correct detections
        fp_ids   - predicted IDs with no matching ground-truth (false positives)
        fn_ids   - ground-truth IDs with no matching prediction (false negatives)
    """
    pred_ids = [int(i) for i in np.unique(pred_mask) if i > 0]
    gt_ids   = [int(i) for i in np.unique(gt_mask)   if i > 0]

    if not pred_ids and not gt_ids:
        return [], [], []
    if not pred_ids:
        return [], [], gt_ids
    if not gt_ids:
        return [], pred_ids, []

    # Build IoU matrix: rows = predictions, cols = ground truth
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for i, pid in enumerate(pred_ids):
        pred_bin = pred_mask == pid
        for j, gid in enumerate(gt_ids):
            gt_bin = gt_mask == gid
            inter = np.logical_and(pred_bin, gt_bin).sum()
            union = np.logical_or(pred_bin,  gt_bin).sum()
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches, fp_ids, fn_ids = [], list(pred_ids), list(gt_ids)
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((pred_ids[r], gt_ids[c], float(iou_matrix[r, c])))
            fp_ids.remove(pred_ids[r])
            fn_ids.remove(gt_ids[c])

    return matches, fp_ids, fn_ids


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(matches, fp_ids, fn_ids) -> dict:
    """
    Compute detection metrics from matched instances.

    Returns dict with: TP, FP, FN, precision, recall, F1, mean_IoU.
    """
    TP = len(matches)
    FP = len(fp_ids)
    FN = len(fn_ids)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    mean_iou  = float(np.mean([m[2] for m in matches])) if matches else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_IoU": mean_iou,
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def find_gt_path(pred_path: Path, gts_dir: Path):
    """Find the ground-truth mask that matches a prediction by stem."""
    for suffix in [pred_path.suffix, ".tif", ".png"]:
        candidate = gts_dir / (pred_path.stem + suffix)
        if candidate.exists():
            return candidate
    return None


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise IOError(f"Could not read mask: {path}")
    return mask


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted instance masks against gold ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--preds",         required=True, type=Path,
                        help="Directory containing predicted mask files")
    parser.add_argument("--gts",           required=True, type=Path,
                        help="Directory containing ground-truth mask files")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="IoU threshold for counting a detection as correct (default: 0.5)")
    parser.add_argument("--output",        type=Path, default=None,
                        help="Optional CSV file to write per-image results")
    args = parser.parse_args()

    if not args.preds.is_dir():
        print(f"ERROR: --preds is not a directory: {args.preds}")
        sys.exit(1)
    if not args.gts.is_dir():
        print(f"ERROR: --gts is not a directory: {args.gts}")
        sys.exit(1)

    pred_files = sorted(
        list(args.preds.glob("*.tif")) + list(args.preds.glob("*.png"))
    )
    if not pred_files:
        print(f"No .tif or .png files found in {args.preds}")
        sys.exit(1)

    # ── Per-image evaluation ──────────────────────────────────────────────────
    col_w = [40, 5, 5, 5, 10, 10, 8, 10]
    header = ["image", "TP", "FP", "FN", "precision", "recall", "F1", "mean_IoU"]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*header))
    print("  " + "-" * (sum(col_w) + 2 * (len(col_w) - 1)))

    per_image_rows = []
    total_tp = total_fp = total_fn = 0

    for pred_path in pred_files:
        gt_path = find_gt_path(pred_path, args.gts)
        if gt_path is None:
            print(f"  [skip] no GT mask for {pred_path.name}")
            continue

        try:
            pred_mask = load_mask(pred_path)
            gt_mask   = load_mask(gt_path)
        except IOError as e:
            print(f"  [error] {e}")
            continue

        matches, fp_ids, fn_ids = match_instances(pred_mask, gt_mask, args.iou_threshold)
        m = compute_metrics(matches, fp_ids, fn_ids)

        total_tp += m["TP"]
        total_fp += m["FP"]
        total_fn += m["FN"]

        row = {
            "image":     pred_path.name,
            "TP":        m["TP"],
            "FP":        m["FP"],
            "FN":        m["FN"],
            "precision": f"{m['precision']:.3f}",
            "recall":    f"{m['recall']:.3f}",
            "F1":        f"{m['F1']:.3f}",
            "mean_IoU":  f"{m['mean_IoU']:.3f}",
        }
        print(fmt.format(*[str(row[k]) for k in header]))
        per_image_rows.append(row)

    if not per_image_rows:
        print("\nNo image pairs evaluated.")
        sys.exit(0)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(per_image_rows)

    def _mean(key):
        return np.mean([float(r[key]) for r in per_image_rows])

    macro_precision = _mean("precision")
    macro_recall    = _mean("recall")
    macro_f1        = _mean("F1")
    macro_iou       = _mean("mean_IoU")

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1        = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                       if (micro_precision + micro_recall) > 0 else 0.0)

    print()
    print(f"--- Summary ({n} images, IoU threshold = {args.iou_threshold}) ---")
    print(f"Total:   TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Macro:   precision={macro_precision:.3f}  recall={macro_recall:.3f}"
          f"  F1={macro_f1:.3f}  mean_IoU={macro_iou:.3f}")
    print(f"Micro:   precision={micro_precision:.3f}  recall={micro_recall:.3f}"
          f"  F1={micro_f1:.3f}")

    # ── CSV output ────────────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(per_image_rows)
            writer.writerow({
                "image":     f"MACRO (n={n})",
                "TP": "", "FP": "", "FN": "",
                "precision": f"{macro_precision:.3f}",
                "recall":    f"{macro_recall:.3f}",
                "F1":        f"{macro_f1:.3f}",
                "mean_IoU":  f"{macro_iou:.3f}",
            })
            writer.writerow({
                "image":     "MICRO",
                "TP": total_tp, "FP": total_fp, "FN": total_fn,
                "precision": f"{micro_precision:.3f}",
                "recall":    f"{micro_recall:.3f}",
                "F1":        f"{micro_f1:.3f}",
                "mean_IoU":  "",
            })
        print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
