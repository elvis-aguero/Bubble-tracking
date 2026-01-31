"""
Fully automatic SAM3-based bubble segmentation pipeline.

Structural decisions:
- Tiling is used instead of global upsampling so small (~10 px) bubbles occupy more pixels in the model input after SAM3's internal resize; this improves small-object visibility without global memory blowups.
- SAM3 resizes inputs to a fixed encoder size, so the apparent bubble scale depends on the tile size; smaller tiles effectively enlarge bubbles in the resized space and help the point-prompted model resolve rims.

Candidate points:
- One positive point is generated per candidate bubble center (LoG/DoG/Hough) and all points are sent in a single SAM3 call per tile/image (no per-point UI).

Consolidation:
- Masks are filtered by area, border exclusion, and optional circularity/solidity; then deduplicated using IoU/containment rules, dropping smaller or lower-score masks when they substantially overlap.

Known failure modes and tuning:
- Very faint rims or low contrast reduce candidate detection; increase contrast normalization, lower blob thresholds, or enable tiling to boost effective scale.
- Overlapping bubbles may merge into one mask; tighten IoU/containment thresholds, enable convex hull for stability, or increase tile overlap.
- Masks truncated at tile edges can be dropped by min_coverage_for_keep; raise tile overlap or reduce that threshold if too many masks are discarded.

Assumptions:
- Bubbles are approximately circular with darker rims on a light background; optional convex-hull conversion supports downstream area estimation under this morphology.

ENVIRONMENT SETUP (copy-paste for this environment):
    export HF_HOME=/users/eaguerov/scratch/hf
    export CUDA_VISIBLE_DEVICES=0  # if needed

"""

# Dependencies (install as needed):
# pip install torch numpy pillow scipy scikit-image transformers
# Plus SAM3 package (facebook/sam3) available in the environment.

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
except ImportError as exc:
    raise RuntimeError("torch is required. Install with: pip install torch") from exc


DEFAULT_CONFIG: Dict[str, Any] = {
    "device": "cuda",
    "seed": 0,
    "sam": {
        "confidence_threshold": 0.4,
        "use_fp16": True,
    },
    "preprocess": {
        "enable_contrast_norm": True,
        "contrast_method": "clahe",  # clahe | rescale | none
        "clahe_clip_limit": 0.03,
        "clahe_kernel_size": 64,
        "rescale_percentiles": [2, 98],
        "gamma": 1.0,
        "invert_gray": False,
    },
    "candidates": {
        "enable_candidates": True,
        "method": "log+dog",  # log | dog | hough | log+dog | all
        "min_diameter_px": 10,
        "max_diameter_px": 200,
        "log_min_sigma": None,
        "log_max_sigma": None,
        "log_num_sigma": 10,
        "log_threshold": 0.02,
        "dog_min_sigma": None,
        "dog_max_sigma": None,
        "dog_sigma_ratio": 1.6,
        "dog_threshold": 0.02,
        "hough_dp": 1.2,
        "hough_minDist": 8,
        "hough_param1": 100,
        "hough_param2": 30,
        "hough_minRadius": None,
        "hough_maxRadius": None,
        "max_candidates_per_image": 2000,
        "dedup_centers": True,
        "dedup_radius_px": 4.0,
        "fallback_on_empty_candidates": True,
    },
    "tiling": {
        "enable_tiling": True,
        "tile_size": 256,
        "tile_h": None,
        "tile_w": None,
        "tile_overlap": 64,
        "pad_mode": "reflect",  # reflect | constant | edge
        "min_coverage_for_keep": 0.5,
    },
    "hole_fill": {
        "enable_hole_fill": True,
        "hole_fill_method": "binary_fill_holes",  # binary_fill_holes | closing
        "closing_radius": 2,
        "min_hole_area_px": 0,
        "fill_only_if_ringlike": False,
        "ringlike_threshold": 0.15,
    },
    "postprocess": {
        "enable_consolidation": True,
        "iou_dedup_thresh": 0.5,
        "containment_thresh": 0.9,
        "min_area_px": 60,
        "max_area_fraction": 0.2,
        "border_exclusion_px": 1,
        "circularity_min": 0.0,
        "solidity_min": 0.0,
        "enable_convex_hull": False,
    },
    "fallback": {
        "mode": "grid",  # grid | transformers
        "grid_spacing_px": 24,
        "grid_jitter_px": 0,
        "max_points": 4096,
    },
    "output": {
        "output_mode": "cutout",  # cutout | overlay
        "overlay_alpha": 128,
        "overlay_colormap": "tab20",
    },
    "debug": {
        "debug_dir": None,
    },
}


TAB20_COLORS: List[Tuple[int, int, int]] = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
]


@dataclass
class Candidate:
    x: float
    y: float
    r: Optional[float] = None


@dataclass
class Instance:
    mask: np.ndarray  # bool HxW
    score: Optional[float]
    area: int
    bbox: Tuple[int, int, int, int]


class Sam3Backend:
    def __init__(self, device: str, confidence_threshold: float, use_fp16: bool) -> None:
        self.device = device
        self.use_fp16 = use_fp16
        self.model, self.processor = self._load_model(confidence_threshold)

    def _load_model(self, confidence_threshold: float):
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "sam3 package is required. Ensure facebook/sam3 is installed and importable."
            ) from exc

        try:
            model = build_sam3_image_model(device=self.device)
        except TypeError:
            model = build_sam3_image_model()

        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()

        try:
            processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
        except TypeError:
            processor = Sam3Processor(model)

        return model, processor

    def _infer(self, fn, *args, **kwargs):
        with torch.no_grad():
            if self.device == "cuda" and self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return fn(*args, **kwargs)
            return fn(*args, **kwargs)

    def segment_with_points(self, image: Image.Image, points_xy: Sequence[Tuple[float, float]]):
        if not points_xy:
            return [], []

        state = self.processor.set_image(image)
        coords = np.array(points_xy, dtype=np.float32)
        labels = np.ones((len(points_xy),), dtype=np.int32)

        methods = [
            "set_point_prompt",
            "set_points_prompt",
            "set_point_prompt_with_boxes",
            "set_image_and_points",
            "set_image_and_points_prompt",
            "set_prompt",
        ]

        param_sets = [
            {"state": state, "point_coords": coords, "point_labels": labels},
            {"state": state, "points": coords, "labels": labels},
            {"state": state, "point_coords": coords[None, ...], "point_labels": labels[None, ...]},
            {"state": state, "points": coords[None, ...], "labels": labels[None, ...]},
            {"image": image, "point_coords": coords, "point_labels": labels},
            {"image": image, "points": coords, "labels": labels},
            {"image": image, "point_coords": coords[None, ...], "point_labels": labels[None, ...]},
        ]

        last_err: Optional[Exception] = None
        for method_name in methods:
            if not hasattr(self.processor, method_name):
                continue
            fn = getattr(self.processor, method_name)
            for params in param_sets:
                try:
                    out = self._call_with_filtered_kwargs(fn, params)
                    return extract_masks_and_scores(out)
                except TypeError as exc:
                    last_err = exc
                    continue
                except Exception as exc:
                    last_err = exc
                    continue

        raise RuntimeError(
            "Unable to call SAM3 point prompt API. Tried methods: " + ", ".join(methods)
        ) from last_err

    def _call_with_filtered_kwargs(self, fn, params: Dict[str, Any]):
        sig = inspect.signature(fn)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return self._infer(fn, **params)

        filtered = {k: v for k, v in params.items() if k in sig.parameters}
        if not any(k in filtered for k in ("point_coords", "points", "point_labels", "labels")):
            raise TypeError("No compatible point parameters for SAM3 prompt call.")
        return self._infer(fn, **filtered)


class TransformersMaskGenerator:
    def __init__(self, device: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for fallback_mode=transformers. Install with: pip install transformers"
            ) from exc

        device_id = 0 if device == "cuda" else -1
        self.pipe = pipeline("mask-generation", model="facebook/sam3", device=device_id)

    def segment_everything(self, image: Image.Image):
        out = self.pipe(image)
        return extract_masks_and_scores(out)


def segment_with_points(backend: Sam3Backend, image: Image.Image, points_xy: Sequence[Tuple[float, float]]):
    return backend.segment_with_points(image, points_xy)


def segment_everything(
    image: Image.Image,
    backend: Sam3Backend,
    fallback_generator: Optional[TransformersMaskGenerator],
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[List[np.ndarray], List[Optional[float]], Optional[TransformersMaskGenerator]]:
    if cfg["fallback"].get("mode") == "transformers":
        if fallback_generator is None:
            fallback_generator = TransformersMaskGenerator(cfg["device"])
        masks, scores = fallback_generator.segment_everything(image)
        return masks, scores, fallback_generator

    spacing = int(cfg["fallback"].get("grid_spacing_px", 24))
    jitter = int(cfg["fallback"].get("grid_jitter_px", 0))
    max_points = int(cfg["fallback"].get("max_points", 0))
    grid_points = generate_grid_points(image.height, image.width, spacing, jitter, max_points, seed)
    masks, scores = backend.segment_with_points(image, grid_points)
    return masks, scores, fallback_generator


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key] = merge_dicts(base[key], val)
        else:
            base[key] = val
    return base


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = merge_dicts(cfg, user_cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 bubble segmentation pipeline")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output RGBA PNG path")
    parser.add_argument("--config", default=None, help="Optional JSON config path")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--debug_dir", default=None, help="Directory for debug PNGs")

    parser.add_argument("--enable_candidates", action="store_true", help="Enable candidate point detection")
    parser.add_argument("--disable_candidates", action="store_true", help="Disable candidate point detection")
    parser.add_argument("--enable_tiling", action="store_true", help="Enable tiling")
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling")
    parser.add_argument("--enable_hole_fill", action="store_true", help="Enable hole filling")
    parser.add_argument("--disable_hole_fill", action="store_true", help="Disable hole filling")
    parser.add_argument("--enable_consolidation", action="store_true", help="Enable consolidation")
    parser.add_argument("--disable_consolidation", action="store_true", help="Disable consolidation")
    parser.add_argument("--output_mode", choices=["cutout", "overlay"], default=None, help="Output mode")

    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.device:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.debug_dir:
        cfg["debug"]["debug_dir"] = args.debug_dir

    if args.enable_candidates:
        cfg["candidates"]["enable_candidates"] = True
    if args.disable_candidates:
        cfg["candidates"]["enable_candidates"] = False
    if args.enable_tiling:
        cfg["tiling"]["enable_tiling"] = True
    if args.disable_tiling:
        cfg["tiling"]["enable_tiling"] = False
    if args.enable_hole_fill:
        cfg["hole_fill"]["enable_hole_fill"] = True
    if args.disable_hole_fill:
        cfg["hole_fill"]["enable_hole_fill"] = False
    if args.enable_consolidation:
        cfg["postprocess"]["enable_consolidation"] = True
    if args.disable_consolidation:
        cfg["postprocess"]["enable_consolidation"] = False
    if args.output_mode:
        cfg["output"]["output_mode"] = args.output_mode

    return cfg


def ensure_output_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def preprocess_gray(image_rgb: Image.Image, cfg: Dict[str, Any]) -> np.ndarray:
    gray = np.array(image_rgb.convert("L"), dtype=np.float32) / 255.0

    if cfg["preprocess"].get("invert_gray", False):
        gray = 1.0 - gray

    if not cfg["preprocess"].get("enable_contrast_norm", True):
        return gray

    method = cfg["preprocess"].get("contrast_method", "clahe")
    if method == "clahe":
        try:
            from skimage import exposure
        except ImportError as exc:
            raise RuntimeError("scikit-image is required for CLAHE preprocessing.") from exc
        gray = exposure.equalize_adapthist(
            gray,
            clip_limit=cfg["preprocess"].get("clahe_clip_limit", 0.03),
            kernel_size=cfg["preprocess"].get("clahe_kernel_size", 64),
        )
    elif method == "rescale":
        p_low, p_high = cfg["preprocess"].get("rescale_percentiles", [2, 98])
        lo, hi = np.percentile(gray, p_low), np.percentile(gray, p_high)
        if hi > lo:
            gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown contrast_method: {method}")

    gamma = cfg["preprocess"].get("gamma", 1.0)
    if gamma and gamma != 1.0:
        gray = np.clip(gray, 0.0, 1.0) ** gamma

    return gray


def diameter_to_sigma(diameter_px: float) -> float:
    return diameter_px / (2.0 * math.sqrt(2.0))


def _sigma_or_default(value: Optional[float], default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and value <= 0:
        return default
    return float(value)


def detect_candidates(gray: np.ndarray, cfg: Dict[str, Any]) -> List[Candidate]:
    candidates: List[Candidate] = []
    method = cfg["candidates"]["method"]
    min_d = cfg["candidates"]["min_diameter_px"]
    max_d = cfg["candidates"]["max_diameter_px"]

    min_sigma = diameter_to_sigma(min_d)
    max_sigma = diameter_to_sigma(max_d)

    if method in ("log", "log+dog", "all"):
        candidates.extend(detect_log_blobs(gray, cfg, min_sigma, max_sigma))
    if method in ("dog", "log+dog", "all"):
        candidates.extend(detect_dog_blobs(gray, cfg, min_sigma, max_sigma))
    if method in ("hough", "all"):
        candidates.extend(detect_hough_circles(gray, cfg, min_sigma, max_sigma))

    if cfg["candidates"].get("dedup_centers", True):
        candidates = dedup_candidates(candidates, cfg["candidates"].get("dedup_radius_px", 4.0))

    max_cands = int(cfg["candidates"].get("max_candidates_per_image", 0))
    if max_cands > 0 and len(candidates) > max_cands:
        idx = np.linspace(0, len(candidates) - 1, max_cands, dtype=int)
        candidates = [candidates[i] for i in idx]

    return candidates


def detect_log_blobs(gray: np.ndarray, cfg: Dict[str, Any], min_sigma: float, max_sigma: float) -> List[Candidate]:
    try:
        from skimage.feature import blob_log
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for LoG blob detection.") from exc

    min_sigma = _sigma_or_default(cfg["candidates"].get("log_min_sigma"), min_sigma)
    max_sigma = _sigma_or_default(cfg["candidates"].get("log_max_sigma"), max_sigma)
    num_sigma = int(cfg["candidates"].get("log_num_sigma", 10))
    threshold = float(cfg["candidates"].get("log_threshold", 0.02))

    blobs = blob_log(gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    results: List[Candidate] = []
    for y, x, sigma in blobs:
        radius = math.sqrt(2.0) * sigma
        results.append(Candidate(float(x), float(y), float(radius)))
    return results


def detect_dog_blobs(gray: np.ndarray, cfg: Dict[str, Any], min_sigma: float, max_sigma: float) -> List[Candidate]:
    try:
        from skimage.feature import blob_dog
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for DoG blob detection.") from exc

    min_sigma = _sigma_or_default(cfg["candidates"].get("dog_min_sigma"), min_sigma)
    max_sigma = _sigma_or_default(cfg["candidates"].get("dog_max_sigma"), max_sigma)
    sigma_ratio = float(cfg["candidates"].get("dog_sigma_ratio", 1.6))
    threshold = float(cfg["candidates"].get("dog_threshold", 0.02))

    blobs = blob_dog(gray, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=threshold)
    results: List[Candidate] = []
    for y, x, sigma in blobs:
        radius = math.sqrt(2.0) * sigma
        results.append(Candidate(float(x), float(y), float(radius)))
    return results


def detect_hough_circles(gray: np.ndarray, cfg: Dict[str, Any], min_sigma: float, max_sigma: float) -> List[Candidate]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for Hough circle detection.") from exc

    min_r = cfg["candidates"].get("hough_minRadius")
    max_r = cfg["candidates"].get("hough_maxRadius")
    if min_r is None:
        min_r = int(round(math.sqrt(2.0) * min_sigma))
    if max_r is None:
        max_r = int(round(math.sqrt(2.0) * max_sigma))

    img8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    circles = cv2.HoughCircles(
        img8,
        cv2.HOUGH_GRADIENT,
        dp=float(cfg["candidates"].get("hough_dp", 1.2)),
        minDist=float(cfg["candidates"].get("hough_minDist", 8)),
        param1=float(cfg["candidates"].get("hough_param1", 100)),
        param2=float(cfg["candidates"].get("hough_param2", 30)),
        minRadius=int(min_r),
        maxRadius=int(max_r),
    )

    results: List[Candidate] = []
    if circles is not None:
        circles = np.squeeze(circles, axis=0)
        for x, y, r in circles:
            results.append(Candidate(float(x), float(y), float(r)))
    return results


def dedup_candidates(candidates: List[Candidate], radius: float) -> List[Candidate]:
    if radius <= 0:
        return candidates

    kept: List[Candidate] = []
    r2 = radius * radius
    for cand in candidates:
        keep = True
        for other in kept:
            dx = cand.x - other.x
            dy = cand.y - other.y
            if dx * dx + dy * dy <= r2:
                keep = False
                break
        if keep:
            kept.append(cand)
    return kept


def generate_grid_points(h: int, w: int, spacing: int, jitter: int, max_points: int, seed: int) -> List[Tuple[float, float]]:
    if spacing <= 0:
        raise ValueError("grid_spacing_px must be > 0")

    ys = np.arange(spacing / 2.0, h, spacing)
    xs = np.arange(spacing / 2.0, w, spacing)
    points = [(float(x), float(y)) for y in ys for x in xs]

    if jitter > 0:
        rng = np.random.default_rng(seed)
        jitter_xy = rng.uniform(-jitter, jitter, size=(len(points), 2))
        points = [(x + jx, y + jy) for (x, y), (jx, jy) in zip(points, jitter_xy)]
        points = [(min(max(x, 0.0), w - 1.0), min(max(y, 0.0), h - 1.0)) for x, y in points]

    if max_points > 0 and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[i] for i in idx]

    return points


def create_tiles(h: int, w: int, cfg: Dict[str, Any]) -> Tuple[List[Tuple[int, int, int, int]], Tuple[int, int]]:
    tile_h = cfg["tiling"].get("tile_h") or cfg["tiling"].get("tile_size")
    tile_w = cfg["tiling"].get("tile_w") or cfg["tiling"].get("tile_size")
    tile_h = int(tile_h)
    tile_w = int(tile_w)

    overlap = cfg["tiling"].get("tile_overlap", 0)
    if isinstance(overlap, float) and 0 < overlap < 1:
        overlap_h = int(round(tile_h * overlap))
        overlap_w = int(round(tile_w * overlap))
    else:
        overlap_h = int(overlap)
        overlap_w = int(overlap)

    stride_h = max(tile_h - overlap_h, 1)
    stride_w = max(tile_w - overlap_w, 1)

    y_starts = list(range(0, max(h - tile_h, 0) + 1, stride_h))
    x_starts = list(range(0, max(w - tile_w, 0) + 1, stride_w))

    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]

    if y_starts[-1] + tile_h < h:
        y_starts.append(h - tile_h)
    if x_starts[-1] + tile_w < w:
        x_starts.append(w - tile_w)

    tiles: List[Tuple[int, int, int, int]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            tiles.append((x0, y0, x0 + tile_w, y0 + tile_h))

    pad_right = max(0, max(x1 for _, _, x1, _ in tiles) - w)
    pad_bottom = max(0, max(y1 for _, y0, _, y1 in tiles) - h)
    return tiles, (pad_bottom, pad_right)


def pad_image(arr: np.ndarray, pad_bottom: int, pad_right: int, mode: str) -> np.ndarray:
    if pad_bottom <= 0 and pad_right <= 0:
        return arr

    if arr.ndim == 2:
        pad_width = ((0, pad_bottom), (0, pad_right))
    else:
        pad_width = ((0, pad_bottom), (0, pad_right), (0, 0))

    return np.pad(arr, pad_width, mode=mode)


def extract_masks_and_scores(output: Any) -> Tuple[List[np.ndarray], List[Optional[float]]]:
    masks = None
    scores = None

    if isinstance(output, dict):
        masks = output.get("masks") or output.get("mask") or output.get("segmentation")
        scores = output.get("scores")
    elif isinstance(output, list) and output and isinstance(output[0], dict):
        pairs = [
            (item.get("mask") or item.get("segmentation"), item.get("score") or item.get("confidence"))
            for item in output
        ]
        pairs = [(m, s) for m, s in pairs if m is not None]
        masks = [m for m, _ in pairs]
        scores = [s for _, s in pairs]
    else:
        masks = output

    if masks is None:
        return [], []

    mask_list = normalize_masks(masks)
    score_list: List[Optional[float]] = []
    if scores is None:
        score_list = [None] * len(mask_list)
    else:
        if torch.is_tensor(scores):
            scores_np = scores.detach().cpu().numpy().reshape(-1)
            score_list = [float(s) for s in scores_np]
        elif isinstance(scores, (list, tuple)):
            score_list = [float(s) if s is not None else None for s in scores]
        elif isinstance(scores, np.ndarray):
            score_list = [float(s) for s in scores.reshape(-1)]
        else:
            score_list = [None] * len(mask_list)

    if len(score_list) < len(mask_list):
        score_list.extend([None] * (len(mask_list) - len(score_list)))

    return mask_list, score_list


def normalize_masks(masks: Any) -> List[np.ndarray]:
    if torch.is_tensor(masks):
        arr = masks.detach().cpu().numpy()
        if arr.ndim == 4:
            if arr.shape[1] == 1:
                arr = arr[:, 0, ...]
        elif arr.ndim == 3:
            pass
        else:
            arr = np.expand_dims(arr, axis=0)
        return [(arr[i] > 0.5) for i in range(arr.shape[0])]

    if isinstance(masks, np.ndarray):
        arr = masks
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, ...]
        if arr.ndim == 3:
            return [(arr[i] > 0.5) for i in range(arr.shape[0])]
        if arr.ndim == 2:
            return [arr > 0.5]

    if isinstance(masks, (list, tuple)):
        result = []
        for mask in masks:
            if mask is None:
                continue
            if torch.is_tensor(mask):
                m = mask.detach().cpu().numpy()
            else:
                m = np.array(mask)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            if m.ndim == 0:
                continue
            result.append(m > 0.5)
        return result

    return []


def resize_mask_to_shape(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.array(mask_img) > 127


def fill_holes(mask: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    if not cfg["hole_fill"].get("enable_hole_fill", True):
        return mask

    method = cfg["hole_fill"].get("hole_fill_method", "binary_fill_holes")
    min_hole_area = int(cfg["hole_fill"].get("min_hole_area_px", 0))
    fill_only_ringlike = cfg["hole_fill"].get("fill_only_if_ringlike", False)
    ringlike_thresh = float(cfg["hole_fill"].get("ringlike_threshold", 0.15))

    if method == "binary_fill_holes":
        try:
            from scipy.ndimage import binary_fill_holes
            from skimage.morphology import remove_small_holes
        except ImportError as exc:
            raise RuntimeError("scipy and scikit-image are required for hole filling.") from exc
        if min_hole_area > 0:
            filled = remove_small_holes(mask, area_threshold=min_hole_area)
        else:
            filled = binary_fill_holes(mask)
        hole_area = int(filled.sum() - mask.sum())
        if fill_only_ringlike and mask.sum() > 0:
            if hole_area / float(mask.sum()) < ringlike_thresh:
                return mask
        return filled

    if method == "closing":
        try:
            from skimage.morphology import binary_closing, disk, remove_small_holes
        except ImportError as exc:
            raise RuntimeError("scikit-image is required for morphological closing.") from exc

        radius = int(cfg["hole_fill"].get("closing_radius", 2))
        closed = binary_closing(mask, disk(radius)) if radius > 0 else mask
        hole_area = int(closed.sum() - mask.sum())
        if fill_only_ringlike and mask.sum() > 0:
            if hole_area / float(mask.sum()) < ringlike_thresh:
                return mask
        if min_hole_area > 0:
            closed = remove_small_holes(closed, area_threshold=min_hole_area)
        return closed

    raise ValueError(f"Unknown hole_fill_method: {method}")


def maybe_convex_hull(mask: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    if not cfg["postprocess"].get("enable_convex_hull", False):
        return mask
    try:
        from skimage.morphology import convex_hull_image
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for convex hull computation.") from exc
    return convex_hull_image(mask)


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)


def touches_border(mask: np.ndarray, border_px: int) -> bool:
    if border_px <= 0:
        return False
    h, w = mask.shape
    b = min(border_px, h, w)
    return (
        mask[:b, :].any()
        or mask[-b:, :].any()
        or mask[:, :b].any()
        or mask[:, -b:].any()
    )


def mask_perimeter(mask: np.ndarray) -> float:
    try:
        from skimage.measure import perimeter
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for perimeter calculation.") from exc
    return float(perimeter(mask, neighborhood=8))


def mask_solidity(mask: np.ndarray) -> float:
    try:
        from skimage.morphology import convex_hull_image
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for solidity calculation.") from exc
    hull = convex_hull_image(mask)
    area = float(mask.sum())
    hull_area = float(hull.sum())
    if hull_area == 0:
        return 0.0
    return area / hull_area


def mask_iou(mask_a: np.ndarray, bbox_a: Tuple[int, int, int, int], mask_b: np.ndarray, bbox_b: Tuple[int, int, int, int]) -> float:
    x0 = max(bbox_a[0], bbox_b[0])
    y0 = max(bbox_a[1], bbox_b[1])
    x1 = min(bbox_a[2], bbox_b[2])
    y1 = min(bbox_a[3], bbox_b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = np.logical_and(mask_a[y0:y1, x0:x1], mask_b[y0:y1, x0:x1]).sum()
    if inter == 0:
        return 0.0
    union = mask_a.sum() + mask_b.sum() - inter
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_containment(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    smaller = min(mask_a.sum(), mask_b.sum())
    if smaller == 0:
        return 0.0
    return float(inter) / float(smaller)


def consolidate_instances(instances: List[Instance], cfg: Dict[str, Any], image_shape: Tuple[int, int]) -> List[Instance]:
    h, w = image_shape
    image_area = h * w

    filtered: List[Instance] = []
    min_area = int(cfg["postprocess"].get("min_area_px", 0))
    max_area_fraction = float(cfg["postprocess"].get("max_area_fraction", 1.0))
    border_px = int(cfg["postprocess"].get("border_exclusion_px", 0))
    circularity_min = float(cfg["postprocess"].get("circularity_min", 0.0))
    solidity_min = float(cfg["postprocess"].get("solidity_min", 0.0))

    for inst in instances:
        if inst.area < min_area:
            continue
        if max_area_fraction > 0 and inst.area > int(image_area * max_area_fraction):
            continue
        if touches_border(inst.mask, border_px):
            continue
        if circularity_min > 0:
            perim = mask_perimeter(inst.mask)
            if perim <= 0:
                continue
            circ = 4.0 * math.pi * inst.area / (perim * perim)
            if circ < circularity_min:
                continue
        if solidity_min > 0:
            sol = mask_solidity(inst.mask)
            if sol < solidity_min:
                continue
        filtered.append(inst)

    if not cfg["postprocess"].get("enable_consolidation", True) or len(filtered) <= 1:
        return filtered

    iou_thresh = float(cfg["postprocess"].get("iou_dedup_thresh", 0.5))
    containment_thresh = float(cfg["postprocess"].get("containment_thresh", 0.9))

    def score_key(inst: Instance) -> float:
        if inst.score is not None:
            return float(inst.score)
        return float(inst.area)

    sorted_instances = sorted(filtered, key=score_key, reverse=True)
    kept: List[Instance] = []

    for inst in sorted_instances:
        drop = False
        for kept_inst in kept:
            iou = mask_iou(inst.mask, inst.bbox, kept_inst.mask, kept_inst.bbox)
            if iou >= iou_thresh:
                drop = True
                break
            containment = mask_containment(inst.mask, kept_inst.mask)
            if containment >= containment_thresh:
                # Drop smaller or lower score
                if score_key(inst) <= score_key(kept_inst):
                    drop = True
                    break
        if not drop:
            kept.append(inst)

    return kept


def build_rgba_cutout(image_rgb: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    h, w, _ = image_rgb.shape
    alpha = np.zeros((h, w), dtype=np.uint8)
    if masks:
        union = np.zeros((h, w), dtype=bool)
        for m in masks:
            union |= m
        alpha[union] = 255
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = alpha
    return rgba


def build_rgba_overlay(image_rgb: np.ndarray, masks: List[np.ndarray], alpha: int, colormap: str) -> np.ndarray:
    h, w, _ = image_rgb.shape
    out = image_rgb.astype(np.float32).copy()
    colors = TAB20_COLORS if colormap == "tab20" else TAB20_COLORS
    a = float(np.clip(alpha, 0, 255)) / 255.0
    for i, mask in enumerate(masks):
        color = np.array(colors[i % len(colors)], dtype=np.float32)
        out[mask] = (1.0 - a) * out[mask] + a * color
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(out, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = 255
    return rgba


def save_candidate_viz(image_rgb: np.ndarray, points: List[Tuple[float, float]], out_path: str) -> None:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x, y in points:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0))
    img.save(out_path)


def save_tile_grid_viz(image_rgb: np.ndarray, tiles: List[Tuple[int, int, int, int]], out_path: str) -> None:
    img = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in tiles:
        draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0))
    img.save(out_path)


def save_hole_fill_viz(before: np.ndarray, after: np.ndarray, out_path: str) -> None:
    left = (before.astype(np.uint8) * 255)
    right = (after.astype(np.uint8) * 255)
    combo = np.concatenate([left, right], axis=1)
    Image.fromarray(combo, mode="L").save(out_path)


def run_pipeline(image: Image.Image, cfg: Dict[str, Any]) -> Tuple[List[Instance], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "candidate_points": [],
        "tiles": [],
        "hole_fill_example": None,
    }

    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = "cpu"
        cfg["device"] = "cpu"

    sam_backend = Sam3Backend(device, cfg["sam"].get("confidence_threshold", 0.4), cfg["sam"].get("use_fp16", True))

    fallback_generator: Optional[TransformersMaskGenerator] = None
    if not cfg["candidates"].get("enable_candidates", True) and cfg["fallback"].get("mode") == "transformers":
        fallback_generator = TransformersMaskGenerator(device)

    image_rgb = np.array(image.convert("RGB"))
    gray = preprocess_gray(image, cfg)

    h, w = gray.shape

    tiles: List[Tuple[int, int, int, int]] = [(0, 0, w, h)]
    pad_bottom = 0
    pad_right = 0
    if cfg["tiling"].get("enable_tiling", True):
        tiles, (pad_bottom, pad_right) = create_tiles(h, w, cfg)

    debug["tiles"] = tiles

    pad_mode = cfg["tiling"].get("pad_mode", "reflect")
    padded_rgb = pad_image(image_rgb, pad_bottom, pad_right, pad_mode)
    padded_gray = pad_image(gray, pad_bottom, pad_right, pad_mode)

    instances: List[Instance] = []

    for (x0, y0, x1, y1) in tiles:
        tile_rgb = padded_rgb[y0:y1, x0:x1]
        tile_gray = padded_gray[y0:y1, x0:x1]
        tile_h, tile_w = tile_gray.shape

        tile_points: List[Tuple[float, float]] = []
        if cfg["candidates"].get("enable_candidates", True):
            cands = detect_candidates(tile_gray, cfg)
            tile_points = [(c.x, c.y) for c in cands]
            for c in cands:
                debug["candidate_points"].append((c.x + x0, c.y + y0))
        else:
            cands = []

        use_fallback_grid = False
        if cfg["candidates"].get("enable_candidates", True):
            if not tile_points and cfg["candidates"].get("fallback_on_empty_candidates", True):
                use_fallback_grid = True
        else:
            use_fallback_grid = True

        tile_pil = Image.fromarray(tile_rgb)

        masks: List[np.ndarray] = []
        scores: List[Optional[float]] = []

        if use_fallback_grid:
            masks, scores, fallback_generator = segment_everything(
                tile_pil, sam_backend, fallback_generator, cfg, cfg["seed"]
            )
        else:
            masks, scores = segment_with_points(sam_backend, tile_pil, tile_points)

        if not masks:
            continue

        for mask, score in zip(masks, scores):
            mask = resize_mask_to_shape(mask, (tile_h, tile_w))
            before = mask.copy()
            mask = fill_holes(mask, cfg)
            mask = maybe_convex_hull(mask, cfg)
            if debug["hole_fill_example"] is None:
                debug["hole_fill_example"] = (before, mask)

            # Map to full image coordinates
            full_mask = np.zeros((h, w), dtype=bool)
            y1_clip = min(y1, h)
            x1_clip = min(x1, w)
            tile_clip_h = y1_clip - y0
            tile_clip_w = x1_clip - x0
            if tile_clip_h <= 0 or tile_clip_w <= 0:
                continue
            full_mask[y0:y1_clip, x0:x1_clip] = mask[:tile_clip_h, :tile_clip_w]

            if cfg["tiling"].get("enable_tiling", True):
                coverage = full_mask.sum() / max(1.0, mask.sum())
                if coverage < float(cfg["tiling"].get("min_coverage_for_keep", 0.0)):
                    continue

            area = int(full_mask.sum())
            if area == 0:
                continue

            bbox = mask_bbox(full_mask)
            instances.append(Instance(mask=full_mask, score=score, area=area, bbox=bbox))

    consolidated = consolidate_instances(instances, cfg, (h, w))
    return consolidated, debug


def save_debug_outputs(debug: Dict[str, Any], image_rgb: np.ndarray, masks: List[np.ndarray], cfg: Dict[str, Any]) -> None:
    debug_dir = cfg["debug"].get("debug_dir")
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)

    if debug.get("candidate_points"):
        save_candidate_viz(image_rgb, debug["candidate_points"], os.path.join(debug_dir, "candidates.png"))
    if debug.get("tiles"):
        save_tile_grid_viz(image_rgb, debug["tiles"], os.path.join(debug_dir, "tiles.png"))
    if debug.get("hole_fill_example"):
        before, after = debug["hole_fill_example"]
        save_hole_fill_viz(before, after, os.path.join(debug_dir, "hole_fill_before_after.png"))
    if masks:
        overlay = build_rgba_overlay(
            image_rgb,
            masks,
            alpha=int(cfg["output"].get("overlay_alpha", 128)),
            colormap=cfg["output"].get("overlay_colormap", "tab20"),
        )
        Image.fromarray(overlay, mode="RGBA").save(os.path.join(debug_dir, "consolidated_overlay.png"))


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    set_deterministic_seed(int(cfg["seed"]))

    image = Image.open(args.input).convert("RGB")
    instances, debug = run_pipeline(image, cfg)

    image_rgb = np.array(image)
    masks = [inst.mask for inst in instances]

    output_mode = cfg["output"].get("output_mode", "cutout")
    if output_mode == "cutout":
        rgba = build_rgba_cutout(image_rgb, masks)
    elif output_mode == "overlay":
        rgba = build_rgba_overlay(
            image_rgb,
            masks,
            alpha=int(cfg["output"].get("overlay_alpha", 128)),
            colormap=cfg["output"].get("overlay_colormap", "tab20"),
        )
    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    ensure_output_dir(args.output)
    Image.fromarray(rgba, mode="RGBA").save(args.output)

    save_debug_outputs(debug, image_rgb, masks, cfg)


if __name__ == "__main__":
    main()
