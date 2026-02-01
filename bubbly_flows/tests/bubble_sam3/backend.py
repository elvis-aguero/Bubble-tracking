from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as exc:
    raise RuntimeError("torch is required. Install with: pip install torch") from exc

from .candidates import generate_grid_points


class Sam3PointBackend:
    _MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
    _PROCESSOR_CACHE: Dict[Tuple[str, str], Any] = {}

    def __init__(self, device: str, sam_cfg: Dict[str, Any]) -> None:
        self.device = device
        self.use_fp16 = bool(sam_cfg.get("use_fp16", True))
        self.model_name = str(sam_cfg.get("model_name", "facebook/sam3"))
        self.backend = str(sam_cfg.get("backend", "tracker"))
        self.points_per_batch = int(sam_cfg.get("points_per_batch", 128))
        self.multimask_output = bool(sam_cfg.get("multimask_output", False))
        self.confidence_threshold = float(sam_cfg.get("confidence_threshold", 0.0))
        self.local_files_only = bool(sam_cfg.get("local_files_only", True))
        self.model, self.processor = self._load_tracker()

    def _load_tracker(self):
        if self.backend != "tracker":
            raise ValueError(f"Unsupported sam.backend={self.backend}. Expected 'tracker'.")

        try:
            from transformers import Sam3TrackerProcessor, Sam3TrackerModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers with Sam3TrackerProcessor/Sam3TrackerModel is required for point prompting."
            ) from exc

        cache_key = (self.model_name, self.device)
        if cache_key in self._MODEL_CACHE and cache_key in self._PROCESSOR_CACHE:
            return self._MODEL_CACHE[cache_key], self._PROCESSOR_CACHE[cache_key]

        logger = logging.getLogger(__name__)
        try:
            processor = Sam3TrackerProcessor.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            )
            model = Sam3TrackerModel.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            )
        except OSError as exc:
            if not self.local_files_only:
                raise
            logger.warning(
                "SAM3 tracker not found in local cache; retrying with downloads enabled."
            )
            processor = Sam3TrackerProcessor.from_pretrained(
                self.model_name, local_files_only=False
            )
            model = Sam3TrackerModel.from_pretrained(
                self.model_name, local_files_only=False
            )

        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()

        self._MODEL_CACHE[cache_key] = model
        self._PROCESSOR_CACHE[cache_key] = processor
        return model, processor

    def _infer(self, fn, *args, **kwargs):
        with torch.inference_mode():
            if self.device == "cuda" and self.use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return fn(*args, **kwargs)
            return fn(*args, **kwargs)

    def _call_with_filtered_kwargs(self, fn, params: Dict[str, Any]):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return self._infer(fn, **params)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return self._infer(fn, **params)

        filtered = {k: v for k, v in params.items() if k in sig.parameters}
        return self._infer(fn, **filtered)

    def _prepare_inputs(
        self, image: Image.Image, points_xy: Sequence[Tuple[float, float]]
    ) -> Dict[str, Any]:
        coords = np.array(points_xy, dtype=np.float32)[None, :, None, :]
        labels = np.ones((1, len(points_xy), 1), dtype=np.int64)

        try:
            inputs = self.processor(
                images=image, input_points=coords, input_labels=labels, return_tensors="pt"
            )
        except Exception:
            inputs = self.processor(
                images=image,
                input_points=coords.tolist(),
                input_labels=labels.tolist(),
                return_tensors="pt",
            )

        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(self.device)
        return inputs

    def _segment_batch(
        self, image: Image.Image, points_xy: Sequence[Tuple[float, float]]
    ) -> Tuple[List[np.ndarray], List[Optional[float]]]:
        inputs = self._prepare_inputs(image, points_xy)
        params = dict(inputs)
        params["multimask_output"] = self.multimask_output
        outputs = self._call_with_filtered_kwargs(self.model, params)
        return extract_masks_and_scores(outputs)

    def segment_with_points(
        self,
        image: Image.Image,
        points_xy: Sequence[Tuple[float, float]],
        points_per_batch: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[Optional[float]]]:
        if not points_xy:
            return [], []

        batch_size = int(points_per_batch or self.points_per_batch or len(points_xy))
        batch_size = max(batch_size, 1)

        all_masks: List[np.ndarray] = []
        all_scores: List[Optional[float]] = []
        for start in range(0, len(points_xy), batch_size):
            batch_points = points_xy[start : start + batch_size]
            masks, scores = self._segment_batch(image, batch_points)
            masks, scores = filter_masks_by_score(masks, scores, self.confidence_threshold)
            all_masks.extend(masks)
            all_scores.extend(scores)

        return all_masks, all_scores


class TransformersMaskGenerator:
    def __init__(self, device: str, model_name: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for fallback_mode=transformers. Install with: pip install transformers"
            ) from exc

        device_id = 0 if device == "cuda" else -1
        self.pipe = pipeline("mask-generation", model=model_name, device=device_id)

    def segment_everything(self, image: Image.Image):
        out = self.pipe(image)
        return extract_masks_and_scores(out)


def segment_with_points(
    backend: Sam3PointBackend, image: Image.Image, points_xy: Sequence[Tuple[float, float]]
) -> Tuple[List[np.ndarray], List[Optional[float]]]:
    if not points_xy:
        return [], []

    logger = logging.getLogger(__name__)
    batch_size = backend.points_per_batch
    while True:
        try:
            return backend.segment_with_points(image, points_xy, points_per_batch=batch_size)
        except RuntimeError as exc:
            if (
                "out of memory" in str(exc).lower()
                and backend.device == "cuda"
                and batch_size > 1
            ):
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "CUDA OOM during SAM3 inference; retrying with points_per_batch=%d",
                    batch_size,
                )
                continue
            raise


def segment_everything(
    image: Image.Image,
    backend: Sam3PointBackend,
    fallback_generator: Optional[TransformersMaskGenerator],
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[List[np.ndarray], List[Optional[float]], Optional[TransformersMaskGenerator]]:
    if cfg["fallback"].get("mode") == "transformers":
        if fallback_generator is None:
            fallback_generator = TransformersMaskGenerator(
                cfg["device"], cfg["sam"].get("model_name", "facebook/sam3")
            )
        masks, scores = fallback_generator.segment_everything(image)
        masks, scores = filter_masks_by_score(
            masks, scores, float(cfg["sam"].get("confidence_threshold", 0.0))
        )
        return masks, scores, fallback_generator

    spacing = int(cfg["fallback"].get("grid_spacing_px", 24))
    jitter = int(cfg["fallback"].get("grid_jitter_px", 0))
    max_points = int(cfg["fallback"].get("max_points", 0))
    grid_points = generate_grid_points(image.height, image.width, spacing, jitter, max_points, seed)
    masks, scores = segment_with_points(backend, image, grid_points)
    return masks, scores, fallback_generator


def extract_masks_and_scores(output: Any) -> Tuple[List[np.ndarray], List[Optional[float]]]:
    masks = None
    scores = None

    def _first_not_none(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    if hasattr(output, "pred_masks"):
        masks = getattr(output, "pred_masks")
        scores = _first_not_none(getattr(output, "iou_scores", None), getattr(output, "scores", None))
    elif isinstance(output, dict):
        masks = output.get("masks") or output.get("mask") or output.get("segmentation")
        scores = _first_not_none(output.get("scores"), output.get("iou_scores"))
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
    score_list = normalize_scores(scores, len(mask_list))

    if len(score_list) < len(mask_list):
        score_list.extend([None] * (len(mask_list) - len(score_list)))

    return mask_list, score_list


def normalize_masks(masks: Any) -> List[np.ndarray]:
    if torch.is_tensor(masks):
        arr = masks.detach().cpu().numpy()
        if arr.ndim == 5:
            if arr.shape[2] > 1:
                arr = arr[:, :, 0, ...]
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        elif arr.ndim == 4:
            if arr.shape[1] > 1 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] > 1:
                arr = arr[:, 0, ...]
            elif arr.shape[1] == 1:
                arr = arr[:, 0, ...]
        elif arr.ndim == 3:
            pass
        else:
            arr = np.expand_dims(arr, axis=0)
        return [(arr[i] > 0.5) for i in range(arr.shape[0])]

    if isinstance(masks, np.ndarray):
        arr = masks
        if arr.ndim == 5:
            if arr.shape[2] > 1:
                arr = arr[:, :, 0, ...]
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, ...]
        elif arr.ndim == 4 and arr.shape[1] > 1:
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


def normalize_scores(scores: Any, expected: int) -> List[Optional[float]]:
    if scores is None:
        return [None] * expected
    if torch.is_tensor(scores):
        arr = scores.detach().cpu().numpy()
    elif isinstance(scores, np.ndarray):
        arr = scores
    elif isinstance(scores, (list, tuple)):
        flat: List[Optional[float]] = []
        for item in scores:
            if isinstance(item, (list, tuple, np.ndarray)):
                if isinstance(item, np.ndarray) and item.ndim > 1:
                    flat.append(float(item.reshape(-1)[0]))
                elif len(item) > 0:
                    flat.append(float(item[0]))
                else:
                    flat.append(None)
            else:
                flat.append(float(item) if item is not None else None)
        return flat
    else:
        return [None] * expected

    if arr.ndim >= 2 and arr.shape[1] > 1:
        arr = arr[:, 0]
    arr = arr.reshape(-1)
    return [float(s) for s in arr]


def filter_masks_by_score(
    masks: List[np.ndarray], scores: List[Optional[float]], threshold: float
) -> Tuple[List[np.ndarray], List[Optional[float]]]:
    if threshold <= 0 or not scores:
        return masks, scores
    filtered_masks: List[np.ndarray] = []
    filtered_scores: List[Optional[float]] = []
    for mask, score in zip(masks, scores):
        if score is None or score >= threshold:
            filtered_masks.append(mask)
            filtered_scores.append(score)
    return filtered_masks, filtered_scores
