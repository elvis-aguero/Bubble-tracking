#!/usr/bin/env python3
"""
facebookresearch SAM3 point-prompt backend for FRST tiled inference.

This uses sam3's Sam3Processor (if available) to issue point prompts per object.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

class FrstPointBackendFB:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "sam3 package is required for facebookresearch point backend."
            ) from exc

        confidence = float(cfg["sam"].get("confidence_threshold", 0.4))
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model, confidence_threshold=confidence)
        self._point_method = self._find_point_method()

    def _find_point_method(self):
        for name in (
            "set_point_prompt",
            "set_points_prompt",
            "set_point_prompts",
            "set_points",
            "set_prompt_points",
        ):
            if hasattr(self.processor, name):
                method = getattr(self.processor, name)
                if self._method_supports_labels(method):
                    return method
        raise RuntimeError(
            "Sam3Processor does not expose a point-prompt method that accepts labels."
        )

    @staticmethod
    def _method_supports_labels(method) -> bool:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            # If we can't inspect, allow and validate at call time.
            return True
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return True
        for name in ("point_labels", "labels", "input_labels"):
            if name in sig.parameters:
                return True
        return False

    def _call_point_method(self, method, state, points_xy, labels):
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return method(state=state, points=points_xy, labels=labels)

        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        params: Dict[str, Any] = {}
        if "state" in sig.parameters:
            params["state"] = state

        points_arr = np.asarray(points_xy, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64)

        point_key = None
        if "point_coords" in sig.parameters:
            point_key = "point_coords"
        elif "points" in sig.parameters:
            point_key = "points"
        elif "input_points" in sig.parameters:
            point_key = "input_points"
        elif has_kwargs:
            point_key = "points"

        if point_key is None:
            raise RuntimeError("Point-prompt method does not accept point coordinates.")
        params[point_key] = points_arr

        label_key = None
        if "point_labels" in sig.parameters:
            label_key = "point_labels"
        elif "labels" in sig.parameters:
            label_key = "labels"
        elif "input_labels" in sig.parameters:
            label_key = "input_labels"
        elif has_kwargs:
            label_key = "labels"

        if label_key is None:
            raise RuntimeError("Point-prompt method does not accept labels for negative points.")
        params[label_key] = labels_arr

        return method(**params)

    def segment_tile(
        self,
        image: Image.Image,
        objects_points_xy: Sequence[Sequence[Tuple[float, float]]],
        objects_labels: Sequence[Sequence[int]],
    ) -> Tuple[List[np.ndarray], List[Optional[float]]]:
        if not objects_points_xy:
            return [], []

        state = self.processor.set_image(image)
        all_masks: List[np.ndarray] = []
        all_scores: List[Optional[float]] = []

        for pts, labs in zip(objects_points_xy, objects_labels):
            if not pts:
                continue
            out = self._call_point_method(self._point_method, state, pts, labs)
            if isinstance(out, dict):
                masks = out.get("masks") or []
                scores = out.get("scores") or [None] * len(masks)
            else:
                masks = out
                scores = [None] * len(masks)

            for mask, score in zip(masks, scores):
                if mask is None:
                    continue
                if hasattr(mask, "cpu"):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)
                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                    mask_np = mask_np[0]
                mask_bool = mask_np.astype(bool)
                all_masks.append(mask_bool)
                if score is None:
                    all_scores.append(None)
                elif hasattr(score, "cpu"):
                    all_scores.append(float(score.cpu().item()))
                else:
                    all_scores.append(float(score))

        return all_masks, all_scores
