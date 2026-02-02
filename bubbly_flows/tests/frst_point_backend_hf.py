#!/usr/bin/env python3
"""
HF transformers SAM3 point-prompt backend for FRST tiled inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from bubble_sam3.backend import Sam3PointBackend, segment_with_object_points


class FrstPointBackendHF:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.backend = Sam3PointBackend(cfg["device"], cfg["sam"])

    def segment_tile(
        self,
        image: Image.Image,
        objects_points_xy: Sequence[Sequence[Tuple[float, float]]],
        objects_labels: Sequence[Sequence[int]],
    ) -> Tuple[List, List[Optional[float]]]:
        return segment_with_object_points(self.backend, image, objects_points_xy, objects_labels)
