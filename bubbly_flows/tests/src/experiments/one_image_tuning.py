from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List

from .provenance import ExperimentSpec


HYBRID_BASELINE_PARAMETERS: Dict[str, Any] = {
    "r_min": 4,
    "r_max": 25,
    "r_step": 2,
    "alpha": 1.4,
    "mag_percentile": 88.0,
    "peak_percentile": 99.0,
    "nms_size": 7,
    "border": 8,
    "max_peaks": 2000,
    "knn_k": 3,
    "hex_radius_factor": 1.5,
    "tile_size_factor": 8.0,
    "tile_overlap_factor": 2.5,
    "area_limit_factor": 16.0,
    "adaptive_area_min": 20,
    "adaptive_area_max": 350,
    "adaptive_circularity_min": 0.4,
    "adaptive_solidity_min": 0.75,
    "adaptive_intensity_max": 160.0,
    "blackhat_split_fused": False,
    "iou_dedup_thresh": 0.5,
    "containment_thresh": 0.9,
    "min_area_px": 60,
    "enable_consolidation": True,
    "enable_hole_fill": True,
}

FRST_BASELINE_PARAMETERS: Dict[str, Any] = {
    key: value
    for key, value in HYBRID_BASELINE_PARAMETERS.items()
    if key not in {
        "adaptive_area_min",
        "adaptive_area_max",
        "adaptive_circularity_min",
        "adaptive_solidity_min",
        "adaptive_intensity_max",
        "blackhat_split_fused",
        "iou_dedup_thresh",
        "containment_thresh",
        "min_area_px",
        "enable_consolidation",
        "enable_hole_fill",
    }
}


@dataclass(frozen=True)
class OneImageSearchSpace:
    families: List[str]
    frst: Dict[str, List[Any]]
    geometry: Dict[str, List[Any]]
    adaptive: Dict[str, List[Any]]
    fusion: Dict[str, List[Any]]
    baseline_family: str = "hybrid_current"
    max_rounds: int = 2
    max_proposals_per_round: int = 8


def _cross_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _ordered_domain(search_space: OneImageSearchSpace) -> Dict[str, List[Any]]:
    merged: Dict[str, List[Any]] = {}
    for group in (search_space.frst, search_space.geometry, search_space.adaptive, search_space.fusion):
        for key, values in group.items():
            merged[key] = list(values)
    return merged


def _coerce_sortable(values: Iterable[Any]) -> List[Any]:
    ordered = list(values)
    if ordered and all(isinstance(value, (bool, int, float)) for value in ordered):
        if all(isinstance(value, bool) for value in ordered):
            return sorted(ordered)
        return sorted(ordered, key=float)
    return ordered


def _neighbor_values(current_value: Any, domain: List[Any]) -> List[Any]:
    values = _coerce_sortable(domain)
    if current_value not in values:
        return values[:2]
    idx = values.index(current_value)
    neighbors: List[Any] = []
    if idx > 0:
        neighbors.append(values[idx - 1])
    if idx + 1 < len(values):
        neighbors.append(values[idx + 1])
    return neighbors


def select_single_image_manifest(dataset_manifest: Dict[str, Any], image_stem: str) -> Dict[str, Any]:
    selected = [item for item in dataset_manifest["items"] if item["stem"] == image_stem]
    if not selected:
        raise ValueError(f"Image stem not found in dataset manifest: {image_stem}")
    manifest = dict(dataset_manifest)
    manifest["items"] = selected
    manifest["count"] = len(selected)
    return manifest


def baseline_hybrid_original_spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="baseline_hybrid_original",
        family="hybrid_current",
        variant="baseline_hybrid_original",
        parameters=dict(HYBRID_BASELINE_PARAMETERS),
    )


def baseline_frst_original_spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="baseline_frst_original",
        family="frst_only",
        variant="baseline_frst_original",
        parameters=dict(FRST_BASELINE_PARAMETERS),
    )


def propose_coordinate_neighbors(
    current_spec: ExperimentSpec,
    search_space: OneImageSearchSpace,
    round_index: int,
    max_proposals: int | None = None,
) -> List[ExperimentSpec]:
    domain = _ordered_domain(search_space)
    proposals: List[ExperimentSpec] = []

    for key, values in domain.items():
        if key not in current_spec.parameters:
            continue
        for neighbor in _neighbor_values(current_spec.parameters[key], values):
            params = dict(current_spec.parameters)
            params[key] = neighbor
            proposals.append(
                ExperimentSpec(
                    experiment_id=f"{current_spec.family}_r{round_index}_{key}_{str(neighbor).replace('.', 'p')}",
                    family=current_spec.family,
                    variant=f"adaptive_round_{round_index}",
                    parameters=params,
                )
            )
            if max_proposals and len(proposals) >= max_proposals:
                return proposals

    return proposals


def build_adaptive_start_specs(search_space: OneImageSearchSpace) -> List[ExperimentSpec]:
    if search_space.baseline_family == "hybrid_current":
        specs: List[ExperimentSpec] = [baseline_hybrid_original_spec()]
        if "frst_only" in search_space.families:
            specs.append(baseline_frst_original_spec())
        return specs
    if search_space.baseline_family == "frst_only":
        return [baseline_frst_original_spec()]
    raise ValueError(f"Unsupported baseline family: {search_space.baseline_family}")


def build_one_image_experiment_specs(search_space: OneImageSearchSpace) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    frst_combos = _cross_product(search_space.frst)
    geometry_combos = _cross_product(search_space.geometry)
    adaptive_combos = _cross_product(search_space.adaptive)
    fusion_combos = _cross_product(search_space.fusion)

    for family in search_space.families:
        idx = 0
        for frst_params in frst_combos:
            for geometry_params in geometry_combos:
                for adaptive_params in adaptive_combos:
                    for fusion_params in fusion_combos:
                        idx += 1
                        parameters: Dict[str, Any] = {}
                        parameters.update(frst_params)
                        parameters.update(geometry_params)
                        if family == "hybrid_current":
                            parameters.update(adaptive_params)
                            parameters.update(fusion_params)
                        specs.append(
                            ExperimentSpec(
                                experiment_id=f"{family}_{idx:03d}",
                                family=family,
                                variant="one_image_tuning",
                                parameters=parameters,
                            )
                        )
        if family not in {"frst_only", "hybrid_current"}:
            raise ValueError(f"Unsupported family for one-image tuning: {family}")

    return specs
