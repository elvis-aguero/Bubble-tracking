from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List

from .provenance import ExperimentSpec


@dataclass(frozen=True)
class HybridSearchSpace:
    frst: Dict[str, List[float]]
    blackhat: Dict[str, List[float]]
    fusion: Dict[str, List[float]]
    families: List[str]
    fusion_rules: List[str]


def _cross_product(grid: Dict[str, List[float]]) -> List[Dict[str, float]]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def build_initial_experiment_specs(search_space: HybridSearchSpace) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []

    frst_combos = _cross_product(search_space.frst)
    blackhat_combos = _cross_product(search_space.blackhat)
    fusion_combos = _cross_product(search_space.fusion)

    for family in search_space.families:
        if family == "frst_only":
            for idx, frst_params in enumerate(frst_combos, start=1):
                specs.append(
                    ExperimentSpec(
                        experiment_id=f"frst_only_{idx:03d}",
                        family=family,
                        variant="baseline",
                        parameters=dict(frst_params),
                    )
                )
        elif family == "blackhat_only":
            for idx, blackhat_params in enumerate(blackhat_combos, start=1):
                specs.append(
                    ExperimentSpec(
                        experiment_id=f"blackhat_only_{idx:03d}",
                        family=family,
                        variant="baseline",
                        parameters=dict(blackhat_params),
                    )
                )
        elif family == "hybrid_current":
            idx = 0
            for frst_params in frst_combos:
                for blackhat_params in blackhat_combos:
                    for fusion_params in fusion_combos:
                        for fusion_rule in search_space.fusion_rules:
                            idx += 1
                            parameters = {}
                            parameters.update(frst_params)
                            parameters.update({f"blackhat.{k}": v for k, v in blackhat_params.items()})
                            parameters.update(fusion_params)
                            parameters["fusion_rule"] = fusion_rule
                            specs.append(
                                ExperimentSpec(
                                    experiment_id=f"hybrid_current_{idx:03d}",
                                    family=family,
                                    variant=fusion_rule,
                                    parameters=parameters,
                                )
                            )
        else:
            raise ValueError(f"Unknown family: {family}")

    return specs
