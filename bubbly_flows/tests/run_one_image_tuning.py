#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from bubbly_flows.tests.src.experiments.gold_eval import prepare_gold_evaluation_set
from bubbly_flows.tests.src.experiments.one_image_tuning import (
    OneImageSearchSpace,
    build_adaptive_start_specs,
    build_one_image_experiment_specs,
    propose_coordinate_neighbors,
    select_single_image_manifest,
)
from bubbly_flows.tests.src.experiments.runner import execute_real_adaptive_experiment_search, execute_real_experiment_batch


DEFAULT_SEARCH = OneImageSearchSpace(
    families=["hybrid_current"],
    frst={
        "r_min": [4, 5],
        "r_max": [22, 25, 28],
        "r_step": [2],
        "alpha": [1.2, 1.4, 1.6],
        "mag_percentile": [84.0, 88.0, 92.0],
        "peak_percentile": [97.0, 99.0],
        "nms_size": [7, 9],
        "border": [6, 8],
        "max_peaks": [1500, 2000, 2500],
    },
    geometry={
        "knn_k": [2, 3],
        "hex_radius_factor": [1.2, 1.5],
        "tile_size_factor": [6.0, 8.0, 10.0],
        "tile_overlap_factor": [2.0, 2.5],
        "area_limit_factor": [12.0, 16.0],
    },
    adaptive={
        "adaptive_area_min": [16, 20],
        "adaptive_area_max": [220, 350],
        "adaptive_circularity_min": [0.3, 0.4],
        "adaptive_solidity_min": [0.7, 0.75],
        "adaptive_intensity_max": [140.0, 160.0],
        "blackhat_split_fused": [False, True],
    },
    fusion={
        "iou_dedup_thresh": [0.4, 0.5],
        "containment_thresh": [0.85, 0.9],
        "min_area_px": [40, 60],
        "enable_consolidation": [True, False],
        "enable_hole_fill": [True, False],
    },
    baseline_family="hybrid_current",
    max_rounds=2,
    max_proposals_per_round=8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-image hybrid tuning against a gold image.")
    parser.add_argument(
        "--image-stem",
        default="ZeroG_FlightDay_Test_C1S0014_img006001",
        help="Gold image stem to overfit intentionally.",
    )
    parser.add_argument(
        "--output-root",
        default="bubbly_flows/tests/output/experiments/one_image_tuning",
        help="Output directory under bubbly_flows/tests/output/experiments/",
    )
    parser.add_argument(
        "--search-json",
        default=None,
        help="Optional JSON file overriding the default search space.",
    )
    parser.add_argument(
        "--strategy",
        choices=["adaptive", "grid"],
        default="adaptive",
        help="Search strategy. Default is adaptive around baseline_hybrid_original.",
    )
    return parser.parse_args()


def _load_search_space(path: str | None) -> OneImageSearchSpace:
    if not path:
        return DEFAULT_SEARCH
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return OneImageSearchSpace(**payload)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    gold_json_dir = root / "bubbly_flows" / "annotations" / "gold" / "gold_seed_v04" / "labels_json"
    output_eval_root = root / "bubbly_flows" / "tests" / "output" / "eval_sets" / "gold_seed_v04"
    image_search_dirs = [
        root / "bubbly_flows" / "data" / "frames" / "images_16bit_png",
        root / "bubbly_flows" / "data" / "frames" / "images_clahe",
        root / "bubbly_flows" / "data" / "frames" / "images_raw",
        root / "bubbly_flows" / "annotations" / "pool",
    ]

    manifest = prepare_gold_evaluation_set(gold_json_dir, output_eval_root, image_search_dirs)
    manifest = select_single_image_manifest(manifest, args.image_stem)

    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    search_space = _load_search_space(args.search_json)
    if args.strategy == "grid":
        specs = build_one_image_experiment_specs(search_space)
        summary = execute_real_experiment_batch(specs, manifest, output_root)
    else:
        initial_specs = build_adaptive_start_specs(search_space)
        summary = execute_real_adaptive_experiment_search(
            initial_specs=initial_specs,
            dataset_manifest=manifest,
            output_root=output_root,
            neighbor_fn=lambda best_spec, round_index: propose_coordinate_neighbors(
                best_spec,
                search_space,
                round_index=round_index,
                max_proposals=search_space.max_proposals_per_round,
            ),
            max_rounds=search_space.max_rounds,
        )
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
