from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from bubbly_flows.tests.src.common.bubble_sam3.config import DEFAULT_CONFIG, merge_dicts

from .gold_eval import rasterize_labelme_shapes
from .provenance import ExperimentSpec


def _default_config() -> Dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_CONFIG))


SAM3_INTERPRETER = "/users/eaguerov/.conda/envs/sam3/bin/python"


def select_experiment_interpreter(spec: ExperimentSpec) -> str:
    if spec.family in {"frst_only", "hybrid_current"}:
        return os.environ.get("BUBBLY_HYBRID_PYTHON", SAM3_INTERPRETER)
    return sys.executable


def write_variant_config(spec: ExperimentSpec, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _default_config()

    blackhat_updates: Dict[str, Any] = {}
    postprocess_updates: Dict[str, Any] = {}
    hole_fill_updates: Dict[str, Any] = {}
    for key, value in spec.parameters.items():
        if key.startswith("blackhat."):
            blackhat_updates[key.split(".", 1)[1]] = value
        elif key in {"iou_dedup_thresh", "containment_thresh", "min_area_px", "enable_consolidation"}:
            postprocess_updates[key] = value
        elif key == "enable_hole_fill":
            hole_fill_updates["enable_hole_fill"] = value

    if blackhat_updates:
        merge_dicts(cfg["blackhat"], blackhat_updates)
    if postprocess_updates:
        merge_dicts(cfg["postprocess"], postprocess_updates)
    if hole_fill_updates:
        merge_dicts(cfg["hole_fill"], hole_fill_updates)

    path = run_dir / "variant_config.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_prediction_command(
    spec: ExperimentSpec,
    image_path: Path,
    run_dir: Path,
    config_path: Path,
) -> List[str]:
    output_dir = run_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_overlay.png"
    output_json = output_dir / f"{image_path.stem}.json"

    command = [
        select_experiment_interpreter(spec),
        "-m",
        "bubbly_flows.tests.src.hybrid.bubble_frst_sam3_mask",
        "--input",
        str(image_path),
        "--output",
        str(output_path),
        "--config",
        str(config_path),
        "--output_json",
        str(output_json),
        "--no_analysis_json",
    ]

    if spec.family == "frst_only":
        command.extend(["--disable_blackhat", "--disable_pcs"])
    elif spec.family == "blackhat_only":
        command.extend(["--disable_candidates", "--disable_pcs", "--enable_blackhat"])
    elif spec.family == "hybrid_current":
        command.extend(["--big_backend", "hf"])
    else:
        raise ValueError(f"Unsupported family: {spec.family}")

    for key in ("r_min", "r_max", "alpha", "mag_percentile", "peak_percentile"):
        if key in spec.parameters:
            command.extend([f"--{key}", str(spec.parameters[key])])

    passthrough_keys = (
        "r_step",
        "nms_size",
        "border",
        "max_peaks",
        "knn_k",
        "hex_radius_factor",
        "tile_size_factor",
        "tile_overlap_factor",
        "area_limit_factor",
        "adaptive_area_min",
        "adaptive_area_max",
        "adaptive_circularity_min",
        "adaptive_solidity_min",
        "adaptive_intensity_max",
    )
    for key in passthrough_keys:
        if key in spec.parameters:
            command.extend([f"--{key}", str(spec.parameters[key])])

    if spec.parameters.get("blackhat_split_fused"):
        command.append("--blackhat_split_fused")

    return command


def labelme_prediction_to_mask(labelme_json_path: Path, destination: Path) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to materialize prediction masks") from exc

    payload = json.loads(labelme_json_path.read_text(encoding="utf-8"))
    width = int(payload["imageWidth"])
    height = int(payload["imageHeight"])
    mask_rows = rasterize_labelme_shapes(width, height, payload.get("shapes", []))
    image = Image.new("I;16", (width, height), color=0)
    for y, row in enumerate(mask_rows):
        for x, value in enumerate(row):
            image.putpixel((x, y), int(value))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def execute_family_prediction(spec: ExperimentSpec, dataset_items: List[Dict[str, Any]], run_dir: Path) -> List[Dict[str, Any]]:
    config_path = write_variant_config(spec, run_dir)
    predictions_dir = run_dir / "predictions"
    previews: List[Dict[str, Any]] = []

    for item in dataset_items:
        image_path = Path(item["source_image"])
        command = build_prediction_command(spec, image_path=image_path, run_dir=run_dir, config_path=config_path)
        subprocess.run(command, check=True)

        labelme_path = run_dir / "images" / f"{image_path.stem}.json"
        pred_mask_path = predictions_dir / f"{image_path.stem}.tif"
        labelme_prediction_to_mask(labelme_path, pred_mask_path)

        previews.append(
            {
                "stem": item["stem"],
                "overlay": str((run_dir / "images" / f"{image_path.stem}_overlay.png").relative_to(run_dir)),
                "labelme_json": str(labelme_path.relative_to(run_dir)),
                "prediction_mask": str(pred_mask_path.relative_to(run_dir)),
            }
        )

    return previews


def parse_evaluation_results(results_csv_path: Path) -> Dict[str, Any]:
    per_image: List[Dict[str, Any]] = []
    aggregate: Dict[str, float] = {}

    with open(results_csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image = row["image"]
            if image.startswith("MACRO"):
                aggregate = {
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "F1": float(row["F1"]),
                    "mean_IoU": float(row["mean_IoU"]),
                }
                continue
            if image == "MICRO":
                continue
            per_image.append(row)

    return {"per_image": per_image, "aggregate": aggregate}


def evaluate_predictions(dataset_manifest: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    dataset_root = Path(dataset_manifest["dataset_root"])
    preds_dir = run_dir / "predictions"
    results_path = run_dir / "results.csv"
    repo_root = Path(__file__).resolve().parents[4]
    evaluate_script = repo_root / "bubbly_flows" / "scripts" / "evaluate.py"

    command = [
        sys.executable,
        str(evaluate_script),
        "--preds",
        str(preds_dir),
        "--gts",
        str(dataset_root / "labels"),
        "--output",
        str(results_path),
    ]
    subprocess.run(command, check=True)
    return parse_evaluation_results(results_path)
