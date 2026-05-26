from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from .provenance import ExperimentSpec, write_run_manifest
from .variant_executor import evaluate_predictions, execute_family_prediction


Predictor = Callable[[ExperimentSpec, List[Dict[str, Any]], Path], List[Dict[str, Any]]]
Evaluator = Callable[[ExperimentSpec, List[Dict[str, Any]], List[Dict[str, Any]], Path], Dict[str, Any]]
NeighborFn = Callable[[ExperimentSpec, int], List[ExperimentSpec]]


def _write_aggregate_metrics(run_dir: Path, aggregate: Dict[str, Any]) -> Path:
    path = run_dir / "aggregate_metrics.json"
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_per_image_metrics(run_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    path = run_dir / "per_image_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else ["image"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_gallery(run_dir: Path, previews: List[Dict[str, Any]]) -> Path:
    path = run_dir / "gallery.md"
    lines = ["# Preview Gallery", ""]
    for preview in previews:
        lines.append(f"- `{preview['stem']}`: `{preview['overlay']}`")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_ranking(output_root: Path, ranking: List[Dict[str, Any]]) -> Path:
    path = output_root / "ranking.csv"
    fieldnames = ["experiment_id", "family", "variant", "F1"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranking:
            writer.writerow(row)
    return path


def _execute_single_experiment(
    spec: ExperimentSpec,
    dataset_name: str,
    dataset_items: List[Dict[str, Any]],
    output_root: Path,
    predictor: Predictor,
    evaluator: Evaluator,
) -> Dict[str, Any]:
    run_dir = output_root / spec.experiment_id
    previews = predictor(spec, dataset_items, run_dir)
    evaluation = evaluator(spec, dataset_items, previews, run_dir)
    aggregate = evaluation["aggregate"]
    per_image = evaluation["per_image"]

    write_run_manifest(run_dir, spec, dataset_name, dataset_items, extra={"previews": previews})
    _write_aggregate_metrics(run_dir, aggregate)
    _write_per_image_metrics(run_dir, per_image)
    _write_gallery(run_dir, previews)

    return {
        "experiment_id": spec.experiment_id,
        "family": spec.family,
        "variant": spec.variant,
        "F1": f"{float(aggregate['F1']):.3f}",
    }


def execute_experiment_batch(
    specs: List[ExperimentSpec],
    dataset_manifest: Dict[str, Any],
    output_root: Path,
    predictor: Predictor,
    evaluator: Evaluator,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_manifest["dataset_name"]
    dataset_items = dataset_manifest["items"]
    ranking: List[Dict[str, Any]] = []

    for spec in specs:
        ranking.append(
            _execute_single_experiment(
                spec=spec,
                dataset_name=dataset_name,
                dataset_items=dataset_items,
                output_root=output_root,
                predictor=predictor,
                evaluator=evaluator,
            )
        )

    ranking.sort(key=lambda row: float(row["F1"]), reverse=True)
    _write_ranking(output_root, ranking)
    return {"ranking": ranking}


def execute_real_experiment_batch(
    specs: List[ExperimentSpec],
    dataset_manifest: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    return execute_experiment_batch(
        specs=specs,
        dataset_manifest=dataset_manifest,
        output_root=output_root,
        predictor=execute_family_prediction,
        evaluator=lambda spec, dataset_items, previews, run_dir: evaluate_predictions(dataset_manifest, run_dir),
    )


def execute_adaptive_experiment_search(
    initial_specs: List[ExperimentSpec],
    dataset_manifest: Dict[str, Any],
    output_root: Path,
    predictor: Predictor,
    evaluator: Evaluator,
    neighbor_fn: NeighborFn,
    max_rounds: int = 2,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_manifest["dataset_name"]
    dataset_items = dataset_manifest["items"]
    ranking: List[Dict[str, Any]] = []
    seen_ids = set()

    def execute_specs(specs: List[ExperimentSpec]) -> None:
        for spec in specs:
            if spec.experiment_id in seen_ids:
                continue
            seen_ids.add(spec.experiment_id)
            ranking.append(
                _execute_single_experiment(
                    spec=spec,
                    dataset_name=dataset_name,
                    dataset_items=dataset_items,
                    output_root=output_root,
                    predictor=predictor,
                    evaluator=evaluator,
                )
            )

    execute_specs(initial_specs)
    ranking.sort(key=lambda row: float(row["F1"]), reverse=True)
    if not ranking:
        _write_ranking(output_root, ranking)
        return {"ranking": ranking}

    for round_index in range(1, max_rounds + 1):
        best_id = ranking[0]["experiment_id"]
        best_spec = next(spec for spec in initial_specs if spec.experiment_id == best_id) if round_index == 1 else current_best_spec
        proposals = neighbor_fn(best_spec, round_index)
        if not proposals:
            break
        execute_specs(proposals)
        ranking.sort(key=lambda row: float(row["F1"]), reverse=True)
        current_best_spec = next(
            (spec for spec in proposals if spec.experiment_id == ranking[0]["experiment_id"]),
            best_spec,
        )

    _write_ranking(output_root, ranking)
    return {"ranking": ranking}


def execute_real_adaptive_experiment_search(
    initial_specs: List[ExperimentSpec],
    dataset_manifest: Dict[str, Any],
    output_root: Path,
    neighbor_fn: NeighborFn,
    max_rounds: int = 2,
) -> Dict[str, Any]:
    return execute_adaptive_experiment_search(
        initial_specs=initial_specs,
        dataset_manifest=dataset_manifest,
        output_root=output_root,
        predictor=execute_family_prediction,
        evaluator=lambda spec, dataset_items, previews, run_dir: evaluate_predictions(dataset_manifest, run_dir),
        neighbor_fn=neighbor_fn,
        max_rounds=max_rounds,
    )
