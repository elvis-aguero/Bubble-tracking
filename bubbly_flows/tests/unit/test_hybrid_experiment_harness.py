import json
import subprocess
import sys
import tempfile
import unittest
from csv import DictReader
from pathlib import Path
from unittest import mock

from bubbly_flows.tests.src.experiments.gold_eval import prepare_gold_evaluation_set
from bubbly_flows.tests.src.experiments.provenance import ExperimentSpec, write_run_manifest
from bubbly_flows.tests.src.experiments.runner import (
    execute_adaptive_experiment_search,
    execute_experiment_batch,
    execute_real_experiment_batch,
)
from bubbly_flows.tests.src.experiments.one_image_tuning import (
    OneImageSearchSpace,
    baseline_hybrid_original_spec,
    build_one_image_experiment_specs,
    propose_coordinate_neighbors,
    select_single_image_manifest,
)
from bubbly_flows.tests.src.experiments.search_space import (
    HybridSearchSpace,
    build_initial_experiment_specs,
)
from bubbly_flows.tests.src.experiments.variant_executor import (
    build_prediction_command,
    parse_evaluation_results,
    write_variant_config,
)


class GoldEvaluationSetTests(unittest.TestCase):
    def test_prepare_gold_evaluation_set_rasterizes_labelme_json_and_caches_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_dir = root / "annotations" / "gold" / "gold_seed_v04" / "labels_json"
            image_dir = root / "data" / "frames" / "images_16bit_png"
            output_dir = root / "tests" / "output" / "eval_sets" / "gold_seed_v04"
            gold_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)

            image_path = image_dir / "example.png"
            image_path.write_bytes(b"fake png bytes")

            labelme = {
                "imagePath": image_path.name,
                "imageHeight": 12,
                "imageWidth": 12,
                "shapes": [
                    {"label": "bubble", "points": [[2, 2], [2, 5], [5, 5], [5, 2]]},
                    {"label": "bubble", "points": [[7, 7], [7, 9], [9, 9], [9, 7]]},
                ],
            }
            (gold_dir / "example.json").write_text(json.dumps(labelme), encoding="utf-8")

            def write_mask(mask_rows, destination):
                payload = {
                    "height": len(mask_rows),
                    "width": len(mask_rows[0]),
                    "ids": sorted({value for row in mask_rows for value in row}),
                }
                destination.write_text(json.dumps(payload), encoding="utf-8")

            result = prepare_gold_evaluation_set(
                gold_json_dir=gold_dir,
                output_dir=output_dir,
                image_search_dirs=[image_dir],
                mask_writer=write_mask,
            )

            self.assertEqual(result["dataset_name"], "gold_seed_v04")
            self.assertEqual(result["count"], 1)
            self.assertTrue((output_dir / "images" / "example.png").exists())
            mask_path = output_dir / "labels" / "example.tif"
            self.assertTrue(mask_path.exists())
            mask = json.loads(mask_path.read_text(encoding="utf-8"))
            self.assertEqual(mask["ids"], [0, 1, 2])
            self.assertEqual(mask["height"], 12)
            self.assertEqual(mask["width"], 12)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_name"], "gold_seed_v04")
            self.assertEqual(manifest["count"], 1)
            self.assertEqual(manifest["items"][0]["image"], "images/example.png")
            self.assertEqual(manifest["items"][0]["label"], "labels/example.tif")


class ProvenanceManifestTests(unittest.TestCase):
    def test_write_run_manifest_records_experiment_spec_and_dataset_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_001"
            run_dir.mkdir(parents=True)

            spec = ExperimentSpec(
                experiment_id="run_001",
                family="hybrid_current",
                variant="conservative_dedup",
                parameters={
                    "r_min": 4,
                    "r_max": 25,
                    "blackhat.percentile": 99.0,
                    "fusion_rule": "conservative",
                },
            )
            dataset_items = [
                {
                    "stem": "example",
                    "image": "images/example.png",
                    "label": "labels/example.tif",
                }
            ]

            manifest_path = write_run_manifest(
                run_dir=run_dir,
                spec=spec,
                dataset_name="gold_seed_v04",
                dataset_items=dataset_items,
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_id"], "run_001")
            self.assertEqual(payload["family"], "hybrid_current")
            self.assertEqual(payload["variant"], "conservative_dedup")
            self.assertEqual(payload["dataset_name"], "gold_seed_v04")
            self.assertEqual(payload["parameters"]["fusion_rule"], "conservative")
            self.assertEqual(payload["dataset_items"][0]["stem"], "example")


class ExperimentBatchRunnerTests(unittest.TestCase):
    def test_execute_experiment_batch_writes_run_artifacts_and_ranking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "tests" / "output" / "experiments"
            dataset_manifest = {
                "dataset_name": "gold_seed_v04",
                "items": [
                    {
                        "stem": "img_a",
                        "image": "images/img_a.png",
                        "label": "labels/img_a.tif",
                    },
                    {
                        "stem": "img_b",
                        "image": "images/img_b.png",
                        "label": "labels/img_b.tif",
                    },
                ],
            }
            specs = [
                ExperimentSpec("run_low", "frst_only", "baseline", {"peak_percentile": 99.0}),
                ExperimentSpec("run_high", "hybrid_current", "conservative", {"peak_percentile": 98.0}),
            ]

            def predictor(spec, dataset_items, run_dir):
                previews = []
                for item in dataset_items:
                    preview_path = run_dir / "images" / f"{item['stem']}_overlay.txt"
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    preview_path.write_text(f"{spec.experiment_id}:{item['stem']}", encoding="utf-8")
                    previews.append({"stem": item["stem"], "overlay": str(preview_path.relative_to(run_dir))})
                return previews

            def evaluator(spec, dataset_items, previews, run_dir):
                base_f1 = 0.45 if spec.experiment_id == "run_low" else 0.72
                per_image = [
                    {"image": item["stem"], "F1": f"{base_f1:.3f}", "precision": "0.800", "recall": "0.700"}
                    for item in dataset_items
                ]
                aggregate = {"F1": base_f1, "precision": 0.8, "recall": 0.7}
                return {"per_image": per_image, "aggregate": aggregate}

            summary = execute_experiment_batch(
                specs=specs,
                dataset_manifest=dataset_manifest,
                output_root=output_root,
                predictor=predictor,
                evaluator=evaluator,
            )

            self.assertEqual(summary["ranking"][0]["experiment_id"], "run_high")
            self.assertTrue((output_root / "run_high" / "manifest.json").exists())
            self.assertTrue((output_root / "run_high" / "aggregate_metrics.json").exists())
            self.assertTrue((output_root / "run_high" / "per_image_metrics.csv").exists())
            self.assertTrue((output_root / "run_high" / "gallery.md").exists())

            with open(output_root / "ranking.csv", newline="", encoding="utf-8") as handle:
                ranking_rows = list(DictReader(handle))
            self.assertEqual([row["experiment_id"] for row in ranking_rows], ["run_high", "run_low"])

    def test_execute_real_experiment_batch_uses_real_executor_and_evaluator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "tests" / "output" / "experiments"
            dataset_manifest = {"dataset_name": "gold_seed_v04", "dataset_root": tmpdir, "items": []}
            spec = ExperimentSpec("run_real", "hybrid_current", "current", {})

            with mock.patch(
                "bubbly_flows.tests.src.experiments.runner.execute_family_prediction",
                return_value=[{"stem": "img_a", "overlay": "images/img_a_overlay.png"}],
            ) as predictor, mock.patch(
                "bubbly_flows.tests.src.experiments.runner.evaluate_predictions",
                return_value={"per_image": [{"image": "img_a", "F1": "0.900"}], "aggregate": {"F1": 0.9}},
            ) as evaluator:
                summary = execute_real_experiment_batch([spec], dataset_manifest, output_root)

            predictor.assert_called_once()
            evaluator.assert_called_once()
            self.assertEqual(summary["ranking"][0]["experiment_id"], "run_real")

    def test_execute_adaptive_experiment_search_recenters_on_best_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "tests" / "output" / "experiments"
            dataset_manifest = {"dataset_name": "gold_seed_v04", "items": [{"stem": "img_a"}]}
            baseline = ExperimentSpec("baseline_hybrid_original", "hybrid_current", "baseline", {"alpha": 1.4})

            scores = {
                "baseline_hybrid_original": 0.20,
                "hybrid_current_r1_alpha_1p6": 0.45,
                "hybrid_current_r1_peak_percentile_97p0": 0.10,
                "hybrid_current_r2_alpha_1p8": 0.70,
            }
            neighbor_calls = []

            def predictor(spec, dataset_items, run_dir):
                preview_path = run_dir / "images" / "img_a_overlay.txt"
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                preview_path.write_text(spec.experiment_id, encoding="utf-8")
                return [{"stem": "img_a", "overlay": str(preview_path.relative_to(run_dir))}]

            def evaluator(spec, dataset_items, previews, run_dir):
                value = scores[spec.experiment_id]
                return {
                    "per_image": [{"image": "img_a", "F1": f"{value:.3f}"}],
                    "aggregate": {"F1": value, "precision": value, "recall": value},
                }

            def neighbor_fn(best_spec, round_index):
                neighbor_calls.append((best_spec.experiment_id, round_index))
                if round_index == 1:
                    return [
                        ExperimentSpec("hybrid_current_r1_alpha_1p6", "hybrid_current", "adaptive_round_1", {"alpha": 1.6}),
                        ExperimentSpec("hybrid_current_r1_peak_percentile_97p0", "hybrid_current", "adaptive_round_1", {"peak_percentile": 97.0}),
                    ]
                if round_index == 2:
                    self.assertEqual(best_spec.experiment_id, "hybrid_current_r1_alpha_1p6")
                    return [
                        ExperimentSpec("hybrid_current_r2_alpha_1p8", "hybrid_current", "adaptive_round_2", {"alpha": 1.8}),
                    ]
                return []

            summary = execute_adaptive_experiment_search(
                initial_specs=[baseline],
                dataset_manifest=dataset_manifest,
                output_root=output_root,
                predictor=predictor,
                evaluator=evaluator,
                neighbor_fn=neighbor_fn,
                max_rounds=2,
            )

            self.assertEqual(summary["ranking"][0]["experiment_id"], "hybrid_current_r2_alpha_1p8")
            self.assertEqual(neighbor_calls, [("baseline_hybrid_original", 1), ("hybrid_current_r1_alpha_1p6", 2)])


class SearchSpaceTests(unittest.TestCase):
    def test_build_initial_experiment_specs_covers_three_families_and_fusion_variants(self):
        search_space = HybridSearchSpace(
            frst={"r_min": [4], "r_max": [25], "alpha": [1.4], "mag_percentile": [88.0], "peak_percentile": [99.0]},
            blackhat={"radius": [5], "percentile": [99.0], "area_min": [30], "area_max": [120]},
            fusion={"iou_dedup_thresh": [0.5], "containment_thresh": [0.9], "min_area_px": [60]},
            families=["frst_only", "blackhat_only", "hybrid_current"],
            fusion_rules=["current", "conservative", "branch_priority"],
        )

        specs = build_initial_experiment_specs(search_space)

        families = {spec.family for spec in specs}
        self.assertEqual(families, {"frst_only", "blackhat_only", "hybrid_current"})
        hybrid_rules = {
            spec.parameters["fusion_rule"] for spec in specs if spec.family == "hybrid_current"
        }
        self.assertEqual(hybrid_rules, {"current", "conservative", "branch_priority"})

    def test_select_single_image_manifest_filters_to_requested_stem(self):
        manifest = {
            "dataset_name": "gold_seed_v04",
            "count": 2,
            "items": [
                {"stem": "img006001", "image": "images/img006001.png", "label": "labels/img006001.tif"},
                {"stem": "img009542", "image": "images/img009542.png", "label": "labels/img009542.tif"},
            ],
        }

        selected = select_single_image_manifest(manifest, "img006001")

        self.assertEqual(selected["count"], 1)
        self.assertEqual(len(selected["items"]), 1)
        self.assertEqual(selected["items"][0]["stem"], "img006001")

    def test_build_one_image_experiment_specs_excludes_blackhat_only_and_carries_hidden_knobs(self):
        search_space = OneImageSearchSpace(
            families=["frst_only", "hybrid_current"],
            frst={"r_min": [4], "r_max": [25], "alpha": [1.4], "peak_percentile": [99.0]},
            geometry={"knn_k": [3], "hex_radius_factor": [1.5], "tile_size_factor": [8.0]},
            adaptive={"adaptive_area_min": [20], "adaptive_area_max": [350]},
            fusion={"iou_dedup_thresh": [0.5], "containment_thresh": [0.9], "min_area_px": [60]},
        )

        specs = build_one_image_experiment_specs(search_space)

        families = {spec.family for spec in specs}
        self.assertEqual(families, {"frst_only", "hybrid_current"})
        self.assertNotIn("blackhat_only", families)

        hybrid = next(spec for spec in specs if spec.family == "hybrid_current")
        self.assertEqual(hybrid.parameters["knn_k"], 3)
        self.assertEqual(hybrid.parameters["hex_radius_factor"], 1.5)
        self.assertEqual(hybrid.parameters["adaptive_area_min"], 20)

    def test_baseline_hybrid_original_spec_matches_current_hybrid_defaults(self):
        spec = baseline_hybrid_original_spec()

        self.assertEqual(spec.experiment_id, "baseline_hybrid_original")
        self.assertEqual(spec.family, "hybrid_current")
        self.assertEqual(spec.variant, "baseline_hybrid_original")
        self.assertEqual(spec.parameters["r_min"], 4)
        self.assertEqual(spec.parameters["r_max"], 25)
        self.assertEqual(spec.parameters["alpha"], 1.4)
        self.assertEqual(spec.parameters["knn_k"], 3)
        self.assertEqual(spec.parameters["hex_radius_factor"], 1.5)
        self.assertEqual(spec.parameters["adaptive_area_min"], 20)
        self.assertEqual(spec.parameters["iou_dedup_thresh"], 0.5)
        self.assertEqual(spec.parameters["enable_consolidation"], True)
        self.assertEqual(spec.parameters["enable_hole_fill"], True)

    def test_propose_coordinate_neighbors_changes_one_knob_at_a_time_around_baseline(self):
        search_space = OneImageSearchSpace(
            families=["hybrid_current"],
            frst={"alpha": [1.2, 1.4, 1.6], "peak_percentile": [97.0, 99.0]},
            geometry={"knn_k": [2, 3]},
            adaptive={"adaptive_area_min": [16, 20, 24]},
            fusion={"iou_dedup_thresh": [0.4, 0.5, 0.6]},
        )

        baseline = baseline_hybrid_original_spec()
        proposals = propose_coordinate_neighbors(baseline, search_space, round_index=1, max_proposals=16)

        self.assertTrue(proposals)
        ids = {spec.experiment_id for spec in proposals}
        self.assertEqual(len(ids), len(proposals))
        for spec in proposals:
            changed = {
                key for key, value in spec.parameters.items()
                if baseline.parameters.get(key) != value
            }
            self.assertEqual(len(changed), 1, msg=f"{spec.experiment_id} changed {changed}")
            self.assertNotEqual(spec.parameters, baseline.parameters)

    def test_run_one_image_tuning_help_works_when_invoked_by_path(self):
        repo_root = Path(__file__).resolve().parents[3]
        script_path = repo_root / "bubbly_flows" / "tests" / "run_one_image_tuning.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Run one-image hybrid tuning against a gold image.", result.stdout)


class VariantExecutorTests(unittest.TestCase):
    def test_write_variant_config_maps_blackhat_and_postprocess_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_cfg"
            spec = ExperimentSpec(
                experiment_id="run_cfg",
                family="hybrid_current",
                variant="current",
                parameters={
                    "blackhat.radius": 7,
                    "blackhat.percentile": 99.5,
                    "iou_dedup_thresh": 0.4,
                    "containment_thresh": 0.85,
                    "min_area_px": 42,
                },
            )

            config_path = write_variant_config(spec, run_dir)
            payload = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["blackhat"]["radius"], 7)
            self.assertEqual(payload["blackhat"]["percentile"], 99.5)
            self.assertEqual(payload["postprocess"]["iou_dedup_thresh"], 0.4)
            self.assertEqual(payload["postprocess"]["containment_thresh"], 0.85)
            self.assertEqual(payload["postprocess"]["min_area_px"], 42)

    def test_build_prediction_command_sets_family_specific_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_cmd"
            run_dir.mkdir(parents=True)
            image_path = Path(tmpdir) / "image.png"
            image_path.write_bytes(b"image")

            spec = ExperimentSpec(
                experiment_id="run_cmd",
                family="blackhat_only",
                variant="baseline",
                parameters={"blackhat.radius": 5},
            )
            config_path = run_dir / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            command = build_prediction_command(
                spec=spec,
                image_path=image_path,
                run_dir=run_dir,
                config_path=config_path,
            )

            rendered = " ".join(command)
            self.assertIn("-m bubbly_flows.tests.src.hybrid.bubble_frst_sam3_mask", rendered)
            self.assertIn("--disable_candidates", rendered)
            self.assertIn("--disable_pcs", rendered)
            self.assertIn("--enable_blackhat", rendered)
            self.assertIn("--output_json", rendered)

    def test_build_prediction_command_uses_sam3_interpreter_for_sam3_backed_families(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_cmd"
            run_dir.mkdir(parents=True)
            image_path = Path(tmpdir) / "image.png"
            image_path.write_bytes(b"image")
            config_path = run_dir / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            blackhat_spec = ExperimentSpec(
                experiment_id="run_blackhat",
                family="blackhat_only",
                variant="baseline",
                parameters={},
            )
            hybrid_spec = ExperimentSpec(
                experiment_id="run_hybrid",
                family="hybrid_current",
                variant="baseline",
                parameters={},
            )

            blackhat_command = build_prediction_command(
                spec=blackhat_spec,
                image_path=image_path,
                run_dir=run_dir,
                config_path=config_path,
            )
            hybrid_command = build_prediction_command(
                spec=hybrid_spec,
                image_path=image_path,
                run_dir=run_dir,
                config_path=config_path,
            )

            self.assertEqual(blackhat_command[0], sys.executable)
            self.assertEqual(hybrid_command[0], "/users/eaguerov/.conda/envs/sam3/bin/python")

    def test_build_prediction_command_for_hybrid_current_forces_hf_big_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_cmd"
            run_dir.mkdir(parents=True)
            image_path = Path(tmpdir) / "image.png"
            image_path.write_bytes(b"image")
            config_path = run_dir / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            hybrid_spec = ExperimentSpec(
                experiment_id="run_hybrid",
                family="hybrid_current",
                variant="baseline",
                parameters={},
            )

            command = build_prediction_command(
                spec=hybrid_spec,
                image_path=image_path,
                run_dir=run_dir,
                config_path=config_path,
            )

            rendered = " ".join(command)
            self.assertIn("--big_backend hf", rendered)

    def test_build_prediction_command_passes_hidden_geometry_and_adaptive_knobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_cmd"
            run_dir.mkdir(parents=True)
            image_path = Path(tmpdir) / "image.png"
            image_path.write_bytes(b"image")
            config_path = run_dir / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            spec = ExperimentSpec(
                experiment_id="run_hybrid",
                family="hybrid_current",
                variant="baseline",
                parameters={
                    "knn_k": 5,
                    "hex_radius_factor": 1.2,
                    "tile_size_factor": 6.0,
                    "tile_overlap_factor": 2.0,
                    "area_limit_factor": 10.0,
                    "adaptive_area_min": 18,
                    "adaptive_area_max": 220,
                    "adaptive_circularity_min": 0.5,
                    "adaptive_solidity_min": 0.8,
                    "adaptive_intensity_max": 140.0,
                    "nms_size": 9,
                    "border": 6,
                    "max_peaks": 300,
                },
            )

            command = build_prediction_command(
                spec=spec,
                image_path=image_path,
                run_dir=run_dir,
                config_path=config_path,
            )

            rendered = " ".join(command)
            self.assertIn("--knn_k 5", rendered)
            self.assertIn("--hex_radius_factor 1.2", rendered)
            self.assertIn("--tile_size_factor 6.0", rendered)
            self.assertIn("--tile_overlap_factor 2.0", rendered)
            self.assertIn("--area_limit_factor 10.0", rendered)
            self.assertIn("--adaptive_area_min 18", rendered)
            self.assertIn("--adaptive_area_max 220", rendered)
            self.assertIn("--adaptive_circularity_min 0.5", rendered)
            self.assertIn("--adaptive_solidity_min 0.8", rendered)
            self.assertIn("--adaptive_intensity_max 140.0", rendered)
            self.assertIn("--nms_size 9", rendered)
            self.assertIn("--border 6", rendered)
            self.assertIn("--max_peaks 300", rendered)

    def test_parse_evaluation_results_reads_per_image_and_macro_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.csv"
            results_path.write_text(
                "\n".join(
                    [
                        "image,TP,FP,FN,precision,recall,F1,mean_IoU",
                        "img_a,2,1,0,0.667,1.000,0.800,0.700",
                        "img_b,1,0,1,1.000,0.500,0.667,0.600",
                        "MACRO (n=2),,,,0.833,0.750,0.734,0.650",
                        "MICRO,3,1,1,0.750,0.750,0.750,",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = parse_evaluation_results(results_path)

            self.assertEqual(len(parsed["per_image"]), 2)
            self.assertEqual(parsed["aggregate"]["F1"], 0.734)
            self.assertEqual(parsed["aggregate"]["precision"], 0.833)


if __name__ == "__main__":
    unittest.main()
