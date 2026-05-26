import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "inventory.py"


def load_inventory_module():
    spec = importlib.util.spec_from_file_location("experiment_inventory_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExperimentInventoryTests(unittest.TestCase):
    def test_build_registry_groups_hybrid_entrypoints_outputs_and_logs(self):
        module = load_inventory_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "bubble_frst_sam3_mask.py").write_text("", encoding="ascii")
            (root / "bubble_sam3" / "pipeline.py").parent.mkdir(parents=True)
            (root / "bubble_sam3" / "pipeline.py").write_text("", encoding="ascii")
            (root / "output" / "result_frst.png").parent.mkdir(parents=True)
            (root / "output" / "result_frst.png").write_text("", encoding="ascii")
            (root / "logs" / "bubble_frst_sam3_20260201.log").parent.mkdir(parents=True)
            (root / "logs" / "bubble_frst_sam3_20260201.log").write_text("", encoding="ascii")
            (root / "img6001.png").write_text("", encoding="ascii")

            registry = module.build_registry(root)

        self.assertEqual([item["name"] for item in registry["pipeline_entrypoints"]], ["bubble_frst_sam3_mask.py"])
        self.assertEqual([item["path"] for item in registry["helper_modules"]], ["bubble_sam3/pipeline.py"])
        self.assertEqual([item["path"] for item in registry["sample_inputs"]], ["img6001.png"])
        self.assertEqual([item["path"] for item in registry["outputs"]], ["output/result_frst.png"])
        self.assertEqual([item["path"] for item in registry["logs"]], ["logs/bubble_frst_sam3_20260201.log"])

    def test_build_registry_detects_known_pipeline_families(self):
        module = load_inventory_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in [
                "bubble_frst_sam3_mask.py",
                "bubble_sam3_mask.py",
                "blackhat_mask.py",
                "classical_test.py",
                "big_bubble_prompt_fb.py",
            ]:
                (root / name).write_text("", encoding="ascii")

            registry = module.build_registry(root)

        families = {item["name"]: item["family"] for item in registry["pipeline_entrypoints"]}
        self.assertEqual(families["bubble_frst_sam3_mask.py"], "hybrid")
        self.assertEqual(families["bubble_sam3_mask.py"], "sam3")
        self.assertEqual(families["blackhat_mask.py"], "deterministic")
        self.assertEqual(families["classical_test.py"], "deterministic")
        self.assertEqual(families["big_bubble_prompt_fb.py"], "prompting")

    def test_render_markdown_lists_known_hybrid_scripts_and_outputs(self):
        module = load_inventory_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "bubble_frst_sam3_mask.py").write_text("", encoding="ascii")
            (root / "blackhat_mask.py").write_text("", encoding="ascii")
            (root / "output" / "result_frst.png").parent.mkdir(parents=True)
            (root / "output" / "result_frst.png").write_text("", encoding="ascii")
            (root / "logs" / "bubble_frst_sam3_20260201.log").parent.mkdir(parents=True)
            (root / "logs" / "bubble_frst_sam3_20260201.log").write_text("", encoding="ascii")

            registry = module.build_registry(root)
            markdown = module.render_markdown(registry)

        self.assertIn("bubble_frst_sam3_mask.py", markdown)
        self.assertIn("blackhat_mask.py", markdown)
        self.assertIn("output/result_frst.png", markdown)
        self.assertIn("logs/bubble_frst_sam3_20260201.log", markdown)

    def test_build_registry_detects_src_category_entrypoints(self):
        module = load_inventory_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src" / "hybrid" / "bubble_frst_sam3_mask.py").parent.mkdir(parents=True)
            (root / "src" / "hybrid" / "bubble_frst_sam3_mask.py").write_text("", encoding="ascii")
            (root / "src" / "deterministic" / "blackhat_mask.py").parent.mkdir(parents=True)
            (root / "src" / "deterministic" / "blackhat_mask.py").write_text("", encoding="ascii")

            registry = module.build_registry(root)

        entrypoints = {item["path"]: item["family"] for item in registry["pipeline_entrypoints"]}
        self.assertEqual(entrypoints["src/hybrid/bubble_frst_sam3_mask.py"], "hybrid")
        self.assertEqual(entrypoints["src/deterministic/blackhat_mask.py"], "deterministic")

    def test_build_registry_detects_common_modules_under_src(self):
        module = load_inventory_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src" / "common" / "bubble_sam3" / "pipeline.py").parent.mkdir(parents=True)
            (root / "src" / "common" / "bubble_sam3" / "pipeline.py").write_text("", encoding="ascii")

            registry = module.build_registry(root)

        self.assertEqual([item["path"] for item in registry["helper_modules"]], ["src/common/bubble_sam3/pipeline.py"])

    def test_render_markdown_includes_curated_experiment_metadata(self):
        module = load_inventory_module()
        registry = {
            "pipeline_entrypoints": [
                {
                    "name": "bubble_frst_sam3_mask.py",
                    "path": "src/hybrid/bubble_frst_sam3_mask.py",
                    "family": "hybrid",
                }
            ],
            "helper_modules": [],
            "sample_inputs": [],
            "outputs": [],
            "logs": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "experiment_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "experiments": [
                            {
                                "path": "src/hybrid/bubble_frst_sam3_mask.py",
                                "label": "FRST + SAM3 hybrid",
                                "status": "tentative",
                                "components": ["FRST", "SAM3", "blackhat"],
                                "question": "Can classical proposals and SAM3 cover both small and large bubbles?",
                                "outputs": ["output/result6001.png"],
                                "notes": "Primary hybrid workbench.",
                            }
                        ]
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="ascii",
            )

            metadata = module.load_experiment_metadata(metadata_path)
            markdown = module.render_markdown(registry, metadata)

        self.assertIn("FRST + SAM3 hybrid", markdown)
        self.assertIn("Status: tentative", markdown)
        self.assertIn("Components: FRST, SAM3, blackhat", markdown)
        self.assertIn(
            "Question: Can classical proposals and SAM3 cover both small and large bubbles?",
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
