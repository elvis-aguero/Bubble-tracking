import importlib.util
import io
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "manage_bubbly.py"


class _FakeGenerator:
    pass


class _FakeNumpy(types.SimpleNamespace):
    def __init__(self):
        super().__init__(
            ndarray=object,
            uint16="uint16",
            float32="float32",
            int32="int32",
            integer=int,
            random=types.SimpleNamespace(Generator=_FakeGenerator, default_rng=lambda *args, **kwargs: _FakeGenerator()),
        )


def load_manage_bubbly():
    os.environ.setdefault("_MANAGE_SKIP_ENV_CHECK", "1")
    spec = importlib.util.spec_from_file_location("manage_bubbly_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    fake_modules = {
        "cv2": types.SimpleNamespace(),
        "numpy": _FakeNumpy(),
    }
    with mock.patch.dict(sys.modules, fake_modules, clear=False):
        spec.loader.exec_module(module)
    return module


class ManageBubblyMenuTests(unittest.TestCase):
    def test_main_menu_routes_happy_path_choices(self):
        module = load_manage_bubbly()
        calls = []
        choices = iter(["1", "2", "3", "4", "q"])

        with mock.patch.object(module, "clear_screen", lambda: None), \
             mock.patch.object(module, "banner", lambda: None), \
             mock.patch.object(module, "promote_to_gold", lambda: calls.append("promote")), \
             mock.patch.object(module, "submit_training_job", lambda: calls.append("train")), \
             mock.patch.object(module, "evaluate_model", lambda: calls.append("evaluate")), \
             mock.patch.object(module, "run_inference_menu", lambda: calls.append("infer"), create=True), \
             mock.patch.object(module, "input_str", lambda prompt, default=None: next(choices)):
            module.main_menu()

        self.assertEqual(calls, ["promote", "train", "evaluate", "infer"])

    def test_advanced_menu_routes_legacy_operations(self):
        module = load_manage_bubbly()
        calls = []
        choices = iter(["a", "1", "2", "3", "q", "q"])

        with mock.patch.object(module, "clear_screen", lambda: None), \
             mock.patch.object(module, "banner", lambda: None), \
             mock.patch.object(module, "update_pool", lambda: calls.append("pool")), \
             mock.patch.object(module, "create_workspace", lambda: calls.append("workspace")), \
             mock.patch.object(module, "export_microsam_dataset", lambda: calls.append("export")), \
             mock.patch.object(module, "input_str", lambda prompt, default=None: next(choices)):
            module.main_menu()

        self.assertEqual(calls, ["pool", "workspace", "export"])

    def test_state_line_reports_latest_gold_dataset_and_run(self):
        module = load_manage_bubbly()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_dir = root / "annotations" / "gold"
            datasets_dir = root / "pipeline" / "datasets"
            trained_dir = root / "trained"

            (gold_dir / "seed_v03").mkdir(parents=True)
            (gold_dir / "seed_v04").mkdir(parents=True)
            (datasets_dir / "seed_v03_train").mkdir(parents=True)
            (datasets_dir / "seed_v03_test").mkdir(parents=True)
            (datasets_dir / "seed_v04_train").mkdir(parents=True)
            (datasets_dir / "seed_v04_test").mkdir(parents=True)
            (trained_dir / "microsam_old_run").mkdir(parents=True)
            (trained_dir / "microsam_1024_run1").mkdir(parents=True)

            old_ts = time.time() - 60
            new_ts = time.time()
            os.utime(gold_dir / "seed_v03", (old_ts, old_ts))
            os.utime(gold_dir / "seed_v04", (new_ts, new_ts))
            os.utime(datasets_dir / "seed_v03_train", (old_ts, old_ts))
            os.utime(datasets_dir / "seed_v03_test", (old_ts, old_ts))
            os.utime(datasets_dir / "seed_v04_train", (new_ts, new_ts))
            os.utime(datasets_dir / "seed_v04_test", (new_ts, new_ts))
            os.utime(trained_dir / "microsam_old_run", (old_ts, old_ts))
            os.utime(trained_dir / "microsam_1024_run1", (new_ts, new_ts))

            line = module.format_pipeline_state_line(
                gold_dir=gold_dir,
                datasets_dir=datasets_dir,
                trained_dir=trained_dir,
            )

        self.assertEqual(
            line,
            "State: gold=seed_v04  dataset=seed_v04_train/test  last_run=microsam_1024_run1",
        )

    def test_train_is_blocked_without_exported_dataset(self):
        module = load_manage_bubbly()
        output = io.StringIO()
        choices = iter(["2", "q"])

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(module, "PIPELINE_DIR", Path(tmpdir) / "pipeline"), \
             mock.patch.object(module, "clear_screen", lambda: None), \
             mock.patch.object(module, "banner", lambda: None), \
             mock.patch.object(module, "format_pipeline_state_line", lambda: "State: gold=none  dataset=none  last_run=none"), \
             mock.patch.object(module, "submit_training_job", side_effect=AssertionError("train handler should not run")), \
             mock.patch.object(module, "input", lambda prompt='': ""), \
             mock.patch.object(module, "input_str", lambda prompt, default=None: next(choices)), \
             mock.patch("sys.stdout", output):
            module.main_menu()

        self.assertIn("No training dataset found. Run Option 1 first.", output.getvalue())

    def test_evaluate_and_inference_are_blocked_without_trained_run(self):
        module = load_manage_bubbly()
        output = io.StringIO()
        choices = iter(["3", "4", "q"])

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(module, "SCRATCH_TRAINED_DIR", Path(tmpdir) / "trained"), \
             mock.patch.object(module, "clear_screen", lambda: None), \
             mock.patch.object(module, "banner", lambda: None), \
             mock.patch.object(module, "format_pipeline_state_line", lambda: "State: gold=seed_v04  dataset=seed_v04_train/test  last_run=none"), \
             mock.patch.object(module, "evaluate_model", side_effect=AssertionError("evaluate handler should not run")), \
             mock.patch.object(module, "run_inference_menu", side_effect=AssertionError("inference handler should not run")), \
             mock.patch.object(module, "input", lambda prompt='': ""), \
             mock.patch.object(module, "input_str", lambda prompt, default=None: next(choices)), \
             mock.patch("sys.stdout", output):
            module.main_menu()

        self.assertEqual(output.getvalue().count("No trained checkpoint found. Run Option 2 first."), 2)

    def test_builtin_model_choice_maps_to_canonical_script_and_config(self):
        module = load_manage_bubbly()
        choice = module.resolve_training_model_choice("2")

        self.assertEqual(choice["label"], "stardist")
        self.assertEqual(choice["script_path"].name, "train_stardist.py")
        self.assertEqual(choice["config_path"].name, "stardist.json")

    def test_train_dataset_listing_only_includes_train_splits(self):
        module = load_manage_bubbly()
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_root = Path(tmpdir)
            (ds_root / "seed_v04_train").mkdir()
            (ds_root / "seed_v04_test").mkdir()
            (ds_root / "scratch").mkdir()
            (ds_root / "gold_v00_train").mkdir()

            datasets = module.list_training_datasets(ds_root)

        self.assertEqual(datasets, ["gold_v00_train", "seed_v04_train"])

    def test_other_model_listing_excludes_builtin_trainers(self):
        module = load_manage_bubbly()
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir)
            for name in [
                "train.py",
                "train_stardist.py",
                "train_yolov9.py",
                "train_maskrcnn.py",
                "train_unet.py",
                "helper.py",
            ]:
                (scripts_dir / name).write_text("", encoding="ascii")

            scripts = module.list_other_training_scripts(scripts_dir)

        self.assertEqual([p.name for p in scripts], ["train_maskrcnn.py", "train_unet.py"])


    def test_main_menu_shows_one_line_tooltips(self):
        module = load_manage_bubbly()
        output = io.StringIO()
        with mock.patch.object(module, "clear_screen", lambda: None),              mock.patch.object(module, "banner", lambda: None),              mock.patch.object(module, "format_pipeline_state_line", lambda: "State: gold=seed_v04  dataset=seed_v04_train/test  last_run=demo_run"),              mock.patch.object(module, "input_str", side_effect=["q"]),              mock.patch("sys.stdout", output):
            module.main_menu()

        rendered = output.getvalue()
        self.assertIn("1. Promote Workspace to Gold     - finalise annotations, create train/test split", rendered)
        self.assertIn("2. Train Model                   - submit Slurm job using configs/<model>.json", rendered)
        self.assertIn("3. Evaluate on Test Set          - run inference + metrics on held-out split", rendered)
        self.assertIn("4. Inference on Image            - run a trained model on any single image", rendered)
        self.assertIn("a. Advanced                      - pool management, workspace creation, dataset export", rendered)

    def test_advanced_menu_shows_one_line_tooltips(self):
        module = load_manage_bubbly()
        output = io.StringIO()
        with mock.patch.object(module, "clear_screen", lambda: None),              mock.patch.object(module, "banner", lambda: None),              mock.patch.object(module, "input_str", side_effect=["q"]),              mock.patch("sys.stdout", output):
            module.advanced_menu()

        rendered = output.getvalue()
        self.assertIn("1. Update Patch Pool      - scan frames dir for new images, rebuild pool index", rendered)
        self.assertIn("2. Create Workspace       - start a new annotation seed from the pool", rendered)
        self.assertIn("3. Export Dataset         - re-run train/test split on an existing gold set", rendered)

    def test_evaluate_lists_only_test_datasets(self):
        module = load_manage_bubbly()
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_root = Path(tmpdir)
            (ds_root / "seed_v04_train").mkdir()
            (ds_root / "seed_v04_test").mkdir()
            (ds_root / "seed_v05_test").mkdir()
            (ds_root / "misc").mkdir()

            datasets = module.list_test_datasets(ds_root)

        self.assertEqual(datasets, ["seed_v04_test", "seed_v05_test"])

    def test_detect_trained_model_type_identifies_known_layouts(self):
        module = load_manage_bubbly()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            micro_dir = root / "micro_run"
            (micro_dir / "checkpoints" / "micro_run").mkdir(parents=True)
            (micro_dir / "checkpoints" / "micro_run" / "best.pt").write_text("", encoding="ascii")

            yolo_dir = root / "yolo_run"
            (yolo_dir / "weights").mkdir(parents=True)
            (yolo_dir / "weights" / "best.pt").write_text("", encoding="ascii")

            sd_dir = root / "sd_run"
            sd_dir.mkdir(parents=True)
            (sd_dir / "thresholds.json").write_text("{}", encoding="ascii")

            self.assertEqual(module.detect_trained_model_type(micro_dir), ("microsam", micro_dir / "checkpoints" / "micro_run" / "best.pt"))
            self.assertEqual(module.detect_trained_model_type(yolo_dir), ("yolov9", yolo_dir / "weights" / "best.pt"))
            self.assertEqual(module.detect_trained_model_type(sd_dir), ("stardist", sd_dir))

    def test_detect_trained_model_type_returns_none_for_unknown_layout(self):
        module = load_manage_bubbly()
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "unknown"
            exp_dir.mkdir()
            self.assertIsNone(module.detect_trained_model_type(exp_dir))

    def test_inference_uses_trained_runs_directory_not_pipeline_models(self):
        module = load_manage_bubbly()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir,              mock.patch.object(module, "SCRATCH_TRAINED_DIR", Path(tmpdir) / "trained"),              mock.patch.object(module, "clear_screen", lambda: None),              mock.patch.object(module, "banner", lambda: None),              mock.patch.object(module, "input", lambda prompt='': ""),              mock.patch.object(module, "input_str", side_effect=["q"]),              mock.patch("sys.stdout", output):
            module.run_inference_menu()

        self.assertIn("No trained models found. Train a model first.", output.getvalue())

    def test_inference_detects_model_type_from_trained_run_layout(self):
        module = load_manage_bubbly()
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_root = Path(tmpdir) / "trained"
            exp_dir = trained_root / "demo_run"
            (exp_dir / "weights").mkdir(parents=True)
            (exp_dir / "weights" / "best.pt").write_text("", encoding="ascii")
            image_path = Path(tmpdir) / "img.png"
            image_path.write_text("", encoding="ascii")

            with mock.patch.object(module, "SCRATCH_TRAINED_DIR", trained_root),                  mock.patch.object(module, "SCRIPTS_DIR", Path(tmpdir) / "scripts"),                  mock.patch.object(module, "input", lambda prompt='': ""),                  mock.patch.object(module.os, "system", return_value=0) as mock_system,                  mock.patch.object(module, "input_int", side_effect=[1]),                  mock.patch.object(module, "input_str", side_effect=[str(image_path), "out.png"]):
                (module.SCRIPTS_DIR).mkdir(parents=True)
                module.run_inference_menu()

            cmd = mock_system.call_args[0][0]
            self.assertIn("weights/best.pt", cmd)
            self.assertIn("--image", cmd)
            self.assertIn("out.png", cmd)
    def test_submit_training_job_passes_config_for_builtin_models(self):
        module = load_manage_bubbly()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir,              tempfile.TemporaryDirectory() as home_tmp,              mock.patch.object(module, "ROOT_DIR", Path(tmpdir) / "bubbly_flows"),              mock.patch.object(module, "PIPELINE_DIR", Path(tmpdir) / "bubbly_flows" / "pipeline"),              mock.patch.object(module, "SCRIPTS_DIR", Path(tmpdir) / "bubbly_flows" / "scripts"),              mock.patch.object(module, "CONFIGS_DIR", Path(tmpdir) / "configs"),              mock.patch.object(module, "SCRATCH_TRAINED_DIR", Path(home_tmp) / "trained"),              mock.patch.object(module, "SCRATCH_MODELS_DIR", Path(home_tmp) / "models"),              mock.patch.object(module, "MODEL_WEIGHTS_MAP", {"train_stardist.py": Path(home_tmp) / "weights_dir"}),              mock.patch.object(module, "input", lambda prompt='': ""),              mock.patch.object(module, "input_str", side_effect=["2", "demo_run", "n"]),              mock.patch.object(module, "input_int", side_effect=[1, 4]),              mock.patch("sys.stdout", output):
            root = module.ROOT_DIR
            (root / "logs").mkdir(parents=True)
            (module.PIPELINE_DIR / "datasets" / "seed_v04_train" / "images").mkdir(parents=True)
            (module.SCRIPTS_DIR).mkdir(parents=True)
            (module.CONFIGS_DIR).mkdir(parents=True)
            (module.CONFIGS_DIR / "stardist.json").write_text('{"training": {"epochs": 123, "batch_size": 2, "val_fraction": 0.15, "n_rays": 64, "grid": [2, 2]}}', encoding='ascii')
            (module.SCRIPTS_DIR / "train_stardist.py").write_text('', encoding='ascii')
            (Path(home_tmp) / "weights_dir").mkdir(parents=True)
            (Path(home_tmp) / "weights_dir" / "dummy.h5").write_text('', encoding='ascii')

            module.submit_training_job()

            script_path = root / "logs" / "submit_demo_run.sh"
            content = script_path.read_text(encoding='ascii')

        self.assertIn("--config", content)
        self.assertIn("stardist.json", content)

    def test_submit_training_job_copies_config_after_successful_submission(self):
        module = load_manage_bubbly()
        with tempfile.TemporaryDirectory() as tmpdir,              tempfile.TemporaryDirectory() as home_tmp,              mock.patch.object(module, "ROOT_DIR", Path(tmpdir) / "bubbly_flows"),              mock.patch.object(module, "PIPELINE_DIR", Path(tmpdir) / "bubbly_flows" / "pipeline"),              mock.patch.object(module, "SCRIPTS_DIR", Path(tmpdir) / "bubbly_flows" / "scripts"),              mock.patch.object(module, "CONFIGS_DIR", Path(tmpdir) / "configs"),              mock.patch.object(module, "SCRATCH_TRAINED_DIR", Path(home_tmp) / "trained"),              mock.patch.object(module, "SCRATCH_MODELS_DIR", Path(home_tmp) / "models"),              mock.patch.object(module, "MODEL_WEIGHTS_MAP", {"train.py": Path(home_tmp) / "vit_b.pt"}),              mock.patch.object(module, "input", lambda prompt='': ""),              mock.patch.object(module, "input_str", side_effect=["1", "demo_run", "y"]),              mock.patch.object(module, "input_int", side_effect=[1, 4]),              mock.patch.object(module.os, "system", return_value=0):
            root = module.ROOT_DIR
            (root / "logs").mkdir(parents=True)
            (module.PIPELINE_DIR / "datasets" / "seed_v04_train" / "images").mkdir(parents=True)
            (module.SCRIPTS_DIR).mkdir(parents=True)
            (module.CONFIGS_DIR).mkdir(parents=True)
            config_body = '{"training": {"epochs": 100, "patch_shape": 1024, "freeze": ["image_encoder"]}}'
            (module.CONFIGS_DIR / "microsam.json").write_text(config_body, encoding='ascii')
            (module.SCRIPTS_DIR / "train.py").write_text('', encoding='ascii')
            (Path(home_tmp) / "vit_b.pt").write_text('', encoding='ascii')

            module.submit_training_job()

            copied = module.SCRATCH_TRAINED_DIR / "demo_run" / "config.json"
            self.assertTrue(copied.exists())
            self.assertEqual(copied.read_text(encoding='ascii'), config_body)


if __name__ == "__main__":
    unittest.main()
