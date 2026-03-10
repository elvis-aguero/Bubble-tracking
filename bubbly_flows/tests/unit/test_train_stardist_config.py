import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_stardist.py"


class _FakeNumpy(types.SimpleNamespace):
    def __init__(self):
        super().__init__(float32="float32", uint16="uint16")


def load_train_stardist():
    spec = importlib.util.spec_from_file_location("train_stardist_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.dict(sys.modules, {"numpy": _FakeNumpy()}, clear=False):
        spec.loader.exec_module(module)
    return module


class TrainStarDistConfigTests(unittest.TestCase):
    def test_parse_args_requires_config(self):
        module = load_train_stardist()
        args = module.parse_args([
            "--dataset", "dataset",
            "--name", "run1",
            "--config", "configs/stardist.json",
        ])

        self.assertEqual(args.dataset, Path("dataset"))
        self.assertEqual(args.name, "run1")
        self.assertEqual(args.config, Path("configs/stardist.json"))
        self.assertFalse(hasattr(args, "epochs"))

    def test_load_training_config_reads_stardist_settings(self):
        module = load_train_stardist()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "stardist.json"
            config_path.write_text(
                """
                {
                  "model": "stardist",
                  "training": {
                    "epochs": 120,
                    "batch_size": 3,
                    "val_fraction": 0.2,
                    "n_rays": 48,
                    "grid": [2, 2],
                    "patch_shape": 1024
                  }
                }
                """.strip(),
                encoding="ascii",
            )

            cfg = module.load_training_config(config_path)

        self.assertEqual(cfg["epochs"], 120)
        self.assertEqual(cfg["batch_size"], 3)
        self.assertEqual(cfg["val_fraction"], 0.2)
        self.assertEqual(cfg["n_rays"], 48)
        self.assertEqual(cfg["grid"], (2, 2))
        self.assertEqual(cfg["patch_shape"], 1024)


if __name__ == "__main__":
    unittest.main()
