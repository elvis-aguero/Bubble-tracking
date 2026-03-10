import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_yolov9.py"


class _FakeNumpy(types.SimpleNamespace):
    def __init__(self):
        super().__init__(ndarray=object, float32="float32", uint16="uint16")


def load_train_yolov9():
    spec = importlib.util.spec_from_file_location("train_yolov9_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with mock.patch.dict(sys.modules, {"numpy": _FakeNumpy()}, clear=False):
        spec.loader.exec_module(module)
    return module


class TrainYoloConfigTests(unittest.TestCase):
    def test_parse_args_requires_config(self):
        module = load_train_yolov9()
        args = module.parse_args([
            "--dataset", "dataset",
            "--name", "run1",
            "--config", "configs/yolov9.json",
        ])

        self.assertEqual(args.dataset, Path("dataset"))
        self.assertEqual(args.name, "run1")
        self.assertEqual(args.config, Path("configs/yolov9.json"))
        self.assertFalse(hasattr(args, "epochs"))

    def test_load_training_config_reads_yolo_settings(self):
        module = load_train_yolov9()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "yolov9.json"
            config_path.write_text(
                """
                {
                  "model": "yolov9",
                  "training": {
                    "epochs": 120,
                    "imgsz": 960,
                    "batch": 6,
                    "val_fraction": 0.2
                  }
                }
                """.strip(),
                encoding="ascii",
            )

            cfg = module.load_training_config(config_path)

        self.assertEqual(cfg["epochs"], 120)
        self.assertEqual(cfg["imgsz"], 960)
        self.assertEqual(cfg["batch"], 6)
        self.assertEqual(cfg["val_fraction"], 0.2)


if __name__ == "__main__":
    unittest.main()
