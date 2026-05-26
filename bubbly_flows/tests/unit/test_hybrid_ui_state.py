import unittest
import subprocess
import sys
from pathlib import Path

from bubbly_flows.tests.src.experiments.one_image_tuning import HYBRID_BASELINE_PARAMETERS
from bubbly_flows.tests.ui.state import HybridUIState


class HybridUIStateTests(unittest.TestCase):
    def test_reset_restores_baseline_parameters(self):
        state = HybridUIState(baseline_parameters=HYBRID_BASELINE_PARAMETERS, debounce_seconds=5.0)

        state.update_params({"r_min": 5}, now=10.0)
        self.assertEqual(state.current_params["r_min"], 5)

        state.reset_to_baseline(now=11.0)
        self.assertEqual(state.current_params["r_min"], HYBRID_BASELINE_PARAMETERS["r_min"])
        self.assertEqual(state.status, "waiting")

    def test_debounce_waits_five_seconds_before_starting_run(self):
        state = HybridUIState(baseline_parameters=HYBRID_BASELINE_PARAMETERS, debounce_seconds=5.0)

        state.update_params({"r_min": 5}, now=100.0)
        self.assertIsNone(state.maybe_begin_run(now=104.9))

        run = state.maybe_begin_run(now=105.0)
        self.assertIsNotNone(run)
        self.assertEqual(run["r_min"], 5)
        self.assertEqual(state.status, "running")

    def test_pending_rerun_is_scheduled_while_run_is_active(self):
        state = HybridUIState(baseline_parameters=HYBRID_BASELINE_PARAMETERS, debounce_seconds=5.0)

        state.update_params({"r_min": 5}, now=0.0)
        first_run = state.maybe_begin_run(now=5.0)
        self.assertIsNotNone(first_run)

        state.update_params({"r_max": 22}, now=6.0)
        self.assertTrue(state.pending_rerun)
        self.assertIsNone(state.maybe_begin_run(now=11.0))

        state.finish_run(summary={"combined_instances": 100}, now=12.0)
        self.assertEqual(state.status, "waiting")
        second_run = state.maybe_begin_run(now=12.0)
        self.assertIsNotNone(second_run)
        self.assertEqual(second_run["r_min"], 5)
        self.assertEqual(second_run["r_max"], 22)

    def test_snapshot_contains_status_and_latest_summary(self):
        state = HybridUIState(baseline_parameters=HYBRID_BASELINE_PARAMETERS, debounce_seconds=5.0)
        snapshot = state.snapshot()

        self.assertEqual(snapshot["status"], "idle")
        self.assertIn("params", snapshot)
        self.assertIn("summary", snapshot)
        self.assertEqual(snapshot["summary"], {})

    def test_run_hybrid_ui_help_works_when_invoked_by_path(self):
        repo_root = Path(__file__).resolve().parents[3]
        script_path = repo_root / "bubbly_flows" / "tests" / "run_hybrid_ui.py"

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Run the local hybrid tuning UI.", result.stdout)


if __name__ == "__main__":
    unittest.main()
