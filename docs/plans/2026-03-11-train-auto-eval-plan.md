# Train With Automatic Evaluation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the separate Evaluate menu action and make every training submission automatically evaluate the newly trained run on the paired `*_test` split through Slurm.

**Architecture:** Keep the existing run-selection and model-selection UX for training, but tighten dataset validation so a selected `*_train` dataset must have a matching `*_test` dataset. Extend the generated training Slurm script to run post-train inference and `evaluate.py`, writing predictions and metrics into the trained run directory. Remove the top-level Evaluate option and update the state/menu/docs accordingly.

**Tech Stack:** Python 3.11, Slurm (`sbatch`), existing `manage_bubbly.py` CLI, MicroSAM / StarDist / YOLOv9 trainers, `unittest`.

---

## File Map

- Modify: `bubbly_flows/scripts/manage_bubbly.py`
  - Remove the standalone Evaluate menu entry.
  - Add paired train/test dataset validation helpers.
  - Extend training job generation with post-train evaluation commands.
  - Route evaluation outputs into the trained run directory.
- Modify: `bubbly_flows/tests/unit/test_manage_bubbly_menu.py`
  - Add failing tests for menu removal, paired test enforcement, and generated Slurm evaluation content.
- Modify: `README.md`
  - Update workflow description from train-then-evaluate to train-with-automatic-evaluation.
- Modify: `TRAINING_GUIDE.md`
  - Update operator flow, outputs, and cluster-safe evaluation behavior.
- Modify: `USER_GUIDE.md`
  - Update workflow steps and menu numbering.
- Modify: `docs/plans/2026-03-09-pipeline-restructure-plan.md`
  - Mark checklist progress and note that evaluation is now automatic within training.
- Update: `.context/tasks/20260310-1302-manage-bubbly-restructure.md`
  - Append implementation and verification notes.

## Chunk 1: Lock Behavior With Tests

### Task 1: Remove standalone Evaluate menu entry

**Files:**
- Modify: `bubbly_flows/tests/unit/test_manage_bubbly_menu.py`
- Modify: `bubbly_flows/scripts/manage_bubbly.py`

- [ ] **Step 1: Write the failing test**

```python
def test_main_menu_shows_train_and_inference_without_evaluate(self):
    ...
    self.assertIn("2. Train Model", output)
    self.assertNotIn("Evaluate on Test Set", output)
    self.assertIn("3. Inference on Image", output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: FAIL because the old menu still includes `Evaluate on Test Set`.

- [ ] **Step 3: Write minimal implementation**

Update `main_menu()` numbering and remove the Evaluate branch.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: PASS for the new menu test.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/unit/test_manage_bubbly_menu.py bubbly_flows/scripts/manage_bubbly.py
git commit -m "refactor: remove standalone evaluate menu entry"
```

### Task 2: Require paired `*_test` split for training submission

**Files:**
- Modify: `bubbly_flows/tests/unit/test_manage_bubbly_menu.py`
- Modify: `bubbly_flows/scripts/manage_bubbly.py`

- [ ] **Step 1: Write the failing test**

```python
def test_submit_training_job_blocks_when_matching_test_split_missing(self):
    ...
    self.assertIn("Matching test dataset not found", output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: FAIL because training currently allows submission with only `*_train` present.

- [ ] **Step 3: Write minimal implementation**

Add a helper that maps `seed_v04_train -> seed_v04_test` and block submission when the paired test dataset is missing.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: PASS for the new pairing guard.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/unit/test_manage_bubbly_menu.py bubbly_flows/scripts/manage_bubbly.py
git commit -m "feat: require paired test split for training"
```

## Chunk 2: Generate Train+Eval Slurm Jobs

### Task 3: Add post-train evaluation commands to generated Slurm script

**Files:**
- Modify: `bubbly_flows/tests/unit/test_manage_bubbly_menu.py`
- Modify: `bubbly_flows/scripts/manage_bubbly.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_submit_training_job_generates_eval_commands_for_microsam(self):
    ...
    self.assertIn("evaluate.py", script_text)
    self.assertIn("results.csv", script_text)
    self.assertIn("seed_v04_test", script_text)


def test_submit_training_job_writes_eval_outputs_into_run_dir(self):
    ...
    self.assertIn("eval_preds", script_text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: FAIL because the Slurm script currently only trains.

- [ ] **Step 3: Write minimal implementation**

Extend `submit_training_job()` to append model-specific post-train inference plus `evaluate.py` to the generated Slurm script, targeting the paired test split and run-local outputs.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: PASS for the new script-content assertions.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/unit/test_manage_bubbly_menu.py bubbly_flows/scripts/manage_bubbly.py
git commit -m "feat: add automatic evaluation to training jobs"
```

### Task 4: Preserve provenance and standard outputs in run directory

**Files:**
- Modify: `bubbly_flows/tests/unit/test_manage_bubbly_menu.py`
- Modify: `bubbly_flows/scripts/manage_bubbly.py`

- [ ] **Step 1: Write the failing test**

```python
def test_submit_training_job_stores_results_csv_under_run_dir(self):
    ...
    self.assertIn("$RUN_DIR/eval/results.csv", script_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: FAIL because evaluation output is not yet written under the run directory.

- [ ] **Step 3: Write minimal implementation**

Route predictions and `results.csv` under `~/scratch/bubble-models/trained/<run>/eval/` and keep `config.json` provenance copy in place.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/tests/unit/test_manage_bubbly_menu.py bubbly_flows/scripts/manage_bubbly.py
git commit -m "feat: store evaluation outputs with trained run"
```

## Chunk 3: Documentation And Verification

### Task 5: Update team docs and plan checklist

**Files:**
- Modify: `README.md`
- Modify: `TRAINING_GUIDE.md`
- Modify: `USER_GUIDE.md`
- Modify: `docs/plans/2026-03-09-pipeline-restructure-plan.md`
- Update: `.context/tasks/20260310-1302-manage-bubbly-restructure.md`

- [ ] **Step 1: Write the failing doc expectations mentally from the approved design**

Required doc changes:
- training now includes automatic evaluation on paired test split
- no separate Evaluate menu item
- outputs include run-local predictions, `results.csv`, and training curves/logs

- [ ] **Step 2: Update the docs minimally**

Describe the new workflow, output locations, and operator expectations.

- [ ] **Step 3: Run focused verification**

Run: `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`
Expected: PASS.

Run: `python3 -m unittest bubbly_flows.tests.unit.test_train_stardist_config bubbly_flows.tests.unit.test_train_yolov9_config -v`
Expected: PASS.

- [ ] **Step 4: Inspect changed files**

Run: `git diff -- bubbly_flows/scripts/manage_bubbly.py bubbly_flows/tests/unit/test_manage_bubbly_menu.py README.md TRAINING_GUIDE.md USER_GUIDE.md docs/plans/2026-03-09-pipeline-restructure-plan.md`
Expected: Diff matches only this feature.

- [ ] **Step 5: Commit**

```bash
git add bubbly_flows/scripts/manage_bubbly.py bubbly_flows/tests/unit/test_manage_bubbly_menu.py README.md TRAINING_GUIDE.md USER_GUIDE.md docs/plans/2026-03-09-pipeline-restructure-plan.md .context/tasks/20260310-1302-manage-bubbly-restructure.md docs/plans/2026-03-11-train-auto-eval-plan.md
git commit -m "refactor: make training run automatic evaluation"
```
