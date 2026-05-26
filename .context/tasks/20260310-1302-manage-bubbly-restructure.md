# Task 20260310-1302-manage-bubbly-restructure

## Summary
- Continue the `manage_bubbly.py` pipeline restructure from the March 9 design/plan docs.
- First establish the exact implementation slice and design approval.
- Then execute the approved slice with disciplined workflow and verification.

## Owner
- codex/20260310-1302

## Status
- open

## Plan
- [ ] Review current repo state against `docs/plans/2026-03-09-pipeline-restructure-*.md`.
- [ ] Brainstorm and confirm the next implementation slice with the user.
- [ ] Implement the approved slice with tests first.
- [ ] Verify behavior and update task log.

## Log
- 2026-03-10 13:02 EST: Created task after reviewing protocol, KB, plan/design docs, and current `manage_bubbly.py` state.
- 2026-03-10 13:02 EST: Observed partial refactor state: `configs/` exists, `pipeline/` rename is present, `train.py` uses `--config`, but `manage_bubbly.py` still uses the legacy 7-option menu and training/evaluation flow is incomplete.
- 2026-03-10 13:02 EST: Observed dirty worktree on `main` with unrelated user-owned changes in `bubbly_flows/scripts/train.py`, `bubbly_flows/scripts/evaluate.py`, plus untracked logs/outputs; these must remain untouched.

## Messages
- 
- 2026-03-10 13:10 EST: User approved first execution slice: replace legacy top-level menu with planned 4-option menu plus Advanced submenu, keeping existing handlers as adapters for now. Out of scope for this slice: state line, prerequisite gating, submenu redesign internals, train/evaluate/inference cleanup.
- 2026-03-10 13:18 EST: Implemented first menu-structure slice in manage_bubbly.py: new 4-option top-level menu, extracted run_inference_menu(), and added advanced_menu() routing to existing pool/workspace/export handlers. Verification intentionally deferred at user request to avoid non-lightweight work on login node.
- 2026-03-10 13:33 EST: Verified first feature in bubbly-train-env: main_menu now routes Promote/Train/Evaluate/Inference with an Advanced submenu, backed by new unit tests.
- 2026-03-10 13:33 EST: Implemented and verified second feature in bubbly-train-env: menu now prints a state line showing latest gold dataset version, latest train/test dataset pair, and latest trained run based on directory scan.
- 2026-03-10 13:43 EST: Implemented prerequisite gating in main_menu(): Train now blocks when no *_train dataset exists; Evaluate and Inference now block when no trained run exists in scratch. Verified with unit tests in bubbly-train-env.
- 2026-03-10 13:43 EST: Updated README.md and TRAINING_GUIDE.md to reflect the new top-level menu, Advanced submenu export path, status line, blocking messages, and MicroSAM config-driven training via configs/microsam.json.
- 2026-03-10 14:03 EST: Redesigned submit_training_job() to select model family first (MicroSAM, StarDist, YOLOv9, Other), restrict datasets to *_train exports, and support structured custom trainer discovery plus manual fallback. Verified with unit tests in bubbly-train-env.
- 2026-03-10 14:03 EST: Updated docs/plans/2026-03-09-pipeline-restructure-plan.md with checklist-style implementation status and manage_bubbly milestones. Synced TRAINING_GUIDE.md with the new train-menu flow and current config-wiring state.
- 2026-03-10 14:12 EST: Refactored train_stardist.py to require --config and read training hyperparameters (epochs, batch_size, val_fraction, n_rays, grid, patch_shape) from configs/stardist.json. Verified with focused unit tests and syntax compile.
- 2026-03-10 14:12 EST: Direct conda-environment execution of train_stardist.py on this cluster still hits an OpenMP shared-memory import failure unrelated to the refactor; unit-test verification uses a numpy stub to isolate CLI/config behavior.
- 2026-03-10 14:20 EST: Refactored train_yolov9.py to require --config and read training hyperparameters (epochs, imgsz, batch, val_fraction) from configs/yolov9.json. Verified with focused unit tests and syntax compile.
- 2026-03-10 14:20 EST: Updated plan checklist: both built-in non-MicroSAM trainers are now config-native. Remaining provenance task is copying chosen config into the submitted run directory.
- 2026-03-10 14:29 EST: Completed Task 5 provenance wiring in manage_bubbly.py: built-in trainers are submitted with --config and successful submission copies the chosen config to ~/scratch/bubble-models/trained/<run>/config.json. Verified through manage_bubbly unit tests.
- 2026-03-10 14:29 EST: Synced TRAINING_GUIDE.md and docs/plans checklist to reflect that MicroSAM, StarDist, and YOLOv9 are all config-native and that submission now writes a frozen config provenance record.
- 2026-03-10 14:36 EST: Added one-line tooltips to the top-level main menu and Advanced submenu in manage_bubbly.py. Verified with menu render assertions in the manage_bubbly unit test suite.
- 2026-03-10 14:45 EST: Cleaned up Evaluate-on-Test-Set selection flow in manage_bubbly.py by adding dedicated helpers for *_test dataset discovery and trained-run model-type detection. evaluate_model() now uses those helpers with clearer, centralized block behavior. Verified with expanded manage_bubbly unit tests.
- 2026-03-10 14:52 EST: Cleaned up Inference-on-Image selection flow in manage_bubbly.py to use scratch-trained runs and centralized model-type detection, matching the evaluate flow structure. Verified with expanded manage_bubbly unit tests.

## 2026-03-10 15:56 Workflow Validation
- Interactive `manage_bubbly.py` validation on node2333 is blocked by an environment/runtime issue before menu startup: importing `cv2` inside `bubbly-train-env` aborts with `OMP: Error #179: Function Can't open SHM2 failed`.
- Confirmed this is not specific to `manage_bubbly.py`; `python3 -c "import cv2"` in the env reproduces the same abort on this node.
- Switched workflow validation to Slurm, which is the correct heavy-work boundary on Oscar.
- Submitted training validation job via Slurm script `bubbly_flows/logs/validate_train_microsam_20260310.sh`.
  - Job ID: `670345`
  - Name: `wfval_msam`
  - Status at check: `RUNNING` on `gpu2507`
  - Command: `train.py --dataset seed_v04_train --name wf_validate_microsam_20260310 --config configs/microsam.json --save_root ~/scratch/bubble-models/trained`
- Submitted evaluation validation job via Slurm script `bubbly_flows/logs/validate_eval_stardist_20260310.sh`.
  - Job ID: `670347`
  - Name: `wfval_eval_sd`
  - Produced predictions and metrics successfully.
  - Output CSV: `bubbly_flows/tests/output/eval_preds/stardist_seed_v04_run1_slurm/results.csv`
  - Summary metrics from log:
    - Total: TP=13 FP=7 FN=1284
    - Macro: precision=0.571 recall=0.007 F1=0.014 mean_IoU=0.538
    - Micro: precision=0.650 recall=0.010 F1=0.020

## 2026-03-11 10:25 Train+Auto-Eval Feature
- Approved workflow change: remove standalone Evaluate menu entry and make training submit a combined train+evaluate Slurm job.
- Added paired dataset enforcement: selecting `<stem>_train` now requires `<stem>_test`.
- Updated generated Slurm training scripts to run post-train inference and `evaluate.py`, writing outputs to `~/scratch/bubble-models/trained/<run>/eval/`.
- Updated docs (`README.md`, `TRAINING_GUIDE.md`, `USER_GUIDE.md`) and plan checklist to reflect automatic evaluation and new menu numbering.
- Verified with `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu -v`.

## 2026-03-11 Project Checkpoint
- Pipeline restructure status:
  - Main menu simplified to Promote / Train / Inference / Advanced.
  - Training is now config-native for MicroSAM, StarDist, and YOLOv9.
  - Training submissions copy `config.json` into the run directory for provenance.
  - Standalone Evaluate menu entry removed.
  - Training now requires paired `<stem>_train` + `<stem>_test` datasets.
  - Generated training Slurm jobs now perform automatic evaluation on the paired test split and write metrics to `~/scratch/bubble-models/trained/<run>/eval/results.csv`.
  - Inference remains top-level option 3.
- Team docs synced: `README.md`, `TRAINING_GUIDE.md`, `USER_GUIDE.md`, and `docs/plans/2026-03-09-pipeline-restructure-plan.md` updated for the new workflow.
- New implementation plan saved at `docs/plans/2026-03-11-train-auto-eval-plan.md`.
- Verification status:
  - `python3 -m unittest bubbly_flows.tests.unit.test_manage_bubbly_menu bubbly_flows.tests.unit.test_train_stardist_config bubbly_flows.tests.unit.test_train_yolov9_config -v` passed (`25` tests).
- Live cluster validation:
  - Prior MicroSAM workflow validation job completed successfully (`670345`).
  - Prior StarDist evaluation validation job completed successfully (`670347`).
  - Real YOLOv9 train+auto-eval submission created from the new workflow script: `submit_yolov9_seed_v04_autoeval_20260311.sh` with Slurm job `682129`.
- Temporary validation artifacts still present and not yet cleaned up:
  - `bubbly_flows/logs/validate_train_microsam_20260310.sh`
  - `bubbly_flows/logs/validate_eval_stardist_20260310.sh`
  - `bubbly_flows/logs/wfval_*`
  - `bubbly_flows/tests/output/eval_preds/stardist_seed_v04_run1_slurm/`

## 2026-03-11 Hybrid Inventory Feature
- Added `bubbly_flows/tests/inventory.py` to scan the hybrid research workspace and classify:
  - pipeline entrypoints
  - helper modules
  - sample inputs
  - outputs
  - logs
- Generated `bubbly_flows/tests/experiment_registry.json` as a machine-readable manifest.
- Generated `bubbly_flows/tests/EXPERIMENT_INDEX.md` as a human-readable index of tested hybrid artifacts.
- Verified with `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`.
- Scratch audit did not reveal extra active hybrid scripts outside the repo; the active research surface appears to already be in `bubbly_flows/tests/`.

## 2026-03-11 Hybrid Layout Reorg
- Reorganized active hybrid research code under `bubbly_flows/tests/src/` by category:
  - `src/hybrid/`
  - `src/sam3/`
  - `src/deterministic/`
  - `src/prompting/`
  - `src/backends/`
  - `src/common/bubble_sam3/`
- Kept outputs, logs, sample images, and generated artifacts in place for provenance.
- Updated internal imports for the moved files.
- Updated the inventory scanner and regenerated `EXPERIMENT_INDEX.md` / `experiment_registry.json` so the new layout is the source of truth.
- Verified with `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`.

## 2026-03-12 Hybrid Experiment Metadata
- Added `bubbly_flows/tests/experiment_metadata.json` as a minimal curated provenance layer for top-level experiment scripts.
- The metadata stays intentionally light and experiment-oriented. Each entry records:
  - `path`
  - `label`
  - `status`
  - `components`
  - `question`
  - `outputs`
  - `notes`
- Updated `bubbly_flows/tests/inventory.py` to load curated metadata and merge it into the generated Markdown index without changing the filesystem-discovery registry format.
- Regenerated `bubbly_flows/tests/EXPERIMENT_INDEX.md` so the index now shows both discovered artifacts and short notes about what each experiment is trying to test.
- Verified with `python3 -m unittest bubbly_flows.tests.unit.test_experiment_inventory -v`.

## 2026-03-12 Hybrid Experiment Harness
- Added a new provenance-first experiment harness package under `bubbly_flows/tests/src/experiments/`:
  - `gold_eval.py`
  - `provenance.py`
  - `runner.py`
  - `search_space.py`
  - `variant_executor.py`
- Implemented a cached gold-evaluation prep path that:
  - scans `annotations/gold/<version>/labels_json`
  - resolves source images
  - rasterizes LabelMe polygons into instance-mask rows
  - writes an evaluation-set manifest under `bubbly_flows/tests/output/...`
- Implemented run-level provenance writing:
  - `manifest.json`
  - `aggregate_metrics.json`
  - `per_image_metrics.csv`
  - `gallery.md`
  - `ranking.csv`
- Implemented the initial search-space generator for the three starting families:
  - `frst_only`
  - `blackhat_only`
  - `hybrid_current`
  with explicit hybrid fusion variants:
  - `current`
  - `conservative`
  - `branch_priority`
- Integrated the real family executor path into the runner:
  - materializes per-run variant config JSON
  - builds family-specific commands against `tests/src/hybrid/bubble_frst_sam3_mask.py`
  - converts predicted LabelMe JSON into instance-mask files for evaluation
  - reuses `bubbly_flows/scripts/evaluate.py` for metric computation
  - exposes `execute_real_experiment_batch(...)` as the concrete runner entrypoint
- Real runtime validation status:
  - Attempted a 3-family 2-image Slurm batch under `bubbly_flows/tests/output/experiments/validate_2img_seed_v04/`
  - Resolved two harness/runtime integration issues:
    - Slurm shell wrapper failed with `set -u` + `source ~/.bashrc`
    - hybrid script had to be launched as a module (`python -m ...`) rather than by file path so `bubbly_flows` imports resolve
  - Current environment blocker for `frst_only` / `hybrid_current`:
    - `transformers` with SAM3 tracker classes is missing
    - `sam3` package is missing
  - Successful end-to-end Slurm validation completed for `blackhat_only` on 2 gold images:
    - job `698942`
    - outputs written under `bubbly_flows/tests/output/experiments/validate_2img_blackhat_only/`
    - provenance artifacts created (`manifest.json`, `aggregate_metrics.json`, `per_image_metrics.csv`, `gallery.md`, `ranking.csv`, prediction masks, overlays, LabelMe JSON)
    - aggregate result on the 2-image slice: `F1=0.0`
- Verified with lightweight login-node-safe commands only:
  - `python3 -m unittest bubbly_flows.tests.unit.test_hybrid_experiment_harness bubbly_flows.tests.unit.test_experiment_inventory -v`
  - `python3 -m py_compile bubbly_flows/tests/src/experiments/__init__.py bubbly_flows/tests/src/experiments/gold_eval.py bubbly_flows/tests/src/experiments/provenance.py bubbly_flows/tests/src/experiments/runner.py bubbly_flows/tests/src/experiments/search_space.py bubbly_flows/tests/src/experiments/variant_executor.py`
