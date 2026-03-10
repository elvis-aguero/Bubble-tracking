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
