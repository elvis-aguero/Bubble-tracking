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
