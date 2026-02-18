# Bubble-tracking Project Structure Guide

This repository contains a full bubble-annotation and model-training workflow.
If you are new to the project, this README explains where things live and what each part is for.

For step-by-step operating instructions, see `USER_GUIDE.md`.

## 1) Repository Layout (Top Level)

```text
Bubble-tracking/
├── bubbly_flows/                # Main project code + datasets + experiment artifacts
├── .context/                    # Agent task memory/protocol (not part of runtime pipeline)
├── USER_GUIDE.md                # Operator guide (labeling/training workflow)
├── environment.yml              # Conda env spec for training + core tooling
├── launch_labeling.*            # OS-specific launchers for X-AnyLabeling
├── output/                      # Root-level ad-hoc output samples
└── x-labeling-env/              # Local Python venv used by labeling launcher
```

Most technical work happens inside `bubbly_flows/`.

## 2) Core Pipeline Map

Data flows through the project in this order:

1. Raw frames: `bubbly_flows/data/frames/images_raw/` (`.tif`)
2. Converted/processed frames: `images_16bit_png/`, `images_clahe/`
3. Patch pool (canonical patch lake): `bubbly_flows/data/patches_pool/images/`
4. Labeling workspace batches: `bubbly_flows/workspaces/<workspace>/`
5. Versioned gold labels: `bubbly_flows/annotations/gold/<gold_version>/`
6. MicroSAM training dataset export: `bubbly_flows/microsam/datasets/<dataset>/`
7. Trained checkpoints: `bubbly_flows/microsam/models/<experiment>/`

Operational logs are tracked in:
- `bubbly_flows/diary.log` (action audit log)
- `bubbly_flows/logs/` (Slurm scripts + `.out/.err`)
- `bubbly_flows/tests/logs/` (SAM3/FRST experiment logs)

## 3) `bubbly_flows/` Breakdown

```text
bubbly_flows/
├── annotations/
├── archive/
├── data/
├── logs/
├── microsam/
├── scripts/
├── tests/
├── workspaces/
└── diary.log
```

### `bubbly_flows/data/`
Long-lived input imagery and patch inventory.

- `frames/images_raw/`
  - Original frame data (`.tif`) from acquisitions.
- `frames/images_16bit_png/`
  - 16-bit PNG conversions of raw frames.
- `frames/images_clahe/`
  - Contrast-normalized frame versions (CLAHE).
- `patches_pool/images/`
  - Canonical, reusable patch pool used for workspace sampling.
  - Filenames encode source frame and tile offset (`__x####_y####`).
- `patches_pool/patch_map.csv`
  - Mapping of patches back to source imagery/tiling metadata.

### `bubbly_flows/workspaces/`
Ephemeral annotation batches for labelers.

Each workspace (for example `seed_v04/`) follows:

```text
workspaces/<workspace_name>/
├── images/          # Patch images assigned to the batch
├── labels/          # LabelMe/X-AnyLabeling JSON labels
└── manifest.csv     # Selected filenames for the batch
```

`scripts/manage_bubbly.py` creates these and later promotes labels to gold.

### `bubbly_flows/annotations/`
Versioned, curated labels (source of truth for supervised training).

- `gold/gold_*` directories are version snapshots (for example: `gold_seed_v00`, `gold_v00`).
- Each gold version typically contains:
  - `labels_json/` (all promoted annotation JSON files)
  - `manifest.csv` (tracked labeled stems)
  - `stats.json` (simple counts/metadata)

### `bubbly_flows/microsam/`
Training inputs and trained model artifacts.

- `datasets/<dataset_name>/images/`
  - Training image patches.
- `datasets/<dataset_name>/labels/`
  - Instance-ID masks (`.tif`) generated from polygon JSON.
- `models/<experiment_name>/`
  - Checkpoints, including `best.pt` under MicroSAM checkpoint folders.

### `bubbly_flows/scripts/`
Primary pipeline entry points.

- `manage_bubbly.py`
  - Main interactive controller.
  - Handles pool update, workspace creation, gold promotion, dataset export, Slurm training submission, inference launch.
- `utils.py`
  - End-to-end preprocessing utility:
  - CLAHE preprocessing, overlap tiling, patch-map generation, optional auto-labeling sidecars, YOLO dataset assembly helpers.
- `train.py`
  - MicroSAM training wrapper around `train_sam`.
  - Reads `images/` + `labels/`, creates train/val split, writes checkpoints to `microsam/models/`.
- `inference.py`
  - Runs model inference for one input image and writes an instance label map.
- `migrate_legacy.py`
  - One-time migration helper for older labeling layouts.
- `xanylabel.sh`
  - Environment bootstrap and launcher for X-AnyLabeling.
- `activate_venv.sh`
  - Activation helper for labeling virtual environment.

### `bubbly_flows/tests/`
Research and validation sandbox for detection/segmentation experiments.

This folder is mixed-purpose:
- Experimental scripts (`classical_test.py`, `detect_bubbles.py`, `blackhat_mask.py`, etc.)
- Composite FRST + SAM3 pipeline (`bubble_frst_sam3_mask.py`)
- Modular SAM3 package (`bubble_sam3/`) containing:
  - `backend.py` (SAM3 model adapters)
  - `pipeline.py` (tiling + prompting orchestration)
  - `candidates.py`, `preprocess.py`, `postprocess.py`, `outputs.py`, `config.py`
- Sample inputs and generated outputs (`img*.png/json`, `output/`, `logs/`)

Treat `tests/` as an experimentation area, not a strict unit-test suite.

### `bubbly_flows/logs/`
Cluster training submission artifacts.

- `submit_*.sh` files: generated Slurm job scripts
- `*.out`, `*.err`: runtime logs per submitted training job

### `bubbly_flows/archive/`
Historical scripts/assets kept for reference (non-active pipeline components).

## 4) Root-Level Runtime Helpers

- `environment.yml`
  - Conda environment (`bubbly-train-env`) used by the training/management pipeline.
- `launch_labeling.command`, `launch_labeling.sh`, `launch_labeling.bat`, `launch_labeling.desktop`
  - Double-click launchers that dispatch to `bubbly_flows/scripts/xanylabel.sh`.
- `x-labeling-env/`
  - Python virtual environment used by X-AnyLabeling launcher.
- `output/`
  - Root-level ad-hoc output samples (separate from `bubbly_flows/tests/output/`).

## 5) Naming and Data Conventions

- Patch names usually preserve frame identity + tile origin:
  - `<frame_stem>__x####_y####.<ext>`
- Workspace manifests list selected patch filenames (`filename` header).
- Gold manifests list promoted label stems (`filename_stem` header).
- MicroSAM datasets require paired directories:
  - `images/` (patches)
  - `labels/` (instance masks where pixel values are instance IDs)

## 6) What Is Stable vs. Ephemeral

Usually stable (versioned data/code):
- `bubbly_flows/scripts/`
- `bubbly_flows/annotations/gold/`
- `bubbly_flows/data/` (as canonical source pool)
- `bubbly_flows/microsam/datasets/` and selected models

Usually ephemeral/generated during runs:
- `bubbly_flows/workspaces/<workspace>/`
- `bubbly_flows/logs/*.out`, `*.err`
- `bubbly_flows/tests/output/`, `bubbly_flows/tests/logs/`
- root `output/`

## 7) Where To Start as a New Contributor

1. Read `USER_GUIDE.md` for operations.
2. Read `bubbly_flows/scripts/manage_bubbly.py` to understand the canonical workflow.
3. Review this data lineage: `data -> workspaces -> annotations/gold -> microsam/datasets -> microsam/models`.
4. Only then dive into `bubbly_flows/tests/` for experimental methods (FRST/SAM3 variants).

