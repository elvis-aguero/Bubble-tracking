# Pipeline Restructure Design
**Date:** 2026-03-09
**Status:** Approved

## Goal

Make the end-to-end ML pipeline — from annotated frames to trained model to evaluation —
runnable by a new user entirely through `manage_bubbly.py`, with no ambiguity and no
runtime engineering decisions. Annotation is out of scope (handled separately in X-AnyLabeling).

## Out of Scope

- `tests/` directory (hybrid SAM3 + classical experiments): left untouched.
- Annotation workflow.

---

## Section 1 — Repository Structure

```
Bubble-tracking/
├── configs/                        # Canonical hyperparameter configs (maintained by engineer)
│   ├── microsam.json
│   ├── stardist.json
│   └── yolov9.json
├── bubbly_flows/
│   ├── annotations/gold/           # Promoted gold annotation sets
│   ├── data/frames/                # Raw image frames
│   ├── pipeline/                   # Renamed from microsam/ (model-agnostic)
│   │   └── datasets/               # Exported train/test splits
│   ├── scripts/
│   │   ├── manage_bubbly.py
│   │   ├── train.py
│   │   ├── train_stardist.py
│   │   ├── train_yolov9.py
│   │   ├── inference.py
│   │   └── evaluate.py
│   ├── workspaces/
│   └── logs/
├── tests/                          # Untouched hybrid experiments
├── docs/plans/                     # Design documents
└── .gitignore                      # x-labeling-env/ added
```

**Key structural changes:**
- `bubbly_flows/microsam/` → `bubbly_flows/pipeline/` (neutral name for multi-model pipeline)
- `x-labeling-env/` added to `.gitignore` (virtual env should never be committed)
- `docs/plans/` created for design documents

**Trained run directory** (in scratch, not in repo):
```
~/scratch/bubble-models/trained/<run_name>/
    checkpoints/<run_name>/best.pt
    config.json          # Frozen copy of configs/<model>.json at submission time
    logs/
```

The `config.json` in each run directory is the permanent provenance record — it captures
exactly what hyperparameters produced that checkpoint.

---

## Section 2 — Config JSON Format

Each `configs/<model>.json` contains model identity, training hyperparameters, inference
hyperparameters, and an engineering notes field. Example for MicroSAM:

```json
{
  "model": "microsam",
  "backbone": "vit_b",
  "training": {
    "patch_shape": 1024,
    "epochs": 100,
    "batch_size": 1,
    "num_workers": 4,
    "freeze": ["image_encoder"],
    "early_stopping_patience": 10,
    "val_fraction": 0.15
  },
  "inference": {
    "model_type": "vit_b",
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95
  },
  "notes": "Full-resolution 1024px crops. Encoder frozen; only UNETR decoder trains."
}
```

**Rules:**
- The engineer (Claude) owns and maintains these files. Values are not debated at runtime.
- To iterate on hyperparameters: edit the JSON, commit the change, resubmit. Git history
  is the experiment log.
- Training scripts accept `--config configs/<model>.json` instead of individual flags.
  All other hyperparameter flags (`--patch_shape`, `--epochs`, etc.) are removed.
- At job submission, `manage_bubbly.py` copies the config JSON into the run directory
  before `sbatch` is called.

---

## Section 3 — Menu Redesign

The main menu exposes only the happy-path pipeline steps, in order. Advanced options
are in a submenu. Every option — at every level — has a one-line tooltip.

### Main menu

```
╔══════════════════════════════════════════════════╗
║           Bubble Tracking Pipeline               ║
╚══════════════════════════════════════════════════╝
  State: gold=seed_v04  dataset=seed_v04_train/test  last_run=microsam_1024_run1

  1. Promote Workspace to Gold     — finalise annotations, create train/test split
  2. Train Model                   — submit Slurm job using configs/<model>.json
  3. Evaluate on Test Set          — run inference + metrics on held-out split
  4. Inference on Image            — run a trained model on any single image
  ──────────────────────────────────────────────────
  a. Advanced                      — pool management, workspace creation, dataset export
  q. Quit
```

The **state line** is computed at startup by scanning `annotations/gold/`,
`pipeline/datasets/`, and `~/scratch/bubble-models/trained/`. No separate state file.

### Train submenu (Option 2)

```
  Train Model
  ─────────────────────────────────────────────────
  Select model:
  1. MicroSAM (ViT-B + UNETR)  — best for dense instance segmentation, needs GPU
  2. StarDist 2D               — fast, good for convex bubble shapes
  3. YOLOv9c-seg               — good generalisation, real-time capable

  Config: configs/microsam.json  [patch=1024, epochs=100, freeze=encoder]

  Select dataset:
  1. seed_v04_train  (44 images)
  ...
```

### Advanced submenu

```
  Advanced Options
  ─────────────────────────────────────────────────
  1. Update Patch Pool      — scan frames dir for new images, rebuild pool index
  2. Create Workspace       — start a new annotation seed from the pool
  3. Export Dataset         — re-run train/test split on an existing gold set
  q. Back
```

---

## Section 4 — Data Flow and Error Handling

### Data flow

```
annotations/gold/seed_v04/
        ↓  promote (Option 1)
pipeline/datasets/seed_v04_train/   80% split — images + uint16 label TIFs
pipeline/datasets/seed_v04_test/    20% split — held out, never seen during training
        ↓  train (Option 2)
~/scratch/bubble-models/trained/<run>/
    checkpoints/<run>/best.pt
    config.json
        ↓  evaluate (Option 3)
tests/output/eval_preds/<run>/
    <image>.png          label maps
    results.csv          per-image TP/FP/FN/F1/IoU + macro/micro summary
        ↓  inference (Option 4)
<user output path>/
    <stem>.png           uint16 label map
    <stem>_vis.png       original image + coloured instance overlay
```

### Prerequisite checks

Menu options block with a clear message if prerequisites are missing:

| Option | Blocks if |
|---|---|
| Train | No `pipeline/datasets/*_train` found |
| Evaluate | No trained checkpoint found in scratch |
| Inference | No trained checkpoint found in scratch |

Block message format: `"No training dataset found. Run Option 1 first."`

### Error handling

- **Slurm submission:** on success, prints job ID and exact log path to tail.
  On failure, prints sbatch stderr — no silent swallowing.
- **Per-image inference failure:** logged as a warning, pipeline continues to next image.
  An evaluate run does not abort because one image fails.
- **Missing GT mask in evaluate:** `[skip]` with filename — existing behaviour, kept.
- **Config missing or malformed:** hard stop with message pointing to `configs/`
  before any job is submitted.
