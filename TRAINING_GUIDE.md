# Bubble Tracking — Intern Training Guide
## From Zero to Hero: Training Models and Assessing Performance

> **Audience**: New contributor who will work on training MicroSAM models and building evaluation pipelines.
> **Platform**: Oscar HPC cluster (Brown University).
> **Assumed starting point**: Gold annotations already exist in `bubbly_flows/annotations/gold/`.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Repository Orientation](#2-repository-orientation)
3. [Environment Setup on Oscar](#3-environment-setup-on-oscar)
4. [Understanding the Data](#4-understanding-the-data)
5. [Step 1 — Export a Training Dataset](#5-step-1--export-a-training-dataset)
6. [Step 2 — Train a Model](#6-step-2--train-a-model)
7. [Step 3 — Monitor Training](#7-step-3--monitor-training)
8. [Step 4 — Run Inference](#8-step-4--run-inference)
9. [Step 5 — Build an Evaluation Pipeline (Your Task)](#9-step-5--build-an-evaluation-pipeline-your-task)
10. [Quick Reference](#10-quick-reference)

---

## 1. What Is This Project?

This project builds a machine-learning pipeline to automatically detect and segment **bubbles** in images taken during zero-gravity flight experiments. Think of it as training a model to answer: *"Where are all the bubbles in this image, and what are their exact shapes?"*

The core model is **MicroSAM**, a fine-tuned version of Meta's Segment Anything Model (SAM) adapted for microscopy and scientific imaging. Fine-tuning means we take a powerful general-purpose model and specialize it on our bubble images using human-labeled training data.

**Your role** as the new intern is to:
- Convert existing gold-standard annotations into training datasets.
- Submit and manage training jobs on the Oscar cluster.
- Design and implement an evaluation pipeline to measure how well the trained models perform.

---

## 2. Repository Orientation

Everything lives inside the `bubbly_flows/` subdirectory. Here is what matters for your work:

```
Bubble-tracking/
├── environment.yml                    # Conda environment spec (your dependencies)
├── bubbly_flows/
│   ├── scripts/
│   │   ├── manage_bubbly.py           # ← Main controller, your primary entry point
│   │   ├── train.py                   # ← Training script (called by Slurm)
│   │   └── inference.py               # ← Inference script (run on a single image)
│   ├── annotations/
│   │   └── gold/                      # ← Human-labeled ground truth (read-only)
│   │       ├── gold_seed_v00/
│   │       │   ├── labels_json/       # LabelMe JSON polygon files
│   │       │   └── manifest.csv       # List of labeled image stems
│   │       └── gold_v00/ ...
│   ├── microsam/
│   │   ├── datasets/                  # ← Exported training datasets (images + masks)
│   │   └── models/                    # ← Trained checkpoints
│   ├── data/
│   │   └── patches_pool/images/       # Source image patches
│   └── logs/                          # Slurm job scripts and output logs
```

**Data lineage** (what feeds what):
```
annotations/gold/  →  microsam/datasets/  →  microsam/models/
   (JSON labels)       (images + TIF masks)    (checkpoints)
```

---

## 3. Environment Setup on Oscar

### 3.1 — Log into Oscar

```bash
ssh <your-username>@ssh.ccv.brown.edu
```

Then navigate to the repository:

```bash
cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
```

### 3.2 — Load the Anaconda Module

Oscar does not have conda available by default. You must load it first every session (or add it to your `~/.bashrc`):

```bash
module load miniforge3
```

> **Tip**: Add `module load miniforge3` to your `~/.bashrc` so you never forget.

### 3.3 — Create the Conda Environment (First Time Only)

The repository manages its own Python environment called `bubbly-train-env`. Create it once:

```bash
conda env create -f environment.yml
```

This takes 5–10 minutes. It installs PyTorch, MicroSAM, torch_em, OpenCV, and all other dependencies.

### 3.4 — Activate the Environment

```bash
conda activate bubbly-train-env
```

You should see `(bubbly-train-env)` in your terminal prompt. You will need to do this every time you log in.

### 3.5 — Verify the Setup

```bash
python -c "import torch; import micro_sam; print('OK', torch.__version__)"
```

If this prints `OK` and a version number, you are ready.

---

## 4. Understanding the Data

### 4.1 — Gold Annotations

Gold annotations are the source of truth for supervised training. Each gold version is a snapshot:

```
bubbly_flows/annotations/gold/gold_seed_v00/
├── labels_json/          # One JSON file per labeled image
│   ├── SomeImage__x0320_y0640.json
│   └── ...
├── manifest.csv          # Tracks which images have been labeled
└── stats.json            # Simple counts
```

Each JSON file is a LabelMe-format polygon annotation. Open one to see its structure:

```bash
python -c "
import json
import glob
jf = sorted(glob.glob('bubbly_flows/annotations/gold/*/labels_json/*.json'))[0]
with open(jf) as f:
    d = json.load(f)
print('Image:', d['imagePath'])
print('Shapes:', len(d['shapes']), 'bubbles annotated')
print('First shape keys:', list(d['shapes'][0].keys()) if d['shapes'] else 'none')
"
```

Each `shape` in the JSON is one bubble, represented as a polygon (list of `[x, y]` points).

### 4.2 — Training Dataset Format

When you export a dataset (Step 1), the pipeline converts polygon JSONs into **instance segmentation masks**:

- `microsam/datasets/<name>/images/` — the image patches (PNG/TIF)
- `microsam/datasets/<name>/labels/` — paired TIF masks where each pixel's value is the instance ID (0 = background, 1 = first bubble, 2 = second bubble, ...)

This is the format that `train.py` and `torch_em` expect.

---

## 5. Step 1 — Export a Training Dataset

Run the interactive controller:

```bash
conda activate bubbly-train-env
python bubbly_flows/scripts/manage_bubbly.py
```

You will see a menu:

```
========================================
   BUBBLY FLOWS - DATASET MANAGER
========================================
1. Initialize / Update Patch Pool
2. Create New Workspace
3. Promote Workspace to Gold (+ Cleanup)
4. Prepare MicroSAM Dataset (Export)     ← This is what you want
5. Train Model (submit job)
6. Run Inference
q. Quit
```

Select **option 4**. The script will:

1. List available gold versions — pick the one you want (e.g. `gold_seed_v00`).
2. Ask for a dataset name — use something descriptive like `v01_seed` or `v01_seed_aug`.
3. Ask whether to enable augmentation — say **y** (recommended). This multiplies your training data 4× using geometric transforms, photometric jitter, and copy-paste augmentation.
4. Ask for an augmentation seed — use `42` for reproducibility.

The output will be written to `bubbly_flows/microsam/datasets/<your-dataset-name>/`.

**Verify the export:**

```bash
ls bubbly_flows/microsam/datasets/<your-dataset-name>/images/ | wc -l
ls bubbly_flows/microsam/datasets/<your-dataset-name>/labels/ | wc -l
```

Both counts should be equal. With augmentation enabled, expect roughly 4× the number of source JSON files.

---

## 6. Step 2 — Train a Model

From the same menu, select **option 5**. The script will:

1. List available datasets — pick the one you just exported.
2. Ask for an experiment name — use something like `train_v01_seed_run1`.
3. Ask for a time limit in hours — `4` is a good starting value for a first run.

The script generates a Slurm job script at `bubbly_flows/logs/submit_<exp_name>.sh` and asks if you want to submit it. Say **y**.

### What Happens on the Cluster

The Slurm job will:
1. Load `miniforge3` and `cuda/11.8`.
2. Activate `bubbly-train-env`.
3. Run `train.py` for 100 epochs with a `vit_b` (ViT-Base) SAM backbone.
4. Save checkpoints to `bubbly_flows/microsam/models/<exp_name>/checkpoints/<exp_name>/`.

The best checkpoint is saved as `best.pt`.

### Checking Job Status

```bash
squeue -u <your-username>
```

### Cancelling a Job

```bash
scancel <job_id>
```

---

## 7. Step 3 — Monitor Training

Training logs are written to `bubbly_flows/logs/`. Two files are created per job:

- `<exp_name>_<job_id>.out` — standard output (training progress, loss values)
- `<exp_name>_<job_id>.err` — standard error (warnings, errors)

Watch a live log:

```bash
tail -f bubbly_flows/logs/<exp_name>_<job_id>.out
```

**What to look for:**

| What you see | What it means |
|---|---|
| `Using device: cuda` | GPU is active — good |
| `Using device: cpu` | No GPU allocated — training will be extremely slow |
| Loss decreasing over epochs | Model is learning |
| `TRAINING COMPLETE` | Job finished successfully |
| Any traceback in `.err` | Something went wrong — read the error |

**Common errors:**

- `CUDA driver version is insufficient` → The GPU partition was not requested. Check the `.sh` script and make sure `#SBATCH -p gpu` is present.
- `Module not found: torch` → The conda environment was not activated correctly in the job script. Re-examine the generated `.sh` file.

---

## 8. Step 4 — Run Inference

After training completes, you can run the model on a new image.

From the menu, select **option 6**:

1. Select the trained experiment.
2. Enter the path to an input image.
3. Enter the desired output path for the mask (e.g. `output/test_mask.png`).

Or call `inference.py` directly:

```bash
python bubbly_flows/scripts/inference.py \
    --model_path bubbly_flows/microsam/models/<exp_name>/checkpoints/<exp_name>/best.pt \
    --image bubbly_flows/data/patches_pool/images/<some_image>.png \
    --output output/<some_image>_pred.png \
    --model_type vit_b
```

The output is a 16-bit TIF or PNG where each pixel value is the predicted instance ID (0 = background, 1 = first predicted bubble, etc.). This is the same format as the gold label masks in `microsam/datasets/*/labels/`.

---

## 9. Step 5 — Build an Evaluation Pipeline (Your Task)

There is no evaluation pipeline yet — building it is your primary research task. This section gives you the conceptual framework and a concrete starting point.

### 9.1 — What You Have

After inference on a test image you have two things to compare:

| | Path | Format |
|---|---|---|
| **Prediction** | wherever you saved it | 16-bit TIF, pixel = predicted instance ID |
| **Ground truth** | `microsam/datasets/<ds>/labels/<stem>.tif` | 16-bit TIF, pixel = gold instance ID |

The key challenge: instance IDs in prediction and ground truth are **arbitrary**. Instance "3" in the prediction might correspond to instance "7" in the ground truth. You must match them before computing any metric.

### 9.2 — Standard Metrics for Instance Segmentation

**Intersection over Union (IoU)** — measures shape overlap between one predicted mask and one ground-truth mask:

```
IoU = |A ∩ B| / |A ∪ B|
```

**Average Precision (AP)** — the standard benchmark metric. At a given IoU threshold (e.g. 0.5), a predicted instance is a True Positive if it matches a ground-truth instance with IoU ≥ threshold. AP summarizes precision across all recall levels. AP@0.5 and AP@0.75 are the most common values to report.

**F1 / Dice score** — harmonic mean of precision and recall at a fixed IoU threshold. Easier to interpret than AP for a first pass.

**Panoptic Quality (PQ)** — combines detection quality and segmentation quality into a single number. More advanced but worth knowing about.

### 9.3 — Matching Instances: the Hungarian Algorithm

To match predicted instances to ground-truth instances, compute the IoU between every pair and solve the assignment problem:

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

def match_instances(pred_mask, gt_mask, iou_threshold=0.5):
    """Match predicted instances to ground truth using Hungarian matching."""
    pred_ids = [i for i in np.unique(pred_mask) if i > 0]
    gt_ids   = [i for i in np.unique(gt_mask)   if i > 0]

    if not pred_ids or not gt_ids:
        return [], [], pred_ids, gt_ids  # all FP or all FN

    # Build IoU matrix: rows = predictions, cols = ground truth
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for i, pid in enumerate(pred_ids):
        for j, gid in enumerate(gt_ids):
            inter = np.logical_and(pred_mask == pid, gt_mask == gid).sum()
            union = np.logical_or( pred_mask == pid, gt_mask == gid).sum()
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    # Solve assignment (minimize cost = maximize IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches, fp_ids, fn_ids = [], list(pred_ids), list(gt_ids)
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((pred_ids[r], gt_ids[c], iou_matrix[r, c]))
            fp_ids.remove(pred_ids[r])
            fn_ids.remove(gt_ids[c])

    return matches, iou_matrix, fp_ids, fn_ids
```

### 9.4 — Computing AP@0.5

Once you have matches:

```python
def compute_metrics(matches, fp_ids, fn_ids):
    TP = len(matches)
    FP = len(fp_ids)
    FN = len(fn_ids)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou  = np.mean([m[2] for m in matches]) if matches else 0.0

    return {"TP": TP, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall,
            "F1": f1, "mean_IoU": mean_iou}
```

### 9.5 — Suggested Evaluation Workflow

1. **Split your dataset into train / test before exporting** — or use a held-out gold version that was never included in training. This is critical: never evaluate on training data.
2. **Run inference on all test images** using the trained checkpoint.
3. **Load each (prediction, ground-truth) pair** and compute metrics using the functions above.
4. **Aggregate** metrics across all test images (mean F1, mean AP@0.5, etc.).
5. **Visualize** — overlay predicted contours on images to spot failure modes (missed bubbles, false positives, shape errors).

### 9.6 — Suggested Script Location

Create your evaluation script at:

```
bubbly_flows/scripts/evaluate.py
```

A minimal skeleton:

```python
#!/usr/bin/env python3
"""
evaluate.py

Compute instance segmentation metrics between a predicted mask directory
and a ground-truth mask directory.

Usage:
    python bubbly_flows/scripts/evaluate.py \
        --preds output/preds/ \
        --gts   bubbly_flows/microsam/datasets/v01_seed/labels/ \
        --iou_threshold 0.5
"""
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import linear_sum_assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, type=Path)
    parser.add_argument("--gts",   required=True, type=Path)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred_files = sorted(args.preds.glob("*.tif")) + sorted(args.preds.glob("*.png"))
    results = []

    for pred_path in pred_files:
        gt_path = args.gts / pred_path.name
        # Try .tif extension if .png not found
        if not gt_path.exists():
            gt_path = gt_path.with_suffix(".tif")
        if not gt_path.exists():
            print(f"  [skip] no GT found for {pred_path.name}")
            continue

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        gt_mask   = cv2.imread(str(gt_path),   cv2.IMREAD_UNCHANGED)

        if pred_mask is None or gt_mask is None:
            print(f"  [error] could not load {pred_path.name}")
            continue

        # TODO: call match_instances() and compute_metrics() here
        print(f"  {pred_path.name}: loaded pred {pred_mask.shape}, gt {gt_mask.shape}")
        results.append(pred_path.name)

    print(f"\nProcessed {len(results)} image pairs.")


if __name__ == "__main__":
    main()
```

Fill in the matching and metric logic from Section 9.3–9.4 and expand from there.

---

## 10. Quick Reference

### Key Commands

| Action | Command |
|---|---|
| Load conda | `module load miniforge3` |
| Activate env | `conda activate bubbly-train-env` |
| Run manager | `python bubbly_flows/scripts/manage_bubbly.py` |
| Check jobs | `squeue -u <your-username>` |
| Watch log | `tail -f bubbly_flows/logs/<job>.out` |
| Cancel job | `scancel <job_id>` |
| Run inference | `python bubbly_flows/scripts/inference.py --model_path ... --image ... --output ...` |

### Key File Paths

| What | Where |
|---|---|
| Gold labels | `bubbly_flows/annotations/gold/<version>/labels_json/` |
| Exported datasets | `bubbly_flows/microsam/datasets/<name>/` |
| Model checkpoints | `bubbly_flows/microsam/models/<exp>/checkpoints/<exp>/best.pt` |
| Slurm logs | `bubbly_flows/logs/<exp>_<job_id>.out` |
| Diary log | `bubbly_flows/diary.log` |

### Pipeline Summary

```
Gold JSON annotations
        ↓  manage_bubbly.py  Option 4
microsam/datasets/<name>/   (images + TIF masks)
        ↓  manage_bubbly.py  Option 5  (Slurm job)
microsam/models/<exp>/checkpoints/<exp>/best.pt
        ↓  inference.py
Predicted instance masks
        ↓  evaluate.py  (your task to build)
Metrics: F1, AP@0.5, mean IoU
```

---

> **Where to go when stuck**: Read `README.md` for project structure, `USER_GUIDE.md` for the annotation workflow, and `bubbly_flows/diary.log` for a chronological record of all past pipeline actions. When in doubt, read the source of `manage_bubbly.py` — it is well commented and is the single source of truth for how data flows through the system.
