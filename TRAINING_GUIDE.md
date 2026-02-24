# Bubble Tracking — Getting Started Guide
## Training Models and Assessing Performance

> **Who this is for**: Anyone joining the team who will work on training segmentation models and building evaluation pipelines.
> **Platform**: Oscar HPC cluster (Brown University).
> **Starting point**: Human-labeled gold annotations already exist in `bubbly_flows/annotations/gold/`.

---

## Table of Contents

0. [Prerequisites — Read These First](#0-prerequisites--read-these-first)
1. [What Is This Project?](#1-what-is-this-project)
2. [How the Repository Is Organized](#2-how-the-repository-is-organized)
3. [Environment Setup on Oscar](#3-environment-setup-on-oscar)
4. [Understanding the Data](#4-understanding-the-data)
5. [Step 1 — Export a Training Dataset](#5-step-1--export-a-training-dataset)
6. [Step 2 — Train a Model](#6-step-2--train-a-model)
7. [Step 3 — Monitor Training](#7-step-3--monitor-training)
8. [Step 4 — Run Inference](#8-step-4--run-inference)
9. [Step 5 — Build an Evaluation Pipeline](#9-step-5--build-an-evaluation-pipeline)
10. [Quick Reference](#10-quick-reference)

---

## 0. Prerequisites — Read These First

Before diving in, it helps to be comfortable with a few tools. Below are useful references — do not worry about memorizing everything, just bookmark them and come back when needed.

**Oscar Cluster (Brown's HPC system)**
- [Connecting to Oscar](https://docs.ccv.brown.edu/oscar/connecting-to-oscar) — how to log in via SSH
- [Oscar overview and getting started](https://docs.ccv.brown.edu/oscar/getting-started)
- [Submitting batch jobs with Slurm](https://docs.ccv.brown.edu/oscar/submitting-jobs/batch) — how to submit and manage compute jobs
- [Oscar storage and file systems](https://docs.ccv.brown.edu/oscar/managing-files/filesystem)

**Linux command line**
- [Linux command line cheat sheet (Cheatography)](https://cheatography.com/davechild/cheat-sheets/linux-command-line/) — the basics: navigating folders, moving files, viewing files
- [The Linux command line for beginners (Ubuntu)](https://ubuntu.com/tutorials/command-line-for-beginners) — a gentle walkthrough if you have never used a terminal before

**Conda (Python environment manager)**
- [Conda cheat sheet (official)](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html) — the commands you will actually use day to day

**Git (version control)**
- [Git basics cheat sheet (GitHub)](https://education.github.com/git-cheat-sheet-education.pdf) — useful for pulling updates and tracking your work

**Segment Anything Model (SAM) — the underlying AI**
- [SAM paper (Meta AI)](https://ai.meta.com/research/publications/segment-anything/) — what SAM is and how it works
- [MicroSAM documentation](https://computational-cell-analytics.github.io/micro-sam/) — the scientific imaging adaptation we build on

---

## 1. What Is This Project?

This project builds a machine-learning pipeline to automatically detect and segment **bubbles** in images taken during zero-gravity flight experiments. The core question the model answers is: *"Where are all the bubbles in this image, and what are their exact shapes?"*

We approach this by **fine-tuning** existing models on our own labeled bubble images. Fine-tuning means taking a powerful model that was trained on general images and re-training it on our specific data so it becomes an expert at bubbles. We work with several model families:

- **MicroSAM** — a version of Meta's Segment Anything Model adapted for scientific imaging. Currently our main training target.
- **CNN-based models** (convolutional neural networks) — classical segmentation architectures that we also benchmark.
- Other architectures may be added over time as the project evolves.

The pipeline covered in this guide handles everything from converting human annotations into a training-ready format, running the training job on the cluster, and eventually measuring how well the trained models perform.

---

## 2. How the Repository Is Organized

Everything lives inside the `bubbly_flows/` folder. Here is what matters for this workflow:

```
Bubble-tracking/
├── environment.yml                    # List of all Python packages needed
├── bubbly_flows/
│   ├── scripts/
│   │   ├── manage_bubbly.py           # ← The main script you will run
│   │   ├── train.py                   # ← Training script (launched automatically by the cluster)
│   │   └── inference.py               # ← Run a trained model on a new image
│   ├── annotations/
│   │   └── gold/                      # ← Human-labeled ground truth (do not edit)
│   │       ├── gold_seed_v00/
│   │       │   ├── labels_json/       # One annotation file per image
│   │       │   └── manifest.csv       # List of which images were labeled
│   │       └── gold_v00/ ...
│   ├── microsam/
│   │   ├── datasets/                  # ← Training-ready datasets (images + masks)
│   │   └── models/                    # ← Saved model weights after training
│   ├── data/
│   │   └── patches_pool/images/       # Source image patches
│   └── logs/                          # Job scripts and training output logs
```

**How data flows through the project:**
```
annotations/gold/  →  microsam/datasets/  →  microsam/models/
 (annotation files)    (images + masks)       (trained model weights)
```

---

## 3. Environment Setup on Oscar

### 3.1 — Log into Oscar

```bash
ssh <your-username>@ssh.ccv.brown.edu
```

Then navigate to the project folder:

```bash
cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
```

### 3.2 — Load the Package Manager

Oscar does not have conda available by default. You need to load it at the start of every session:

```bash
module load miniforge3
```

> **Tip**: Add `module load miniforge3` to your `~/.bashrc` file so it loads automatically every time you log in. If you are not sure how to do this, ask a teammate.

### 3.3 — Create the Project Environment (First Time Only)

The project uses a dedicated Python environment called `bubbly-train-env` to keep all its packages separate from everything else on the system. Create it once:

```bash
conda env create -f environment.yml
```

This takes 5–10 minutes. It installs PyTorch, MicroSAM, and everything else the project needs.

### 3.4 — Activate the Environment

```bash
conda activate bubbly-train-env
```

You should see `(bubbly-train-env)` appear in your terminal prompt. You need to do this every time you log in.

### 3.5 — Verify Everything Installed Correctly

```bash
python -c "import torch; import micro_sam; print('All good!', torch.__version__)"
```

If this prints `All good!` followed by a version number, you are ready to go.

---

## 4. Understanding the Data

### 4.1 — Gold Annotations

Gold annotations are the human-labeled ground truth that the models learn from. Each "gold version" is a saved snapshot of the labels at a point in time:

```
bubbly_flows/annotations/gold/gold_seed_v00/
├── labels_json/          # One file per labeled image
│   ├── SomeImage__x0320_y0640.json
│   └── ...
├── manifest.csv          # Which images are in this version
└── stats.json            # How many labels, which workspace they came from
```

Each `.json` file describes where the bubbles are in one image, as a set of polygons — one polygon per bubble. You can peek inside one:

```bash
python -c "
import json, glob
jf = sorted(glob.glob('bubbly_flows/annotations/gold/*/labels_json/*.json'))[0]
with open(jf) as f:
    d = json.load(f)
print('Image:', d['imagePath'])
print('Bubbles labeled:', len(d['shapes']))
"
```

### 4.2 — Training Dataset Format

When you export a dataset (Step 1 below), the pipeline reads those polygon files and converts them into **pixel masks** that the training code understands:

- `microsam/datasets/<name>/images/` — the image patches
- `microsam/datasets/<name>/labels/` — paired mask files where each pixel's value is a bubble ID (0 = background, 1 = first bubble, 2 = second bubble, ...)

Each image file and its corresponding mask file share the same name, just with a `.tif` extension for the mask.

---

## 5. Step 1 — Export a Training Dataset

Run the main management script:

```bash
conda activate bubbly-train-env
python bubbly_flows/scripts/manage_bubbly.py
```

A menu will appear:

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

Select **option 4**. The script will walk you through:

1. **Choose a gold version** — pick the one you want to train from (e.g. `gold_seed_v00`).
2. **Name the dataset** — use something descriptive like `v01_seed` or `v01_seed_aug`.
3. **Enable augmentation?** — say **y** (recommended). This automatically creates extra training variations of each image using flips, rotations, brightness changes, and copy-paste. You end up with roughly 4× more training examples at no extra labeling cost.
4. **Augmentation seed** — type `42`. This number makes the augmentation reproducible — using the same seed will always produce the same result.

The exported dataset will be written to `bubbly_flows/microsam/datasets/<your-dataset-name>/`.

**Check the export worked:**

```bash
ls bubbly_flows/microsam/datasets/<your-dataset-name>/images/ | wc -l
ls bubbly_flows/microsam/datasets/<your-dataset-name>/labels/ | wc -l
```

Both numbers should match. With augmentation on, expect roughly 4× the number of original annotation files.

---

## 6. Step 2 — Train a Model

From the same menu, select **option 5**. The script will ask:

1. **Which dataset?** — pick the one you just exported.
2. **Experiment name** — something like `train_v01_seed_run1`. Pick a name that will help you remember what this run was.
3. **Time limit in hours** — `4` is a reasonable starting point for a first run.

The script writes a job file to `bubbly_flows/logs/` and asks if you want to submit it. Say **y**.

### What Happens on the Cluster

The cluster job will:
1. Allocate a GPU on Oscar.
2. Activate the `bubbly-train-env` environment.
3. Run `train.py` for 100 passes through the training data ("epochs"), using a MicroSAM ViT-B model as the starting point.
4. Save the best-performing model weights to `bubbly_flows/microsam/models/<your-experiment-name>/checkpoints/<your-experiment-name>/best.pt`.

### Checking Whether Your Job Is Running

```bash
squeue -u <your-username>
```

This shows all your active jobs. Look for the job ID in the leftmost column — you will need it for the log files.

### Cancelling a Job

```bash
scancel <job_id>
```

---

## 7. Step 3 — Monitor Training

Training logs are saved in `bubbly_flows/logs/`. Two files are created per job:

- `<exp_name>_<job_id>.out` — the main output: training progress, loss values per epoch
- `<exp_name>_<job_id>.err` — warnings and errors

Watch the log update in real time:

```bash
tail -f bubbly_flows/logs/<exp_name>_<job_id>.out
```

Press `Ctrl+C` to stop watching.

**What to look for:**

| What you see | What it means |
|---|---|
| `Using device: cuda` | GPU is active — good |
| `Using device: cpu` | No GPU was allocated — training will be impractically slow |
| Loss numbers decreasing over epochs | The model is learning |
| `TRAINING COMPLETE` | The job finished successfully |
| Any error message in the `.err` file | Something went wrong — read the error carefully |

**Common problems:**

- `CUDA driver version is insufficient` → The job was not allocated a GPU. Check that the generated job file under `bubbly_flows/logs/submit_*.sh` contains `#SBATCH -p gpu`.
- `Module not found: torch` → The environment was not activated correctly inside the job. Look at the generated `.sh` file and confirm the `conda activate bubbly-train-env` line is present.

---

## 8. Step 4 — Run Inference

Once training is done, you can run the model on a new image to see what it predicts.

From the menu, select **option 6** and follow the prompts:

1. Pick the trained experiment.
2. Give the path to an input image.
3. Give a path where the output mask should be saved (e.g. `output/test_mask.png`).

Or run `inference.py` directly from the terminal:

```bash
python bubbly_flows/scripts/inference.py \
    --model_path bubbly_flows/microsam/models/<exp_name>/checkpoints/<exp_name>/best.pt \
    --image bubbly_flows/data/patches_pool/images/<some_image>.png \
    --output output/<some_image>_pred.png \
    --model_type vit_b
```

The output is a mask image where each pixel's value is the predicted bubble ID (0 = background, 1 = first bubble, 2 = second bubble, ...). This is the same format as the gold masks in `microsam/datasets/*/labels/`, which makes them easy to compare.

---

## 9. Step 5 — Build an Evaluation Pipeline

There is no evaluation pipeline yet — building one is a core open task for the team. This section explains the problem, the standard approach, and gives starter code to build from.

### 9.1 — What You Have to Work With

After running inference on a test image you have two mask files to compare:

| | Path | Format |
|---|---|---|
| **Prediction** | wherever you saved it | mask file, pixel value = predicted bubble ID |
| **Ground truth** | `microsam/datasets/<ds>/labels/<stem>.tif` | mask file, pixel value = labeled bubble ID |

There is one important catch: bubble IDs in the prediction and ground truth are **not aligned**. Bubble "3" in the prediction might be the same physical bubble as "7" in the ground truth. You have to figure out which predicted bubble matches which labeled bubble before you can compute any score — this is called the **matching problem**.

### 9.2 — Standard Metrics for Instance Segmentation

**Intersection over Union (IoU)** — measures how much two masks overlap. A perfect match scores 1.0; no overlap scores 0.0:

```
IoU = (pixels in both masks) / (pixels in either mask)
```

**F1 score** — the balance between finding all real bubbles (recall) and not crying wolf on non-bubbles (precision). A good first metric to track.

**Average Precision (AP)** — the standard rigorous benchmark. At IoU ≥ 0.5, a predicted bubble counts as correct if it overlaps enough with a labeled bubble. AP@0.5 and AP@0.75 (stricter) are the values most papers report.

**Panoptic Quality (PQ)** — a single number combining how many bubbles were found and how well their shapes were drawn. More advanced, worth reading about once the basics are working.

### 9.3 — Solving the Matching Problem

The standard solution is the **Hungarian algorithm** — it finds the best one-to-one assignment between predicted and labeled bubbles by maximizing total overlap:

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

def match_instances(pred_mask, gt_mask, iou_threshold=0.5):
    """
    Match predicted bubbles to labeled bubbles.
    Returns:
        matches  - list of (pred_id, gt_id, iou) for correct detections
        fp_ids   - predicted bubbles with no matching label (false positives)
        fn_ids   - labeled bubbles with no matching prediction (false negatives)
    """
    pred_ids = [i for i in np.unique(pred_mask) if i > 0]
    gt_ids   = [i for i in np.unique(gt_mask)   if i > 0]

    if not pred_ids or not gt_ids:
        return [], pred_ids, gt_ids

    # Build a table of IoU scores: one row per prediction, one column per label
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    for i, pid in enumerate(pred_ids):
        for j, gid in enumerate(gt_ids):
            inter = np.logical_and(pred_mask == pid, gt_mask == gid).sum()
            union = np.logical_or( pred_mask == pid, gt_mask == gid).sum()
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    # Find the best assignment (maximize IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches, fp_ids, fn_ids = [], list(pred_ids), list(gt_ids)
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((pred_ids[r], gt_ids[c], iou_matrix[r, c]))
            fp_ids.remove(pred_ids[r])
            fn_ids.remove(gt_ids[c])

    return matches, fp_ids, fn_ids
```

### 9.4 — Computing Scores

Once you have the matches:

```python
def compute_metrics(matches, fp_ids, fn_ids):
    TP = len(matches)   # correct detections
    FP = len(fp_ids)    # predicted bubbles that don't exist in the labels
    FN = len(fn_ids)    # labeled bubbles the model missed

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou  = np.mean([m[2] for m in matches]) if matches else 0.0

    return {"TP": TP, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall,
            "F1": f1, "mean_IoU": mean_iou}
```

### 9.5 — Recommended Workflow

1. **Use held-out test images** — either reserve a subset of gold annotations before training, or use a gold version that was never included in any training run. Never test on images the model was trained on.
2. **Run inference on all test images** and save the output masks.
3. **Compare each prediction to its ground truth** using `match_instances` and `compute_metrics`.
4. **Aggregate across all images** — report mean F1, mean AP@0.5, etc.
5. **Visualize failures** — draw predicted bubble outlines on top of the original images to see where the model is going wrong (missed bubbles, phantom detections, bad shapes).

### 9.6 — Where to Put Your Code

Start a new file at `bubbly_flows/scripts/evaluate.py`. Here is a minimal skeleton to fill in:

```python
#!/usr/bin/env python3
"""
evaluate.py

Compare predicted masks against gold-standard masks and report metrics.

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
    parser.add_argument("--preds", required=True, type=Path,
                        help="Folder containing predicted mask files")
    parser.add_argument("--gts",   required=True, type=Path,
                        help="Folder containing gold-standard mask files")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Minimum overlap to count a detection as correct")
    args = parser.parse_args()

    pred_files = sorted(args.preds.glob("*.tif")) + sorted(args.preds.glob("*.png"))
    results = []

    for pred_path in pred_files:
        gt_path = args.gts / pred_path.name
        if not gt_path.exists():
            gt_path = gt_path.with_suffix(".tif")
        if not gt_path.exists():
            print(f"  [skip] no gold mask found for {pred_path.name}")
            continue

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        gt_mask   = cv2.imread(str(gt_path),   cv2.IMREAD_UNCHANGED)

        if pred_mask is None or gt_mask is None:
            print(f"  [error] could not read {pred_path.name}")
            continue

        # TODO: call match_instances() and compute_metrics() here
        print(f"  {pred_path.name}: prediction shape {pred_mask.shape}, gt shape {gt_mask.shape}")
        results.append(pred_path.name)

    print(f"\nProcessed {len(results)} image pairs.")
    # TODO: print aggregated metrics


if __name__ == "__main__":
    main()
```

---

## 10. Quick Reference

### Key Commands

| Action | Command |
|---|---|
| Load the package manager | `module load miniforge3` |
| Activate the project environment | `conda activate bubbly-train-env` |
| Open the management menu | `python bubbly_flows/scripts/manage_bubbly.py` |
| Check running jobs | `squeue -u <your-username>` |
| Watch a training log live | `tail -f bubbly_flows/logs/<job>.out` |
| Cancel a job | `scancel <job_id>` |
| Run inference directly | `python bubbly_flows/scripts/inference.py --model_path ... --image ... --output ...` |

### Key Locations

| What | Where |
|---|---|
| Gold annotation files | `bubbly_flows/annotations/gold/<version>/labels_json/` |
| Exported training datasets | `bubbly_flows/microsam/datasets/<name>/` |
| Saved model weights | `bubbly_flows/microsam/models/<name>/checkpoints/<name>/best.pt` |
| Training logs | `bubbly_flows/logs/<name>_<job_id>.out` |
| Action history | `bubbly_flows/diary.log` |

### Pipeline at a Glance

```
Gold annotation files  (labels_json/*.json)
         ↓   manage_bubbly.py — Option 4
Training dataset  (microsam/datasets/<name>/)
         ↓   manage_bubbly.py — Option 5  →  Slurm cluster job
Trained model weights  (microsam/models/<name>/best.pt)
         ↓   inference.py
Predicted masks
         ↓   evaluate.py  (to be built)
Performance metrics: F1, AP@0.5, mean IoU
```

---

> **When stuck**: `README.md` explains the full project structure, `USER_GUIDE.md` covers the annotation workflow, and `bubbly_flows/diary.log` is a chronological record of every action taken in the pipeline. The source code of `manage_bubbly.py` is also well commented and is the best reference for understanding how data moves through the system.
