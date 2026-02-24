# Bubble Tracking
## A quick guide on training models

This guide serves as a tutorial for anyone who wants to train their own models using our infrastructure in Oscar CCV to classify (segment) bubbles and evaluate their performance.
Human-labeled annotations are assumed to exist in `bubbly_flows/annotations/gold/`.

---

## Table of Contents

0. [Prerequisites](#0-prerequisites)
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

## 0. Prerequisites

Oscar cluster:
- [Connecting to Oscar](https://docs.ccv.brown.edu/oscar/connecting-to-oscar)
- [Oscar overview and getting started](https://docs.ccv.brown.edu/oscar/getting-started)
- [Submitting batch jobs with Slurm](https://docs.ccv.brown.edu/oscar/submitting-jobs/batch)
- [Oscar storage and file systems](https://docs.ccv.brown.edu/oscar/managing-files/filesystem)

Linux command line:
- [Linux command line cheat sheet](https://cheatography.com/davechild/cheat-sheets/linux-command-line/)
- [The Linux command line for beginners (Ubuntu)](https://ubuntu.com/tutorials/command-line-for-beginners)

Conda, Git, and the underlying model:
- [Conda cheat sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
- [Git cheat sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [SAM paper (Meta AI)](https://ai.meta.com/research/publications/segment-anything/)
- [MicroSAM documentation](https://computational-cell-analytics.github.io/micro-sam/)

---

## 1. What Is This Project?

This project builds a machine-learning pipeline to automatically detect and segment bubbles in images taken during zero-gravity flight experiments — essentially, training a model to answer: *where are all the bubbles in this image, and what are their exact shapes?*

We do this by fine-tuning existing models on our own labeled data. Fine-tuning means taking a model already trained on general images and re-training it on our specific bubble images, so it becomes specialized for this task. We work with several model families: MicroSAM (a version of Meta's Segment Anything Model adapted for scientific imaging, and our current main focus), CNN-based segmentation architectures, and others as the project evolves.

This guide covers converting human annotations into a training-ready format, running the training job on the cluster, and measuring how well the trained models perform.

---

## 2. How the Repository Is Organized

Everything lives inside `bubbly_flows/`. Here is what matters for this workflow:

```
Bubble-tracking/
├── environment.yml                    # All Python packages needed
├── bubbly_flows/
│   ├── scripts/
│   │   ├── manage_bubbly.py           # The main script you will run
│   │   ├── train.py                   # Training script (launched by the cluster)
│   │   └── inference.py               # Run a trained model on a new image
│   ├── annotations/
│   │   └── gold/                      # Human-labeled ground truth (do not edit)
│   │       ├── gold_seed_v00/
│   │       │   ├── labels_json/       # One annotation file per image
│   │       │   └── manifest.csv       # List of which images were labeled
│   │       └── gold_v00/ ...
│   ├── microsam/
│   │   ├── datasets/                  # Training-ready datasets (images + masks)
│   │   └── models/                    # Saved model weights after training
│   ├── data/
│   │   └── patches_pool/images/       # Source image patches
│   └── logs/                          # Job scripts and training output logs
```

Data flows in one direction through the project:
```
annotations/gold/  →  microsam/datasets/  →  microsam/models/
 (annotation files)    (images + masks)       (trained model weights)
```

---

## 3. Environment Setup on Oscar

Log in and navigate to the project:

```bash
ssh <your-username>@ssh.ccv.brown.edu
cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
```

Oscar does not load conda by default, so you need to do this at the start of every session:

```bash
module load miniforge3
```

Adding that line to your `~/.bashrc` will save you from having to remember it each time.

The first time you work with this project, create the Python environment (takes about 5–10 minutes):

```bash
conda env create -f environment.yml
```

Then activate it — you will need to do this every session:

```bash
conda activate bubbly-train-env
```

To verify everything installed correctly:

```bash
python -c "import torch; import micro_sam; print('All good!', torch.__version__)"
```

---

## 4. Understanding the Data

### Gold annotations

Gold annotations are the human-labeled ground truth. Each gold version is a snapshot of the labels at a point in time:

```
bubbly_flows/annotations/gold/gold_seed_v00/
├── labels_json/          # One file per labeled image
│   ├── SomeImage__x0320_y0640.json
│   └── ...
├── manifest.csv          # Which images are in this version
└── stats.json            # Counts and provenance metadata
```

Each `.json` file describes where the bubbles are in one image as a set of polygons, one polygon per bubble. To peek inside one:

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

### Training dataset format

When you export a dataset (Step 1 below), those polygon files get converted into pixel masks that the training code can read:

- `microsam/datasets/<name>/images/` — the image patches
- `microsam/datasets/<name>/labels/` — paired mask files where each pixel's value is a bubble ID (0 = background, 1 = first bubble, 2 = second bubble, ...)

Each image and its mask share the same filename, with a `.tif` extension for the mask.

---

## 5. Step 1 — Export a Training Dataset

```bash
conda activate bubbly-train-env
python bubbly_flows/scripts/manage_bubbly.py
```

A menu will appear. Select option 4 (Prepare MicroSAM Dataset):

```
========================================
   BUBBLY FLOWS - DATASET MANAGER
========================================
1. Initialize / Update Patch Pool
2. Create New Workspace
3. Promote Workspace to Gold (+ Cleanup)
4. Prepare MicroSAM Dataset (Export)
5. Train Model (submit job)
6. Run Inference
q. Quit
```

The script will ask you to pick a gold version (e.g. `gold_seed_v00`), give the dataset a name (e.g. `v01_seed_aug`), and whether to enable augmentation. Say yes to augmentation — it automatically generates extra training variations of each image using flips, rotations, and brightness changes, ending up with roughly 4× more training examples at no extra labeling cost. Use seed `42` to keep it reproducible.

The exported dataset lands in `bubbly_flows/microsam/datasets/<your-dataset-name>/`. A quick sanity check:

```bash
ls bubbly_flows/microsam/datasets/<your-dataset-name>/images/ | wc -l
ls bubbly_flows/microsam/datasets/<your-dataset-name>/labels/ | wc -l
```

Both counts should match.

---

## 6. Step 2 — Train a Model

From the same menu, select option 5. Pick the dataset you just exported, give the run a name (e.g. `train_v01_seed_run1`), and set a time limit in hours (4 is reasonable for a first run).

The script generates a Slurm job file in `bubbly_flows/logs/` and asks if you want to submit it. The job allocates a GPU, activates the environment, and runs `train.py` for 100 epochs using MicroSAM ViT-B as the starting point. The best-performing weights get saved to `bubbly_flows/microsam/models/<name>/checkpoints/<name>/best.pt`.

To check whether the job is running:

```bash
squeue -u <your-username>
```

To cancel it:

```bash
scancel <job_id>
```

---

## 7. Step 3 — Monitor Training

Logs are saved to `bubbly_flows/logs/`. Two files are created per job:

- `<exp_name>_<job_id>.out` — training progress and loss values
- `<exp_name>_<job_id>.err` — warnings and errors

To watch the log live:

```bash
tail -f bubbly_flows/logs/<exp_name>_<job_id>.out
```

| What you see | What it means |
|---|---|
| `Using device: cuda` | GPU is active |
| `Using device: cpu` | No GPU was allocated — training will be impractically slow |
| Loss decreasing over epochs | The model is learning |
| `TRAINING COMPLETE` | Done |
| Error message in the `.err` file | Something went wrong |

Two common problems:
- `CUDA driver version is insufficient` — the job was not allocated a GPU. Check that the generated `.sh` file in `bubbly_flows/logs/` contains `#SBATCH -p gpu`.
- `Module not found: torch` — the environment was not activated in the job script. Look for `conda activate bubbly-train-env` in the `.sh` file.

---

## 8. Step 4 — Run Inference

Once training is done, select option 6 from the menu to run the model on a new image, or call the script directly:

```bash
python bubbly_flows/scripts/inference.py \
    --model_path bubbly_flows/microsam/models/<exp_name>/checkpoints/<exp_name>/best.pt \
    --image bubbly_flows/data/patches_pool/images/<some_image>.png \
    --output output/<some_image>_pred.png \
    --model_type vit_b
```

The output is a mask image where each pixel's value is a predicted bubble ID (0 = background, 1 = first bubble, etc.), the same format as the gold masks in `microsam/datasets/*/labels/`.

---

## 9. Step 5 — Build an Evaluation Pipeline

There is no evaluation pipeline yet. This section explains the problem and gives a starting point.

### What you have to work with

After running inference on a test image, you have two masks to compare:

| | Path | Format |
|---|---|---|
| Prediction | wherever you saved it | mask file, pixel = predicted bubble ID |
| Ground truth | `microsam/datasets/<ds>/labels/<stem>.tif` | mask file, pixel = labeled bubble ID |

The catch: bubble IDs in the prediction and ground truth are not aligned. Bubble "3" in the prediction might be the same physical bubble as "7" in the ground truth. You have to figure out which predicted bubble corresponds to which labeled bubble before computing any score — this is called the matching problem.

### Metrics

Intersection over Union (IoU) measures overlap between two masks. A perfect match scores 1.0; no overlap scores 0.0:

```
IoU = (pixels in both masks) / (pixels in either mask)
```

F1 score balances finding all real bubbles (recall) against avoiding false detections (precision). It is a good first metric to get working.

Average Precision (AP) is the standard rigorous benchmark. At IoU ≥ 0.5, a predicted bubble counts as a correct detection if it overlaps enough with a labeled bubble. AP@0.5 and AP@0.75 are what most papers report.

Panoptic Quality (PQ) combines how many bubbles were found with how well their shapes were drawn. Worth reading about once the basics are working.

### Solving the matching problem

The standard solution is the Hungarian algorithm, which finds the best one-to-one assignment between predicted and labeled bubbles:

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

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches, fp_ids, fn_ids = [], list(pred_ids), list(gt_ids)
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((pred_ids[r], gt_ids[c], iou_matrix[r, c]))
            fp_ids.remove(pred_ids[r])
            fn_ids.remove(gt_ids[c])

    return matches, fp_ids, fn_ids
```

### Computing scores

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

### Suggested workflow

Reserve some gold annotations as a test set before training — either a held-out subset or a gold version that was never used in any training run. Run inference on all test images, compare each prediction against its ground truth using the functions above, and report aggregated metrics (mean F1, AP@0.5, etc.). Visualizing failures is also worth doing early: draw predicted bubble outlines on top of the original images to see where things go wrong.

A good place to start the evaluation script is `bubbly_flows/scripts/evaluate.py`. Here is a skeleton:

```python
#!/usr/bin/env python3
"""
evaluate.py — compare predicted masks against gold masks and report metrics.

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
        print(f"  {pred_path.name}: pred {pred_mask.shape}, gt {gt_mask.shape}")
        results.append(pred_path.name)

    print(f"\nProcessed {len(results)} image pairs.")
    # TODO: print aggregated metrics


if __name__ == "__main__":
    main()
```

---

## 10. Quick Reference

| Action | Command |
|---|---|
| Load conda | `module load miniforge3` |
| Activate environment | `conda activate bubbly-train-env` |
| Open management menu | `python bubbly_flows/scripts/manage_bubbly.py` |
| Check running jobs | `squeue -u <your-username>` |
| Watch training log | `tail -f bubbly_flows/logs/<job>.out` |
| Cancel a job | `scancel <job_id>` |
| Run inference | `python bubbly_flows/scripts/inference.py --model_path ... --image ... --output ...` |

| What | Where |
|---|---|
| Gold annotations | `bubbly_flows/annotations/gold/<version>/labels_json/` |
| Training datasets | `bubbly_flows/microsam/datasets/<name>/` |
| Model weights | `bubbly_flows/microsam/models/<name>/checkpoints/<name>/best.pt` |
| Training logs | `bubbly_flows/logs/<name>_<job_id>.out` |
| Action history | `bubbly_flows/diary.log` |

```
annotations/gold/  (labels_json/*.json)
        ↓  manage_bubbly.py — option 4
microsam/datasets/<name>/
        ↓  manage_bubbly.py — option 5  →  Slurm job
microsam/models/<name>/best.pt
        ↓  inference.py
predicted masks
        ↓  evaluate.py
F1, AP@0.5, mean IoU
```

---

When stuck, `README.md` has the full project structure, `USER_GUIDE.md` covers the annotation workflow, and `bubbly_flows/diary.log` is a running record of every action taken in the pipeline. The source of `manage_bubbly.py` is also well commented and is the best reference for how data moves through the system.
