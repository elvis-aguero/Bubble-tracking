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
4. [Download Base Model Weights (one-time)](#4-download-base-model-weights-one-time)
5. [Understanding the Data](#5-understanding-the-data)
6. [Step 1 — Export a Training Dataset](#6-step-1--export-a-training-dataset)
7. [Step 2 — Train a Model](#7-step-2--train-a-model)
8. [Step 3 — Monitor Training](#8-step-3--monitor-training)
9. [Step 4 — Run Inference](#9-step-4--run-inference)
10. [Step 5 — Evaluate](#10-step-5--evaluate)
11. [Quick Reference](#11-quick-reference)

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
│   │   ├── download_models.sh         # One-time weight download (run before first training)
│   │   ├── train.py                   # MicroSAM ViT-B training
│   │   ├── train_stardist.py          # StarDist 2D (HZDR pre-trained weights)
│   │   ├── train_yolov9.py            # YOLOv9c-seg (COCO pre-trained weights)
│   │   ├── train_maskrcnn.py          # Mask R-CNN ResNet-50+FPN (COCO pre-trained)
│   │   └── inference.py               # Run a trained model on a new image
│   ├── annotations/
│   │   └── gold/                      # Human-labeled ground truth (do not edit)
│   │       ├── gold_seed_v00/
│   │       │   ├── labels_json/       # One annotation file per image
│   │       │   └── manifest.csv       # List of which images were labeled
│   │       └── gold_v00/ ...
│   ├── microsam/
│   │   └── datasets/                  # Training-ready datasets (images + masks)
│   ├── data/
│   │   └── patches_pool/images/       # Source image patches
│   └── logs/                          # Job scripts and training output logs
~/scratch/bubble-models/
│   ├── microsam/models/vit_b.pt       # MicroSAM ViT-B light-microscopy weights
│   ├── stardist/hzdr_2022/            # Hessenkemper 2022 bubble-specific StarDist
│   ├── stardist/hzdr_bubble_column/   # Hessenkemper 2024 bubble column StarDist
│   ├── yolo/yolov9c-seg.pt            # YOLOv9c-seg COCO weights
│   ├── bubmask/mask_rcnn_bubble.h5    # BubMask (Kim & Park 2021, downloaded from Google Drive)
│   └── trained/<exp_name>/            # All post-training checkpoints land here
```

Data flows in one direction through the project:
```
annotations/gold/  →  microsam/datasets/  →  ~/scratch/bubble-models/trained/
 (annotation files)    (images + masks)          (trained model weights)
```

---

## 3. Environment Setup on Oscar

Log in to Oscar:

```bash
ssh <your-username>@ssh.ccv.brown.edu
```

The repository lives in shared project storage. Navigate there (ask a team member for the exact path if you are not sure):

```bash
cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
```

Oscar does not load conda by default, so you need to do this at the start of every session:

```bash
module load miniforge3
```

Adding that line to your `~/.bashrc` will save you from having to remember it each time.

To create the Python environment for the first time, use `mamba` — it is much faster than `conda` for large environments like this one (expect 10–15 minutes instead of 1+ hour):

```bash
mamba env create -f environment.yml
```

If that fails with `CondaValueError: prefix already exists`, the environment was previously created but may be incomplete. Fix it with:

```bash
mamba env update --name bubbly-train-env --file environment.yml
```

Then activate it — you will need to do this every session:

```bash
conda activate bubbly-train-env
```

To verify everything installed correctly:

```bash
python -c "import torch; import micro_sam; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` will return `False` on the login node even if CUDA is properly installed — that is expected. The GPU is only available inside a Slurm job.

---

## 4. Download Base Model Weights (one-time)

Before training, you need to stage pre-trained model weights to your scratch space. Run this once with the conda environment active:

```bash
bash bubbly_flows/scripts/download_models.sh
```

This creates `~/scratch/bubble-models/` with the following weights:

| Model | Why |
|---|---|
| **MicroSAM vit_b_lm** | Fine-tuned on light microscopy images — best starting point for bubble fine-tuning (our primary model) |
| **StarDist HZDR 2022** | Bubble-specific checkpoints from Hessenkemper et al. 2022; AP@0.5 ~0.91 on air-water flow |
| **StarDist HZDR bubble column** | Second HZDR bubble-specific release (Hessenkemper et al. 2024) |
| **YOLOv9c-seg** | Real-time ~20 fps option; no bubble-specific release from papers, so this starts from COCO weights |
| **BubMask (Mask R-CNN)** | Kim & Park 2021; downloaded from Google Drive via `gdown` |

The script is idempotent — safe to re-run. It prints a status summary at the end.

`manage_bubbly.py` checks for the required weights before submitting any Slurm job and exits with a clear error if they are missing.

---

## 5. Understanding the Data

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

## 6. Step 1 — Export a Training Dataset

```bash
conda activate bubbly-train-env
python bubbly_flows/scripts/manage_bubbly.py
```

Select option 4 (Prepare Training Dataset). The menu looks like this:

```
========================================
   BUBBLY FLOWS - DATASET MANAGER
========================================
1. Initialize / Update Patch Pool
2. Create New Workspace
3. Promote Workspace to Gold (+ Cleanup)
4. Prepare Training Dataset (Export)
5. Train Model (submit job)
6. Run Inference (Stub)
q. Quit
```

Pick a gold version, then answer the prompts:

1. **Enable train/test split?** — say `y`. The script will randomly hold out a fraction of the labeled images as a test set, which you will use later for evaluation. Images in the test set are never seen by the model during training.
2. **Test fraction** — accept the default (`0.2` = 20% held out).
3. **Split seed** — accept `42` for reproducibility.
4. **Base dataset name** — give it a name like `v01_seed`. The script appends `_train` and `_test` automatically.

The script exports two datasets:
- `bubbly_flows/microsam/datasets/v01_seed_train/` — training images with augmentation (~4× the source count)
- `bubbly_flows/microsam/datasets/v01_seed_test/`  — test images, no augmentation, for evaluation only

A quick sanity check:

```bash
ls bubbly_flows/microsam/datasets/v01_seed_train/images/ | wc -l
ls bubbly_flows/microsam/datasets/v01_seed_train/labels/ | wc -l
```

Both counts should match.

---

## 7. Step 2 — Train a Model

From the same menu, select option 5. Pick the dataset you just exported. You will then be asked which training script to use:

- Option 1 is always the built-in MicroSAM ViT-B (`train.py`).
- Any custom scripts named `train_<modelname>.py` in `bubbly_flows/scripts/` appear as additional options automatically.
- A final option lets you type a path manually.

Four training scripts are currently available:

| Script | Model | Starting weights | Notes |
|---|---|---|---|
| `train.py` | **MicroSAM ViT-B** | `~/scratch/bubble-models/microsam/models/vit_b.pt` (light-microscopy fine-tuned SAM) | Primary model; best domain match |
| `train_stardist.py` | **StarDist 2D** | `~/scratch/bubble-models/stardist/hzdr_2022/` (Hessenkemper 2022, AP@0.5 ~0.91) | Fast inference; TF/Keras |
| `train_yolov9.py` | **YOLOv9c-seg** | `~/scratch/bubble-models/yolo/yolov9c-seg.pt` (COCO) | Real-time capable; converts masks to YOLO polygon format on the fly |
| `train_maskrcnn.py` | **Mask R-CNN** | COCO weights (auto-downloaded by torchvision) | Standard CNN baseline; pure PyTorch |

Give the run a name (e.g. `microsam_v01_seed_run1`) and set a time limit in hours (4 is reasonable for a first run).

The script generates a Slurm job file in `bubbly_flows/logs/` and asks if you want to submit it. The job allocates a GPU, activates the environment, sets `MICROSAM_CACHEDIR`, and runs your selected script for 100 epochs. All trained weights are saved to `~/scratch/bubble-models/trained/<exp_name>/`.

To check whether the job is running:

```bash
squeue -u <your-username>
```

To cancel it:

```bash
scancel <job_id>
```

### Writing a custom training script

To add support for a new model, place a file named `train_<modelname>.py` in `bubbly_flows/scripts/` (e.g. `train_unet.py`, `train_cellpose.py`). It will appear automatically as an option in the training menu — no other changes needed.

The script must accept these command-line arguments:

| Argument | Type | Required | What it is |
|---|---|---|---|
| `--dataset PATH` | path | yes | Root of the training dataset (has `images/` and `labels/` subdirectories) |
| `--name STR` | string | yes | Experiment name, used for checkpoint folder naming |
| `--epochs INT` | int | no (default 50) | Number of training epochs |
| `--save_root PATH` | path | no | Root directory for saving checkpoints; `manage_bubbly.py` passes `~/scratch/bubble-models/trained/` automatically |

The dataset format is fixed regardless of model: `images/<stem>.png` paired with `labels/<stem>.tif`, where each pixel value is an integer instance ID (0 = background, 1 = first bubble, 2 = second bubble, ...). This is exactly what option 4 of the menu produces.

When `--save_root` is provided, save checkpoints to `<save_root>/<name>/`. When it is absent, fall back to `bubbly_flows/microsam/models/<name>/`. The menu always passes `--save_root`, so trained models land in `~/scratch/bubble-models/trained/` on the cluster. Print training progress to stdout — Slurm captures it in the `.out` log file automatically.

A minimal skeleton to start from:

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True, type=Path)
    parser.add_argument("--name",      required=True, type=str)
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--save_root", type=Path, default=None)
    args = parser.parse_args()

    images_dir = args.dataset / "images"
    labels_dir = args.dataset / "labels"
    if args.save_root:
        save_dir = args.save_root / args.name
    else:
        save_dir = Path(__file__).resolve().parent.parent / "microsam" / "models" / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- your model training code here ---
    print(f"Training {args.name} for {args.epochs} epochs.")
    print(f"Images:  {images_dir}")
    print(f"Labels:  {labels_dir}")
    print(f"Output:  {save_dir}")

if __name__ == "__main__":
    main()
```

That is the entire interface. The menu and Slurm scaffolding handle everything else.

---

## 8. Step 3 — Monitor Training

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

Three common problems:
- `CUDA driver version is insufficient` — the job was not allocated a GPU. Check that the generated `.sh` file in `bubbly_flows/logs/` contains `#SBATCH -p gpu`.
- `Module not found: torch` — the environment was not activated in the job script. Look for `conda activate bubbly-train-env` in the `.sh` file.
- `Using device: cpu` despite `--gres=gpu:1` — PyTorch was installed without CUDA support. The conda-forge build of PyTorch is CPU-only. To fix, reinstall with the CUDA-enabled build from the `pytorch` channel (the `environment.yml` already pins `pytorch::pytorch` and `pytorch::pytorch-cuda=11.8` to ensure this). If the environment was created before this fix, run:

  ```bash
  conda install -n bubbly-train-env pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch::pytorch-cuda=11.8
  ```

---

## 9. Step 4 — Run Inference

There is no unified inference script across all model families yet. Each one uses its own library API. Option 6 in the menu ("Run Inference (Stub)") only covers MicroSAM and also looks in the wrong directory — use the snippets below directly.

All trained checkpoints land under `~/scratch/bubble-models/trained/<exp_name>/`.

### MicroSAM

```bash
python bubbly_flows/scripts/inference.py \
    --model_path ~/scratch/bubble-models/trained/<exp_name>/checkpoints/<exp_name>/best.pt \
    --image bubbly_flows/data/patches_pool/images/<some_image>.png \
    --output output/<some_image>_pred.png \
    --model_type vit_b
```

### StarDist

StarDist saves its model in a directory structure it manages itself. Load it by pointing to the basedir and name:

```python
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from pathlib import Path
import numpy as np, cv2, tifffile

model = StarDist2D(None, name="<exp_name>",
                   basedir=str(Path.home() / "scratch/bubble-models/trained"))

img = cv2.imread("path/to/image.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img = normalize(img, 1, 99.8)
labels, _ = model.predict_instances(img)
tifffile.imwrite("output_mask.tif", labels.astype(np.uint16))
```

### YOLOv9

```python
from ultralytics import YOLO
import numpy as np, cv2, tifffile
from pathlib import Path

model = YOLO(str(Path.home() / "scratch/bubble-models/trained/<exp_name>/weights/best.pt"))
results = model.predict("path/to/image.png", imgsz=512, conf=0.25)

# Convert YOLO instance masks to a labeled mask
img = cv2.imread("path/to/image.png")
label_map = np.zeros(img.shape[:2], dtype=np.uint16)
if results[0].masks is not None:
    for i, m in enumerate(results[0].masks.data.cpu().numpy()):
        mask = cv2.resize(m, (img.shape[1], img.shape[0])) > 0.5
        label_map[mask] = i + 1
tifffile.imwrite("output_mask.tif", label_map)
```

### Mask R-CNN

```python
import torch, cv2, numpy as np, tifffile
from pathlib import Path
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes=2):
    m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    in_fm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(in_fm, 256, num_classes)
    return m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model()
model.load_state_dict(torch.load(
    Path.home() / "scratch/bubble-models/trained/<exp_name>/best.pt",
    map_location=device))
model.to(device).eval()

img_bgr = cv2.imread("path/to/image.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(tensor)[0]

label_map = np.zeros(img_bgr.shape[:2], dtype=np.uint16)
scores = out["scores"].cpu().numpy()
masks  = out["masks"].squeeze(1).cpu().numpy()
for i, (score, mask) in enumerate(zip(scores, masks)):
    if score > 0.5:
        label_map[mask > 0.5] = i + 1
tifffile.imwrite("output_mask.tif", label_map)
```

In all cases the output mask has the same format: pixel value = bubble instance ID (0 = background, 1 = first bubble, 2 = second bubble, ...), matching the gold masks in `microsam/datasets/*/labels/`.

---

## 10. Step 5 — Evaluate

`bubbly_flows/scripts/evaluate.py` is a complete evaluation script. Point it at a directory of predicted masks and the ground-truth labels from your test dataset:

```bash
python bubbly_flows/scripts/evaluate.py \
    --preds output/preds/ \
    --gts   bubbly_flows/microsam/datasets/v01_seed_test/labels/ \
    --iou_threshold 0.5 \
    --output results.csv       # optional
```

It prints a per-image table followed by an aggregate summary:

```
image                                     TP     FP     FN     precision   recall     F1       mean_IoU
  ------------------------------------------------------------------------------------------------
  SomeImage__x0320_y0640_pred.tif         12     1      2      0.923       0.857      0.889    0.781
  ...

--- Summary (24 images, IoU threshold = 0.5) ---
Total:   TP=287  FP=14  FN=23
Macro:   precision=0.941  recall=0.893  F1=0.916  mean_IoU=0.804
Micro:   precision=0.953  recall=0.926  F1=0.939
```

**How it works:**
- Bubble IDs in predictions and ground truth are not aligned (the model's "bubble 3" may be the same physical bubble as the label's "bubble 7"). The script solves this with the Hungarian algorithm, which finds the optimal one-to-one matching between predicted and labeled instances.
- A predicted bubble counts as a correct detection (TP) if its IoU with the matched ground-truth bubble is ≥ `--iou_threshold`.
- Unmatched predictions are false positives (FP); unmatched ground-truth bubbles are false negatives (FN).
- **Macro** aggregates by giving equal weight to each image; **Micro** pools all TP/FP/FN counts first.

The metrics reported (precision, recall, F1, mean IoU) are the standard benchmarks used in the bubble segmentation literature (e.g., Hessenkemper et al. 2022 report AP@0.5, which is equivalent to precision@0.5 with 1:1 matching).

---

## 11. Quick Reference

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
| Base model weights | `~/scratch/bubble-models/{microsam,stardist,yolo,bubmask}/` |
| Trained model weights | `~/scratch/bubble-models/trained/<exp_name>/` |
| Training logs | `bubbly_flows/logs/<name>_<job_id>.out` |
| Action history | `bubbly_flows/diary.log` |

```
annotations/gold/  (labels_json/*.json)
        ↓  manage_bubbly.py — option 4  (train/test split + export)
microsam/datasets/<name>_train/   microsam/datasets/<name>_test/
        ↓  manage_bubbly.py — option 5  →  Slurm job
~/scratch/bubble-models/trained/<name>/
        ↓  inference.py / model-specific snippet
predicted masks  (run on test images)
        ↓  evaluate.py --preds ... --gts datasets/<name>_test/labels/
precision, recall, F1, mean IoU
```

---

When stuck, `README.md` has the full project structure, `USER_GUIDE.md` covers the annotation workflow, and `bubbly_flows/diary.log` is a running record of every action taken in the pipeline. The source of `manage_bubbly.py` is also well commented and is the best reference for how data moves through the system.
