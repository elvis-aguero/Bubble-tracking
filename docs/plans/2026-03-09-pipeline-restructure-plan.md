# Pipeline Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the end-to-end ML pipeline so a new user can go from annotated frames to trained model to evaluation entirely through `manage_bubbly.py`, with all engineering decisions pre-made in config JSON files.

**Architecture:** Config-first — each model has a canonical `configs/<model>.json` owned by the engineer. Training scripts read from that file; no hyperparameter flags at the CLI. A frozen copy of the config is written into each trained run directory for provenance. `manage_bubbly.py` is rewritten with a 4-option happy-path main menu, per-submenu tooltips, a state line, and prerequisite checks.

**Tech Stack:** Python 3.11, micro_sam 1.7.5, StarDist2D, Ultralytics YOLOv9, Slurm (sbatch), Oscar HPC cluster (Brown CCV). Conda env: `bubbly-train-env`.

**Design doc:** `docs/plans/2026-03-09-pipeline-restructure-design.md`

## Implementation Status

- [x] Task 1: Create `configs/` directory with per-model JSON files
- [x] Task 2: Rename `microsam/` → `pipeline/` on disk and in code
- [x] Task 3: Refactor `train.py` to read from config JSON
- [x] Task 4: Refactor `train_stardist.py` and `train_yolov9.py` to read from config JSON
  - [x] `train_stardist.py` now reads `--config`
  - [x] `train_yolov9.py` now reads `--config`
- [x] Task 5: Config provenance — copy JSON into run directory at submission

### `manage_bubbly.py` Milestones

- [x] Replace legacy top-level menu with 4-option happy path + `Advanced`
- [x] Add `Advanced` submenu for patch pool, workspace creation, and dataset export
- [x] Add computed state line: `gold=... dataset=... last_run=...`
- [x] Add prerequisite blocking for Train / Evaluate / Inference
- [x] Redesign `Train Model` selection to choose model family first (`MicroSAM`, `StarDist`, `YOLOv9`, `Other`)
- [x] Restrict training dataset picker to exported `*_train` datasets
- [x] Add per-option tooltips in menus
- [x] Copy chosen config into trained run directory after submission
- [x] Refactor built-in StarDist and YOLO trainers to consume their config JSONs directly
  - [x] StarDist complete
  - [x] YOLOv9 complete
- [x] Redesign Evaluate flow to fully match the approved plan
- [x] Redesign Inference flow to fully match the approved plan

---

## Task 1: Create `configs/` directory with per-model JSON files

**Files:**
- Create: `configs/microsam.json`
- Create: `configs/stardist.json`
- Create: `configs/yolov9.json`

**Step 1: Create configs/microsam.json**

```json
{
  "model": "microsam",
  "backbone": "vit_b",
  "training": {
    "patch_shape": 1024,
    "epochs": 100,
    "batch_size": 1,
    "num_workers": 4,
    "val_fraction": 0.15,
    "early_stopping_patience": 10,
    "freeze": ["image_encoder"]
  },
  "inference": {
    "model_type": "vit_b"
  },
  "notes": "Full-resolution 1024px crops. Encoder frozen — only UNETR decoder trains. patch_shape matches frame resolution (1024x1024) to avoid crop-vs-inference mismatch."
}
```

**Step 2: Create configs/stardist.json**

```json
{
  "model": "stardist",
  "training": {
    "epochs": 100,
    "batch_size": 2,
    "val_fraction": 0.15,
    "n_rays": 64,
    "grid": [2, 2]
  },
  "inference": {
    "prob_thresh": null,
    "nms_thresh": 0.4
  },
  "notes": "Fine-tuned from HZDR pre-trained weights when available. prob_thresh=null means use the auto-optimised threshold saved by training. grid=[2,2] matches bubble scale at 1024px resolution."
}
```

**Step 3: Create configs/yolov9.json**

```json
{
  "model": "yolov9",
  "training": {
    "epochs": 100,
    "imgsz": 1024,
    "batch": 4,
    "val_fraction": 0.15
  },
  "inference": {
    "imgsz": 1024,
    "conf": 0.25,
    "iou": 0.7
  },
  "notes": "imgsz=1024 matches frame resolution. Previously trained at 512 — that caused inference mismatch with full-frame test images."
}
```

**Step 4: Commit**

```bash
git add configs/
git commit -m "feat: add per-model hyperparameter config JSONs

All engineering decisions live here. Training scripts will read
these instead of accepting individual CLI flags. Each trained run
gets a frozen copy for provenance.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Rename `microsam/` → `pipeline/` on disk and in code

The `microsam/` directory name is misleading now that we have four model backends.

**Files:**
- Modify: `bubbly_flows/scripts/manage_bubbly.py` — `MICROSAM_DIR` constant + all references
- Modify: `README.md`, `USER_GUIDE.md`, `TRAINING_GUIDE.md` — update path references

**Step 1: Rename directory on disk**

```bash
mv bubbly_flows/microsam bubbly_flows/pipeline
```

**Step 2: Update the constant in manage_bubbly.py**

Find line ~194:
```python
MICROSAM_DIR = ROOT_DIR / "microsam"
```
Change to:
```python
PIPELINE_DIR = ROOT_DIR / "pipeline"
```

Then replace all remaining uses of `MICROSAM_DIR` with `PIPELINE_DIR` throughout the file.
Also replace string literals `"microsam/datasets"` → `"pipeline/datasets"` in any print statements.

**Step 3: Update docs (grep for microsam/ path references)**

```bash
grep -rn "microsam/" README.md USER_GUIDE.md TRAINING_GUIDE.md
```
Update each occurrence of `microsam/datasets` → `pipeline/datasets`.

**Step 4: Add x-labeling-env to .gitignore**

Append to `.gitignore`:
```
x-labeling-env/
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: rename microsam/ → pipeline/, gitignore x-labeling-env

pipeline/ is model-agnostic. microsam/ name was misleading since
StarDist and YOLOv9 also use the same datasets directory.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Refactor `train.py` to read from config JSON

Remove all hyperparameter CLI flags. Accept `--config` only.

**Files:**
- Modify: `bubbly_flows/scripts/train.py`

**Step 1: Rewrite the argument parser**

Replace current `parse_args` block (lines ~24-31) with:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True, type=Path,
                        help="Path to dataset root (images/ and labels/ subdirs)")
    parser.add_argument("--name",     required=True, type=str,
                        help="Experiment name (used for checkpoint directory)")
    parser.add_argument("--config",   required=True, type=Path,
                        help="Path to model config JSON (e.g. configs/microsam.json)")
    parser.add_argument("--save_root", type=Path, default=None,
                        help="Root for checkpoints. Default: ~/scratch/bubble-models/trained")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    t = cfg["training"]
    patch_shape   = t.get("patch_shape", 1024)
    epochs        = t.get("epochs", 100)
    batch_size    = t.get("batch_size", 1)
    num_workers   = t.get("num_workers", 4)
    val_fraction  = t.get("val_fraction", 0.15)
    early_stop    = t.get("early_stopping_patience", 10)
    freeze        = t.get("freeze", ["image_encoder"])
    backbone      = cfg.get("backbone", "vit_b")

    print(f"STARTING TRAINING: {args.name}")
    print(f"Config:      {args.config}")
    print(f"Dataset:     {args.dataset}")
    print(f"Epochs:      {epochs}  patch={patch_shape}  batch={batch_size}  val={val_fraction}")
```

**Step 2: Replace hardcoded values in the rest of train.py**

- `patch_shape=(512, 512)` → `patch_shape=(patch_shape, patch_shape)` (both loaders)
- `batch_size=1` → `batch_size=batch_size`
- `num_workers=2` / `num_workers=1` → `num_workers=num_workers` (train/val respectively)
- `split_idx = max(1, int(len(...) * 0.9))` → `split_idx = max(1, int(len(...) * (1 - val_fraction)))`
- `freeze=["image_encoder"]` → `freeze=freeze`
- `model_type="vit_b"` → `model_type=backbone`
- `n_epochs=args.epochs` → `n_epochs=epochs`

**Step 3: Add `import json` at top of file if not present**

**Step 4: Verify train.py loads and prints correctly (dry-run)**

```bash
conda run -n bubbly-train-env python3 bubbly_flows/scripts/train.py \
  --dataset bubbly_flows/pipeline/datasets/seed_v04_train \
  --name test_config_read \
  --config configs/microsam.json \
  --save_root /tmp/test_train
```
Expected: prints config summary, then starts data loading (can Ctrl-C immediately).

**Step 5: Commit**

```bash
git add bubbly_flows/scripts/train.py
git commit -m "feat: train.py reads hyperparams from --config JSON

Removes --epochs, --patch_shape individual flags.
All values sourced from configs/microsam.json.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Refactor `train_stardist.py` and `train_yolov9.py` to read from config JSON

Same pattern as Task 3.

**Files:**
- Modify: `bubbly_flows/scripts/train_stardist.py`
- Modify: `bubbly_flows/scripts/train_yolov9.py`

**Step 1: train_stardist.py — update argument parser**

Replace `--epochs` flag with `--config`. Load JSON:

```python
parser.add_argument("--config", required=True, type=Path)
# ...
with open(args.config) as f:
    cfg = json.load(f)
t = cfg["training"]
epochs       = t.get("epochs", 100)
batch_size   = t.get("batch_size", 2)
val_fraction = t.get("val_fraction", 0.15)
```

Replace all hardcoded references to these values throughout the file.

**Step 2: train_yolov9.py — update argument parser**

```python
parser.add_argument("--config", required=True, type=Path)
# ...
with open(args.config) as f:
    cfg = json.load(f)
t = cfg["training"]
i = cfg["inference"]
epochs       = t.get("epochs", 100)
imgsz        = t.get("imgsz", 1024)
batch        = t.get("batch", 4)
val_fraction = t.get("val_fraction", 0.15)
```

Replace hardcoded `imgsz=512` → `imgsz=imgsz` and `batch=4` → `batch=batch`.

**Step 3: Dry-run verify both scripts load config without crashing**

```bash
conda run -n bubbly-train-env python3 bubbly_flows/scripts/train_stardist.py \
  --dataset bubbly_flows/pipeline/datasets/seed_v04_train \
  --name test_sd --config configs/stardist.json --save_root /tmp && echo OK

conda run -n bubbly-train-env python3 bubbly_flows/scripts/train_yolov9.py \
  --dataset bubbly_flows/pipeline/datasets/seed_v04_train \
  --name test_yolo --config configs/yolov9.json --save_root /tmp && echo OK
```
Both should print config summary then exit or start loading data.

**Step 4: Commit**

```bash
git add bubbly_flows/scripts/train_stardist.py bubbly_flows/scripts/train_yolov9.py
git commit -m "feat: train_stardist and train_yolov9 read from --config JSON

Removes per-flag hyperparams. imgsz corrected to 1024 for YOLOv9
(was 512 — caused inference mismatch with full-frame test images).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Config provenance — copy JSON into run directory at submission

**Files:**
- Modify: `bubbly_flows/scripts/manage_bubbly.py` — `submit_training_job()` function

**Step 1: Locate submit_training_job() in manage_bubbly.py (~line 973)**

Find where the Slurm script is built and `sbatch` is called.

**Step 2: After the job is submitted successfully, copy the config**

```python
import shutil

# After sbatch returns job_id:
run_dir = SCRATCH_TRAINED_DIR / exp_name
run_dir.mkdir(parents=True, exist_ok=True)
config_dest = run_dir / "config.json"
shutil.copy2(config_path, config_dest)
print(f"Config saved to: {config_dest}")
```

Where `config_path` is the `Path` to `configs/<model>.json` chosen in the menu.

**Step 3: Pass config path through to the train script in the Slurm job body**

The Slurm script body must pass `--config <path>` to the train script instead of
the old `--epochs` / `--patch_shape` flags. Update the f-string that generates
the sbatch script accordingly.

**Step 4: Verify manually**

Check that after submitting a job, `~/scratch/bubble-models/trained/<run>/config.json`
exists and contains the correct values.

**Step 5: Commit**

```bash
git add bubbly_flows/scripts/manage_bubbly.py
git commit -m "feat: copy config.json into run dir at submission

Each trained run now contains a frozen provenance record of the
exact hyperparameters used to produce its checkpoint.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Rewrite `manage_bubbly.py` main menu

Replace the current 7-option flat menu with a 4-option happy-path menu + Advanced submenu.

**Files:**
- Modify: `bubbly_flows/scripts/manage_bubbly.py` — `main_menu()` function and new helpers

**Step 1: Add a `_pipeline_state()` helper function**

Returns a one-line string for the state banner. Scans filesystem — no state file.

```python
def _pipeline_state() -> str:
    """Compute a one-line pipeline state summary from filesystem."""
    # Gold
    gold_dir = GOLD_DIR
    golds = sorted([d.name for d in gold_dir.iterdir() if d.is_dir()]) if gold_dir.exists() else []
    gold_str = golds[-1] if golds else "none"

    # Datasets
    ds_dir = PIPELINE_DIR / "datasets"
    trains = sorted([d.name for d in ds_dir.iterdir()
                     if d.is_dir() and d.name.endswith("_train")]) if ds_dir.exists() else []
    ds_str = trains[-1].replace("_train", "_train/test") if trains else "none"

    # Last run
    runs = sorted([d.name for d in SCRATCH_TRAINED_DIR.iterdir()
                   if d.is_dir()]) if SCRATCH_TRAINED_DIR.exists() else []
    run_str = runs[-1] if runs else "none"

    return f"gold={gold_str}  dataset={ds_str}  last_run={run_str}"
```

**Step 2: Rewrite `main_menu()`**

```python
def main_menu():
    while True:
        clear_screen()
        banner()
        print(f"  State: {_pipeline_state()}\n")
        print("  1. Promote Workspace to Gold     — finalise annotations, create train/test split")
        print("  2. Train Model                   — submit Slurm job using configs/<model>.json")
        print("  3. Evaluate on Test Set          — run inference + metrics on held-out split")
        print("  4. Inference on Image            — run a trained model on any single image")
        print("  " + "─" * 50)
        print("  a. Advanced                      — pool management, workspace creation, dataset export")
        print("  q. Quit\n")

        choice = input_str("Select option")
        if choice.lower() == 'q':
            break
        elif choice == '1':
            promote_to_gold()
        elif choice == '2':
            submit_training_job()
        elif choice == '3':
            evaluate_model()
        elif choice == '4':
            _inference_menu()
        elif choice.lower() == 'a':
            _advanced_menu()
        else:
            print("Invalid option.")
            input("Press Enter...")
```

**Step 3: Add `_advanced_menu()`**

```python
def _advanced_menu():
    while True:
        clear_screen()
        banner()
        print("  Advanced Options")
        print("  " + "─" * 50)
        print("  1. Update Patch Pool      — scan frames dir for new images, rebuild pool index")
        print("  2. Create Workspace       — start a new annotation seed from the pool")
        print("  3. Export Dataset         — re-run train/test split on an existing gold set")
        print("  q. Back\n")

        choice = input_str("Select option")
        if choice.lower() == 'q':
            break
        elif choice == '1':
            update_pool()
        elif choice == '2':
            create_workspace()
        elif choice == '3':
            export_microsam_dataset()
        else:
            print("Invalid option.")
            input("Press Enter...")
```

**Step 4: Add `_inference_menu()` — wraps the existing Option 6 logic**

Extract the existing Option 6 block into a named function `_inference_menu()` so it is
callable from both the new menu and directly. Add a tooltip header:

```python
def _inference_menu():
    print("\n[ Inference on Image ]")
    print("  Run a trained model on any image. Outputs label map + colour overlay.")
    print("  Tip: models live in ~/scratch/bubble-models/trained/<run>/\n")
    # ... existing logic unchanged ...
```

**Step 5: Update `submit_training_job()` to show config summary**

After model selection, before dataset selection, print:

```python
config_path = SCRIPTS_DIR.parent.parent / "configs" / config_file_map[script_choice]
with open(config_path) as f:
    cfg = json.load(f)
t = cfg["training"]
print(f"\n  Config: {config_path.relative_to(ROOT_DIR.parent)}")
print(f"  [{', '.join(f'{k}={v}' for k, v in t.items() if k != 'freeze')}]")
print(f"  freeze={cfg['training'].get('freeze', [])}")
print(f"  notes: {cfg.get('notes', '')}\n")
```

**Step 6: Add prerequisite checks**

At the top of `submit_training_job()`:
```python
ds_dir = PIPELINE_DIR / "datasets"
trains = [d for d in ds_dir.iterdir() if d.is_dir() and d.name.endswith("_train")] \
         if ds_dir.exists() else []
if not trains:
    print("No training dataset found. Run Option 1 (Promote Workspace to Gold) first.")
    input("Press Enter..."); return
```

At the top of `evaluate_model()` and `_inference_menu()`:
```python
if not SCRATCH_TRAINED_DIR.exists() or not any(SCRATCH_TRAINED_DIR.iterdir()):
    print("No trained models found. Run Option 2 (Train Model) first.")
    input("Press Enter..."); return
```

**Step 7: Smoke test the menu**

```bash
conda run -n bubbly-train-env python3 bubbly_flows/scripts/manage_bubbly.py
```

Navigate through each option, verify state line appears, tooltips render, and
prerequisite checks fire correctly when no dataset/model exists (test with a temp
empty dir if needed).

**Step 8: Commit**

```bash
git add bubbly_flows/scripts/manage_bubbly.py
git commit -m "feat: rewrite manage_bubbly.py menu

4-option happy-path main menu with state line and per-option tooltips.
Advanced submenu for pool/workspace/export. Prerequisite checks block
gracefully with clear messages. Config summary shown before training.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Update docs to reflect new structure

**Files:**
- Modify: `README.md`
- Modify: `USER_GUIDE.md`
- Modify: `TRAINING_GUIDE.md`

**Step 1: Update all mentions of the old 7-option menu to the new 4-option menu**

**Step 2: Replace `microsam/datasets/` path references with `pipeline/datasets/`**

**Step 3: Add a "Hyperparameter Configuration" section to TRAINING_GUIDE.md**

Explain that `configs/<model>.json` is the single place to change hyperparameters,
what each field does, and that the copy in each run directory is the provenance record.

**Step 4: Commit**

```bash
git add README.md USER_GUIDE.md TRAINING_GUIDE.md
git commit -m "docs: update for restructured pipeline and config-first approach

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Priority Order

Given time pressure, implement in this order:

1. **Task 1** (configs/) — unblocks fast iteration immediately
2. **Task 3 + 4** (train scripts read config) — required for configs to take effect
3. **Task 5** (provenance copy) — one-liner addition to submission
4. **Task 2** (rename) — clean but not urgent; can do after a training run
5. **Task 6** (menu rewrite) — important for new users, do after core pipeline works
6. **Task 7** (docs) — last
