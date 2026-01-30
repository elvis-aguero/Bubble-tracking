# Bubbly Flows Annotation Pipeline

## Overview
This repository implements a scalable, versioned human-in-the-loop annotation pipeline for bubble instance segmentation (MicroSAM). The architecture separates the raw data lake from ephemeral annotation workspaces to ensure reproducibility and data integrity across 200k+ patches.

**Key Features:**
*   **Separation of Concerns**: Raw data is read-only; Workspaces are ephemeral; Gold Standard is versioned.
*   **Cross-Platform Launchers**: Double-click scripts for Windows, macOS, and Linux to start labeling immediately.
*   **Auto-Managed Environments**: The system automatically maintains two separate environments: `bubbly-train-env` (for PyTorch training) and `x-labeling-env` (for the labeling GUI), preventing dependency conflicts.
*   **Cluster Ready**: Integrated Slurm job submission for the Oscar cluster.

---

## Quick Start (Labelers)

**Mac / Linux**:
 Double-click `launch_labeling.command` in the repository root.

**Windows**:
 Double-click `launch_labeling.bat` in the repository root.

*This will open X-AnyLabeling with the correct GPU/CPU settings automatically.*

---

## 1. Environment Setup (Cluster & Dev)

The pipeline manages its own dependencies via **Conda** to ensure robust support for critical libraries (pytorch, micro_sam, elf, nifty).

1.  **Prerequisites**:
    *   **Cluster (Oscar)**: Load the anaconda module first: `module load anaconda/3-2023.07` (or similar).
    *   **Local**: Install Miniconda or Anaconda.

2.  **First Run**: Simply run the master script.
    ```bash
    python bubbly_flows/scripts/manage_bubbly.py
    ```

3.  **Auto-Installation**:
    *   The script will detect if the Conda environment `bubbly-train-env` is missing.
    *   It will ask to create it from `environment.yml` (Say **Yes**).
    *   *Note*: This process takes 5-10 minutes as it solves the environment and installs binary packages.

4.  **Activation**:
    *   Once created, you may need to restart the script, or run:
        ```bash
        conda activate bubbly-train-env
        python bubbly_flows/scripts/manage_bubbly.py
        ```

---

## 2. Architecture

```text
bubbly_flows/
├── data/
│   ├── patches_pool/       # [Read-Only] Canonical patch storage (The Data Lake)
│   └── frames/             # Raw acquisition frames (Source data)
├── workspaces/             # Active labeling batches (hardlinked, ephemeral)
├── annotations/
│   └── gold/               # Versioned cumulative datasets (The Source of Truth)
├── microsam/               # Model artifacts & training datasets
├── logs/                   # Slurm job logs (*.out, *.err) and submission scripts
└── scripts/
    ├── manage_bubbly.py    # Master Controller (CLI Menu)
    └── utils.py            # Patch Generator & Utilities
```

---

## 3. Workflow Protocol

### Phase 1: Data Ingestion (New Experiments)

**Objective**: Digitize raw experimental TIFFs/PNGs into uniform patches and feed them into the immutable Data Lake.

**Step 1: Generate Patches**
Use `utils.py` to tile raw frames into 640x640 overlapping patches.
```bash
python3 bubbly_flows/scripts/utils.py \
    --src bubbly_flows/data/frames/images_raw \
    --patch-out patches/staging \
    --tile 640 --overlap 0.3
```

**Step 2: Ingest to Pool**
Move the generated patches from your staging area into the `patches_pool`.
1.  Run `manage_bubbly.py`.
2.  Select **Option 1 (Initialize/Update Pool)**.
3.  *Action*: The script safely moves files and updates the global pool index.

---

### Phase 2: Annotation Loop (Daily Workflow)

**Objective**: Create a small, manageable batch of images, label them, and merge them into the Gold Standard.

**Step 1: Create Workspace**
**Generate a specific batch of images to work on properly.**
1.  Select **Option 2 (Create Workspace)**.
2.  Choose your sampling method:
    *   **Random**: Good for general coverage.
    *   **Manifest**: Load specific filenames (e.g. from an Active Learning query).
    *   **Pattern**: Select files matching a string.
3.  *Action*: A new folder `workspaces/active_batch_01` is created. Images are hardlinked (saving space).

**Step 2: Labeling**
**Draw polygons around bubbles using the X-AnyLabeling GUI.**
1.  Use the launcher (`launch_labeling.command / .bat`) to open the tool.
2.  **Open Dir**: Navigate to `workspaces/active_batch_01/images`.
3.  **Change Output Dir (CRITICAL)**: Go to `File > Change Output Directory` and select `workspaces/active_batch_01/labels`.
    *   *Why?* We keep JSONs separate from images to allow automated cleanup and validation.
4.  Reference the specific labeling guide for polygon rules.

**Step 3: Promotion (Quality Control)**
**Commit your finished workspace to the permanent Gold Standard.**
1.  Select **Option 3 (Promote Workspace)**.
2.  *Action*:
    *   **Auto-Cleanup**: The script scans the `images/` folder for misplaced JSONs (a common mistake) and moves them to `labels/`.
    *   **Merge**: Validated annotations are copied to `annotations/gold/gold_vXX`.
3.  *Result*: Your workspace can now be safely deleted; the data is safe in Gold.

---

### Phase 3: Model Operation (Cluster)

**Objective**: Train the MicroSAM model using the accumulated Gold Standard data.

**Step 1: Export Dataset**
**Convert abstract JSON polygons into concrete binary masks for training.**
1.  Select **Option 4 (Prepare MicroSAM Dataset)**.
2.  Choose a Gold Version (e.g., `gold_seed_v00`).
3.  *Action*: The script reads every JSON and generates a corresponding TIF mask in `microsam/datasets`.

**Step 2: Train Model**
**Submit a job to the Oscar cluster to fine-tune the model.**
1.  Select **Option 5 (Train Model)**.
2.  **Dependency Check**: The script verifies PyTorch is installed. If not, it requests permission to self-repair (install it).
3.  **Submission**:
    *   Select the exported dataset.
    *   Set Experiment Name (e.g., `train_v01_run2`).
    *   Set Time Limit (e.g., 4 hours).
4.  *Action*: A Slurm script is generated in `logs/` and submitted via `sbatch`.
    *   Logs will appear as `logs/ExperimentName_JobID.out`.

---

## Troubleshooting

*   **"Module not found: torch"**:
    run `manage_bubbly.py`, select "Train", and say **Yes** to the install prompt.
*   **"Output directory error" in X-AnyLabeling**:
    Ensure you set "Change Output Directory" *immediately* after opening the images folder.
*   **Slurm Job fails immediately**:
    Check `logs/*.err`. If it says "CUDA driver version is insufficient", ensure you are requesting a GPU partition (`#SBATCH -p gpu`).
