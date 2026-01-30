# Bubbly Flows Annotation Pipeline

## Overview
This repository implements a scalable, versioned human-in-the-loop annotation pipeline for bubble instance segmentation (micro-sam). The architecture separates the raw data lake from ephemeral annotation workspaces to ensure reproducibility and data integrity across 200k+ patches.

## System Components

The pipeline relies on two primary executables located in `scripts/`:

1.  **`manage_bubbly.py` (Master Controller)**
    *   **Role**: Examining the data lake, creating annotation workspaces, and versioning datasets.
    *   **Usage**: The primary interface for daily operations.
2.  **`utils.py` (Patch Generator)**
    *   **Role**: Pre-processing raw acquisition frames (TIFF/PNG) into standardized 640x640 overlapping patches.
    *   **Usage**: Run rarely, only when ingesting new raw experimental data.

## Architecture

```text
bubbly_flows/
├── data/
│   ├── patches_pool/       # [Read-Only] Canonical patch storage (The Data Lake)
│   └── frames/             # Raw acquisition frames (Source data)
├── workspaces/             # Active labeling batches (hardlinked, ephemeral)
├── annotations/
│   └── gold/               # Versioned cumulative datasets (The Source of Truth)
├── microsam/               # Model artifacts & training datasets
└── scripts/
    ├── manage_bubbly.py    # Pipeline controller
    └── utils.py            # Patch generation utility
```

## Workflow Protocol

### Phase 1: Data Ingestion (New Experiments)

If you have new raw frames, you must first tokenize them into patches and ingest them into the data lake.

**1. Generate Patches**
Use `utils.py` to tile raw frames into patches.
```bash
# Example: Process frames from data/frames/experiment_01 into a staging area
python3 bubbly_flows/scripts/utils.py \
    --src bubbly_flows/data/frames/images_raw \
    --patch-out patches/staging \
    --tile 640 --overlap 0.3
```

**2. Ingest to Pool**
Move the generated patches into the immutable `patches_pool`.
*   Run: `python3 bubbly_flows/scripts/manage_bubbly.py`
*   Select **Option 1 (Initialize/Update Pool)**.
*   *Outcome*: Patches are moved from `staging` to `data/patches_pool`, and the global index is updated.

---

### Phase 2: Annotation Loop (Daily Workflow)

All annotation tasks are managed via the master controller:
```bash
python3 bubbly_flows/scripts/manage_bubbly.py
```

**Step 1: Create Workspace (Sampling)**
Initialize a specific batch for annotation (e.g., active learning query or random sample).
*   Select **Option 2 (Create Workspace)**.
*   **Result**: Creates `workspaces/<batch_id>/` populated with images.
    *   *Note*: Images are hardlinked. They consume negligible disk space but are physically distinct files from the pool.

**Step 2: Annotation (X-AnyLabeling)**
Perform the human labeling task.
*   Launch X-AnyLabeling.
*   **Input**: Open Directory `workspaces/<batch_id>/images`.
*   **Output Configuration (CRITICAL)**:
    *   Go to **File > Change Output Directory**.
    *   Select `workspaces/<batch_id>/labels`.
    *   *Constraint*: Labels must be sequestered from image data to prevent contamination and allow automated cleanup.
*   **Save Format**: JSON Polygons.

**Step 3: Promotion (Quality Control & Merge)**
Commit the batch to the permanent record.
*   Select **Option 3 (Promote Workspace)** in the master script.
*   **Process**:
    1.  **Sanitization**: Detects JSONs accidentally saved in `images/` and migrates them to `labels/`.
    2.  **Merge**: Integrates valid annotations into a new or existing Gold Version (e.g., `gold_v02`).

---

### Phase 3: Model Operation

**Export for Training**
Generate training assets for micro-sam.
*   Select **Option 4 (Prepare MicroSAM Dataset)**.
*   **Output**: Converts Gold Standard JSON polygons into dense instance-id masks (uint16 TIFF/PNG) required by the training loader.
    *   Location: `microsam/datasets/<version>/`
