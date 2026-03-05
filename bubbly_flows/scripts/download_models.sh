#!/usr/bin/env bash
# download_models.sh — One-time setup: stage pre-trained model weights to ~/scratch/bubble-models/
#
# Usage:
#   bash bubbly_flows/scripts/download_models.sh
#
# Safe to re-run: existing files and extracted zips are skipped.
#
# Models staged:
#   microsam/   MicroSAM vit_b_lm  — fine-tuned on light microscopy (BioImage.IO)
#   stardist/   HZDR bubble-specific StarDist checkpoints (Hessenkemper 2022, RODARE)
#   yolo/       YOLOv9c-seg  — Ultralytics COCO-pretrained (no bubble release from papers)
#   bubmask/    BubMask Mask-RCNN  — downloaded from Google Drive (Kim & Park 2021)

set -euo pipefail

MODELS_DIR="${HOME}/scratch/bubble-models"

# ── Directory structure ──────────────────────────────────────────────────────
echo "Creating directory structure at ${MODELS_DIR} ..."
mkdir -p \
    "${MODELS_DIR}/microsam/models" \
    "${MODELS_DIR}/stardist/hzdr_2022" \
    "${MODELS_DIR}/stardist/hzdr_bubble_column" \
    "${MODELS_DIR}/yolo" \
    "${MODELS_DIR}/bubmask" \
    "${MODELS_DIR}/trained"

# ── Helpers ──────────────────────────────────────────────────────────────────
download_if_missing() {
    local dest="$1"
    local url="$2"
    local label="$3"
    if [[ -f "${dest}" ]]; then
        echo "  [skip]     ${label} — already at ${dest}"
    else
        echo "  [download] ${label} ..."
        wget -q --show-progress -O "${dest}" "${url}"
        echo "  [ok]       ${label}"
    fi
}

# ── 1. MicroSAM vit_b_lm ────────────────────────────────────────────────────
# Fine-tuned on diverse light microscopy images — closest domain match to bubbles.
# Used by the built-in train.py.  MICROSAM_CACHEDIR should point to microsam/.
# Source: https://computational-cell-analytics.github.io/micro-sam/
echo ""
echo "1/4  MicroSAM vit_b_lm"
download_if_missing \
    "${MODELS_DIR}/microsam/models/vit_b.pt" \
    "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt" \
    "MicroSAM vit_b_lm (~375 MB)"

# ── 2. HZDR StarDist 2022 (Hessenkemper et al. 2022, RODARE 2566) ──────────
# Bubble-specific StarDist + UNet + Mask-RCNN .h5 checkpoints. AP@0.5 ~0.91.
# Source: https://rodare.hzdr.de/record/2566
echo ""
echo "2/4  HZDR StarDist 2022 (RODARE 2566)"
if [[ -f "${MODELS_DIR}/stardist/hzdr_2022/.extracted" ]]; then
    echo "  [skip]     already extracted"
else
    TMP_ZIP="${MODELS_DIR}/stardist/hzdr_2022/Models.zip"
    wget -q --show-progress -O "${TMP_ZIP}" \
        "https://rodare.hzdr.de/record/2566/files/Models.zip?download=1"
    echo "  [unzip]    HZDR StarDist 2022 ..."
    unzip -q -o "${TMP_ZIP}" -d "${MODELS_DIR}/stardist/hzdr_2022/"
    rm -f "${TMP_ZIP}"
    touch "${MODELS_DIR}/stardist/hzdr_2022/.extracted"
    echo "  [ok]       HZDR StarDist 2022"
fi

# ── 3. HZDR StarDist bubble column (RODARE 2174) ────────────────────────────
# Two-phase flow bubble column StarDist models.
# Source: https://rodare.hzdr.de/record/2174
echo ""
echo "3/4  HZDR StarDist bubble column (RODARE 2174)"
if [[ -f "${MODELS_DIR}/stardist/hzdr_bubble_column/.extracted" ]]; then
    echo "  [skip]     already extracted"
else
    TMP_ZIP="${MODELS_DIR}/stardist/hzdr_bubble_column/Models.zip"
    wget -q --show-progress -O "${TMP_ZIP}" \
        "https://rodare.hzdr.de/record/2174/files/Models.zip?download=1"
    echo "  [unzip]    HZDR bubble column models ..."
    unzip -q -o "${TMP_ZIP}" -d "${MODELS_DIR}/stardist/hzdr_bubble_column/"
    rm -f "${TMP_ZIP}"
    touch "${MODELS_DIR}/stardist/hzdr_bubble_column/.extracted"
    echo "  [ok]       HZDR StarDist bubble column"
fi

# ── 4. YOLOv9c-seg ──────────────────────────────────────────────────────────
# Ultralytics COCO-pretrained segmentation weights.  No bubble-specific release
# from Yang et al. 2025 or Nizovtseva et al. 2024.
echo ""
echo "4/4  YOLOv9c-seg"
download_if_missing \
    "${MODELS_DIR}/yolo/yolov9c-seg.pt" \
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c-seg.pt" \
    "YOLOv9c-seg (~100 MB)"

# ── 5. BubMask (Mask R-CNN, Kim & Park 2021) ────────────────────────────────
# Download from Google Drive via gdown (pip package, already in env).
# Source: https://github.com/ywflow/BubMask
echo ""
echo "5/5  BubMask (Mask R-CNN)"
BUBMASK_DEST="${MODELS_DIR}/bubmask/mask_rcnn_bubble.h5"
BUBMASK_GDRIVE_ID="1BSi4djQtR0QKYEp-nFGsGi0e6UVEx5ug"
BUBMASK_GDRIVE_URL="https://drive.google.com/file/d/${BUBMASK_GDRIVE_ID}/view?usp=sharing"

if [[ -L "${BUBMASK_DEST}" || -f "${BUBMASK_DEST}" ]]; then
    echo "  [skip]     already present"
else
    echo "  [download] Downloading BubMask from Google Drive (Kim & Park 2021) ..."
    if ! command -v gdown &>/dev/null; then
        pip install gdown -q
    fi
    if gdown "${BUBMASK_GDRIVE_ID}" -O "${BUBMASK_DEST}"; then
        echo "  [ok]       BubMask downloaded"
    else
        echo "  [MISSING]  gdown failed. Download manually and place at:"
        echo "             ${BUBMASK_DEST}"
        echo "             Link: ${BUBMASK_GDRIVE_URL}"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo " Model Weight Status"
echo "════════════════════════════════════════"

check_file() {
    local path="$1"
    local label="$2"
    if [[ -f "${path}" || -L "${path}" ]]; then
        echo "  [OK]      ${label}"
    else
        echo "  [MISSING] ${label}"
        echo "            → ${path}"
    fi
}

check_dir() {
    local path="$1"
    local label="$2"
    if [[ -d "${path}" ]] && [[ -n "$(ls -A "${path}" 2>/dev/null)" ]]; then
        echo "  [OK]      ${label}"
    else
        echo "  [MISSING] ${label}"
        echo "            → ${path}"
    fi
}

check_file "${MODELS_DIR}/microsam/models/vit_b.pt"                "MicroSAM vit_b_lm"
check_dir  "${MODELS_DIR}/stardist/hzdr_2022"                      "StarDist HZDR 2022"
check_dir  "${MODELS_DIR}/stardist/hzdr_bubble_column"             "StarDist HZDR bubble column"
check_file "${MODELS_DIR}/yolo/yolov9c-seg.pt"                     "YOLOv9c-seg"
check_file "${MODELS_DIR}/bubmask/mask_rcnn_bubble.h5"             "BubMask (Mask R-CNN)"

echo ""
echo "Training outputs will be saved to:  ${MODELS_DIR}/trained/"
echo "MICROSAM_CACHEDIR=${MODELS_DIR}/microsam"
echo "(The Slurm job template sets this automatically.)"
echo "════════════════════════════════════════"
