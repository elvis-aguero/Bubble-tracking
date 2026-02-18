#!/usr/bin/env python3
"""
Bubbly Flows interactive dataset manager.

This script is the main operator entry point for the annotation/training pipeline.
It launches an interactive menu for day-to-day data operations:
1. Initialize / update the patch pool from newly generated patches.
2. Create labeling workspaces from full-frame images (default) or patch pool sources.
3. Promote completed workspace JSON labels into a versioned Gold snapshot.
4. Export a selected Gold snapshot to a MicroSAM-ready dataset (images + masks).
5. Submit model training jobs (Slurm).
6. Run one-off inference with an existing trained checkpoint.

Typical workflow:
- Prepare or update source data (pool/frames).
- Create workspace and annotate in X-AnyLabeling.
- Promote validated annotations to a new Gold version.
- Export Gold to `microsam/datasets`, then train or run inference.

Directory model:
- `workspaces/`: transient labeling batches.
- `annotations/gold/`: versioned source-of-truth labels for reproducible training.
- `microsam/datasets/`: materialized training datasets derived from Gold.
"""

HELP_EXAMPLES = """Examples:
  python bubbly_flows/scripts/manage_bubbly.py --help
  python bubbly_flows/scripts/manage_bubbly.py
  conda activate bubbly-train-env && python bubbly_flows/scripts/manage_bubbly.py

After launch (interactive):
  1) Create a workspace -> annotate -> 3) Promote to Gold -> 4) Export dataset.
"""

import sys
import os
import argparse
import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for this interactive script."""
    parser = argparse.ArgumentParser(
        prog="manage_bubbly.py",
        description=__doc__.strip(),
        epilog=HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    # Parse early so `--help` exits before any environment bootstrapping/import checks.
    parse_cli_args()

# --- Helpers (Moved to Top for Bootstrap) ---
def input_str(prompt: str, default: str = None) -> str:
    if default:
        p = f"{prompt} [{default}]: "
    else:
        p = f"{prompt}: "
    val = input(p).strip()
    if not val and default:
        return default
    return val

def input_int(prompt: str, default: int = None) -> int:
    while True:
        s = input_str(prompt, str(default) if default is not None else None)
        try:
            return int(s)
        except ValueError:
            print("Invalid integer.")

def clear_screen():
    print("\033[H\033[J", end="")

def banner():
    print("========================================")
    print("   BUBBLY FLOWS - DATASET MANAGER")
    print("========================================")
    print(f"Root: {ROOT_DIR}")
    print("========================================\n")

import os
import sys
import shutil
import json
from pathlib import Path

# Auto-relaunch under 'bubbly-train-env' (Conda) if available and not already active.
# This ensures binaries (nifty, torch, cuda) are correct.

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
_env_name = "bubbly-train-env"
_yaml_path = _repo_root / "environment.yml"

# Helper to check if we are in the target conda env
# CONDA_DEFAULT_ENV is reliable for active conda env
_active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
_in_target_env = (_active_env == _env_name)

if not os.environ.get("_MANAGE_SKIP_ENV_CHECK"):
    if not _in_target_env:
        print(f"\n[!] You are NOT running in the dedicated Conda environment '{_env_name}'.")
        print(f"    Current: {_active_env if _active_env else 'System/Other'}")
        
        # Check if conda installed
        if shutil.which("conda") is None:
            print("    [ERROR] 'conda' command not found. Please load anaconda/miniconda module.")
            print("    On Cluster: module load anaconda/3-2023.07 (or similar)")
            # Continue anyway? might fail imports
        else:
            # Check if env exists
            import subprocess
            try:
                # fast check of entries
                envs = subprocess.check_output(["conda", "env", "list", "--json"], text=True)
                envs_json = json.loads(envs)
                env_paths = envs_json.get("envs", [])
                
                target_env_path = None
                for p in env_paths:
                    if Path(p).name == _env_name:
                        target_env_path = p
                        break
                        
                if target_env_path:
                    # Case A: Env Exists -> Relaunch
                    print(f"    Found existing environment: {_env_name}")
                    print("    Switching...")
                    
                    # Relaunch using 'conda run' protocol is safest to get all vars
                    # But os.execv is cleaner process-wise.
                    # Best: Find python in that env
                    _data_py = Path(target_env_path) / "bin" / "python"
                    if not _data_py.exists():
                         # Windows? 
                        _data_py = Path(target_env_path) / "python.exe"
                    
                    if _data_py.exists():
                        os.environ["_MANAGE_SKIP_ENV_CHECK"] = "1"
                        # We use execv on the python binary directly. 
                        # Note: This doesn't run activation scripts perfectly but usually enough for python libs.
                        try:
                            os.execv(str(_data_py), [str(_data_py)] + sys.argv)
                        except OSError as e:
                             print(f"    Failed to switch: {e}")
                    else:
                        print("    [!] Could not locate python binary in env.")

                else:
                    # Case B: Create Env
                    print(f"    Environment '{_env_name}' is missing.")
                    print("    We recommend creating it via Conda to handle C++ dependencies (nifty/elf).")
                    
                    if input_str(f"    Create '{_env_name}' from environment.yml? (y/n)", "y").lower() == 'y':
                        print(f"    Creating {_env_name} (this takes 5-10m)...")
                        cmd = ["conda", "env", "create", "-f", str(_yaml_path)]
                        ret = subprocess.call(cmd)
                        if ret == 0:
                            print("    Environment created successfully.")
                            # We can't easily auto-switch to a NEW conda env with execv because params aren't set
                            print(f"    PLEASE RE-RUN THIS SCRIPT to auto-switch, or run: conda activate {_env_name}")
                            sys.exit(0)
                        else:
                            print("    [!] Conda env creation failed.")
            except Exception as e:
                print(f"    Conda check failed: {e}")





# --- Imports & Setup ---
import shutil
import glob
import json
import csv
import random
import datetime
# Note: cv2 and numpy are imported safely below after checks

# --- Configuration ---
ROOT_DIR = Path(__file__).resolve().parent.parent
POOL_DIR = ROOT_DIR / "data" / "patches_pool" / "images"
POOL_MAP = ROOT_DIR / "data" / "patches_pool" / "patch_map.csv"
WORKSPACES_DIR = ROOT_DIR / "workspaces"
GOLD_DIR = ROOT_DIR / "annotations" / "gold"
MICROSAM_DIR = ROOT_DIR / "microsam"
SCRIPTS_DIR = ROOT_DIR / "scripts"
DIARY_LOG = ROOT_DIR / "diary.log"

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FULL_IMAGES_DIR = ROOT_DIR / "data" / "frames" / "images_16bit_png"
RAW_IMAGES_DIR = ROOT_DIR / "data" / "frames" / "images_raw"
CLAHE_IMAGES_DIR = ROOT_DIR / "data" / "frames" / "images_clahe"
IMAGE_SEARCH_DIRS = [FULL_IMAGES_DIR, CLAHE_IMAGES_DIR, RAW_IMAGES_DIR, POOL_DIR]



# --- Setup Paths ---
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent

# Now safe to import (we assume conda env is correct)
try:
    import cv2
    import numpy as np
except ImportError:
    print("\n[!] Critical libraries (cv2/numpy) missing.")
    print("    If you just created the conda environment, please restart the script.")
    sys.exit(1)


def log_command(action: str, details: str = ""):
    """Logs action to the diary file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {action}: {details}"
    try:
        with open(DIARY_LOG, "a") as f:
            f.write(entry + "\n")
    except Exception as e:
        print(f"Warning: Could not write to diary: {e}")


def list_valid_images(source_dir: Path) -> List[Path]:
    """Return sorted image files from source_dir using known image extensions."""
    if not source_dir.exists():
        return []
    return sorted([p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def find_source_image(image_stem: str, image_name: Optional[str] = None) -> Optional[Path]:
    """Find an image by exact filename first, then by stem across known source dirs."""
    if image_name:
        exact_name = Path(image_name).name
        for src_dir in IMAGE_SEARCH_DIRS:
            candidate = src_dir / exact_name
            if candidate.exists():
                return candidate

    for src_dir in IMAGE_SEARCH_DIRS:
        for ext in VALID_EXTS:
            candidate = src_dir / f"{image_stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def make_image_rng(global_seed: int, image_stem: str) -> np.random.Generator:
    """Create deterministic RNG keyed by image stem."""
    digest = hashlib.sha256(f"{global_seed}:{image_stem}".encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def apply_geometric_transform(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply geometric transforms to image and instance mask together."""
    aug_img = image.copy()
    aug_mask = mask.copy()
    meta: Dict[str, Any] = {"hflip": False, "vflip": False, "rot90_k": 0, "affine": {}}

    if rng.random() < 0.5:
        aug_img = np.flip(aug_img, axis=1).copy()
        aug_mask = np.flip(aug_mask, axis=1).copy()
        meta["hflip"] = True

    if rng.random() < 0.5:
        aug_img = np.flip(aug_img, axis=0).copy()
        aug_mask = np.flip(aug_mask, axis=0).copy()
        meta["vflip"] = True

    rot_k = int(rng.integers(0, 4))
    if rot_k:
        aug_img = np.rot90(aug_img, k=rot_k).copy()
        aug_mask = np.rot90(aug_mask, k=rot_k).copy()
    meta["rot90_k"] = rot_k

    h, w = aug_mask.shape
    scale = float(rng.uniform(0.95, 1.05))
    tx = float(rng.uniform(-0.03 * w, 0.03 * w))
    ty = float(rng.uniform(-0.03 * h, 0.03 * h))
    affine = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)

    aug_img = cv2.warpAffine(
        aug_img,
        affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    aug_mask = cv2.warpAffine(
        aug_mask.astype(np.uint16),
        affine,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.uint16)

    meta["affine"] = {"scale": scale, "tx": tx, "ty": ty}
    return aug_img, aug_mask, meta


def apply_photometric_transform(image: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply photometric transforms to image only."""
    out = image.astype(np.float32)
    orig_dtype = image.dtype
    if np.issubdtype(orig_dtype, np.integer):
        dtype_max = float(np.iinfo(orig_dtype).max)
    else:
        dtype_max = float(np.max(image)) if float(np.max(image)) > 0 else 1.0
    out = out / max(dtype_max, 1.0)

    brightness = float(rng.uniform(0.90, 1.10))
    contrast = float(rng.uniform(0.90, 1.10))
    gamma = float(rng.uniform(0.90, 1.10))

    out = out * brightness
    out = (out - 0.5) * contrast + 0.5
    out = np.clip(out, 0.0, 1.0)
    out = np.power(out, gamma)

    noise_sigma = float(rng.uniform(0.0, 0.02))
    if noise_sigma > 0:
        out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)

    blur_applied = bool(rng.random() < 0.30)
    if blur_applied:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    out = np.clip(out, 0.0, 1.0)
    if np.issubdtype(orig_dtype, np.integer):
        out = np.clip(np.round(out * dtype_max), 0, dtype_max).astype(orig_dtype)
    else:
        out = out.astype(orig_dtype)

    meta = {
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "noise_sigma": noise_sigma,
        "blur_applied": blur_applied,
    }
    return out, meta


def extract_instance_bank(image: np.ndarray, mask: np.ndarray) -> List[Dict[str, Any]]:
    """Extract per-instance crops from an image/mask pair for copy-paste."""
    bank: List[Dict[str, Any]] = []
    for inst_id in [int(v) for v in np.unique(mask) if int(v) > 0]:
        inst_mask = mask == inst_id
        ys, xs = np.where(inst_mask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop_mask = inst_mask[y0:y1, x0:x1]
        crop_image = image[y0:y1, x0:x1].copy()
        area = int(np.sum(crop_mask))
        if area == 0:
            continue
        bank.append(
            {
                "instance_id": inst_id,
                "mask": crop_mask,
                "image": crop_image,
                "height": int(crop_mask.shape[0]),
                "width": int(crop_mask.shape[1]),
                "area": area,
            }
        )
    return bank


def apply_copy_paste_limited_overlap(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    overlap_cap: float = 0.20,
    max_attempts: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Apply copy-paste augmentation while limiting overlap with existing instances."""
    out_img = image.copy()
    out_mask = mask.copy().astype(np.uint16)
    bank = extract_instance_bank(out_img, out_mask)
    meta: Dict[str, Any] = {
        "requested": True,
        "applied": False,
        "target_pastes": 0,
        "applied_pastes": 0,
        "overlap_cap": overlap_cap,
        "max_attempts": max_attempts,
        "reason": "",
    }

    if not bank:
        meta["reason"] = "no_instances"
        return out_img, out_mask, meta

    img_h, img_w = out_mask.shape
    max_id = int(out_mask.max())
    target_pastes = int(rng.integers(1, 3))
    applied_pastes = 0
    meta["target_pastes"] = target_pastes

    for _ in range(target_pastes):
        inst = bank[int(rng.integers(0, len(bank)))]
        inst_mask = inst["mask"]
        inst_img = inst["image"]
        ih, iw = inst["height"], inst["width"]
        if ih >= img_h or iw >= img_w:
            continue

        placed = False
        for _attempt in range(max_attempts):
            y = int(rng.integers(0, img_h - ih + 1))
            x = int(rng.integers(0, img_w - iw + 1))
            roi_mask = out_mask[y : y + ih, x : x + iw]

            overlap_area = int(np.logical_and(roi_mask > 0, inst_mask).sum())
            overlap_ratio = overlap_area / max(inst["area"], 1)
            if overlap_ratio > overlap_cap:
                continue

            if out_img.ndim == 2:
                roi_img = out_img[y : y + ih, x : x + iw]
                roi_img[inst_mask] = inst_img[inst_mask]
                out_img[y : y + ih, x : x + iw] = roi_img
            else:
                roi_img = out_img[y : y + ih, x : x + iw, :]
                roi_img[inst_mask] = inst_img[inst_mask]
                out_img[y : y + ih, x : x + iw, :] = roi_img

            max_id += 1
            roi_mask[inst_mask] = np.uint16(max_id)
            out_mask[y : y + ih, x : x + iw] = roi_mask
            applied_pastes += 1
            placed = True
            break

        if not placed:
            continue

    meta["applied_pastes"] = applied_pastes
    meta["applied"] = applied_pastes > 0
    if not meta["applied"]:
        meta["reason"] = "placement_failed"
    return out_img, out_mask, meta


def validate_image_mask_pair(image: np.ndarray, mask: np.ndarray) -> bool:
    """Validate shape and dtype contract for exported training pairs."""
    if image is None or mask is None:
        return False
    if mask.ndim != 2:
        return False
    if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
        return False
    return True


def create_augmented_variant(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    with_copy_paste: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate one augmented variant from a base image/mask pair."""
    aug_img, aug_mask, geom_meta = apply_geometric_transform(image, mask, rng)
    aug_img, photo_meta = apply_photometric_transform(aug_img, rng)

    copy_meta: Dict[str, Any] = {"requested": with_copy_paste, "applied": False, "reason": "not_requested"}
    if with_copy_paste:
        aug_img, aug_mask, copy_meta = apply_copy_paste_limited_overlap(aug_img, aug_mask, rng)

    meta = {"geometric": geom_meta, "photometric": photo_meta, "copy_paste": copy_meta}
    return aug_img, aug_mask.astype(np.uint16), meta



# --- 1. Initialize / Update Pool ---
def update_pool():
    print("\n[ Update Patch Pool ]")
    # Assuming new patches might be in a 'patches/images' folder if the user ran utils.py again
    # Or asking user where they are.
    # For now, we assume standard behavior: patches generated by utils.py might be in 'patches/images' inside root?
    # But we moved 'patches' folder. So if user runs utils.py, it likely creates 'patches/images' again.
    
    src_default = ROOT_DIR.parent / "patches" / "images"
    print(f"Checking for new patches in default location: {src_default}")
    
    src_dir_str = input_str("Enter source directory for new patches", str(src_default))
    src_dir = Path(src_dir_str)
    
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        input("Press Enter to continue...")
        return

    print(f"Moving content from {src_dir} to {POOL_DIR}...")
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for f in src_dir.glob("*"):
        if f.is_file():
            shutil.move(str(f), str(POOL_DIR / f.name))
            count += 1
            
    # Also look for patch_map.csv
    src_map = src_dir.parent / "patch_map.csv"
    if src_map.exists():
        # Append to existing map or create
        print("Merging patch_map.csv...")
        mode = "a" if POOL_MAP.exists() else "w"
        header_written = POOL_MAP.exists()
        
        with open(src_map, "r") as f_src, open(POOL_MAP, mode) as f_dst:
            reader = csv.reader(f_src)
            writer = csv.writer(f_dst)
            rows = list(reader)
            if rows:
                if header_written and rows[0][0] == "patch_file":
                    writer.writerows(rows[1:])
                else:
                    writer.writerows(rows)
        # remove source map
        src_map.unlink()
        
    # Remove source dir if empty
    try:
        src_dir.rmdir() 
        src_dir.parent.rmdir()
    except:
        pass

    print(f"Moved {count} new files to pool.")
    log_command("update_pool", f"Moved {count} files to pool")
    input("Press Enter to continue...")

# --- 2. Create Workspace ---
def create_workspace():
    print("\n[ Create New Workspace ]")
    name = input_str("Workspace name (e.g. batch_01)", f"batch_{random.randint(100,999)}")
    ws_dir = WORKSPACES_DIR / name
    
    if ws_dir.exists():
        print(f"Error: Workspace {name} already exists.")
        input("Press Enter...")
        return
        
    print("Seed methods:")
    print("1. Random sample")
    print("2. From manifest file (csv)")
    print("3. By pattern (names)")
    # Future: 4. From prediction confidence
    
    mode = input_int("Select method", 1)
    
    print("\nImage source:")
    print(f"1. Full frames (default): {FULL_IMAGES_DIR}")
    print(f"2. Patch pool: {POOL_DIR}")
    source_mode = input_int("Select image source", 1)
    if source_mode == 2:
        source_dir = POOL_DIR
        source_label = "patch_pool"
    else:
        source_dir = FULL_IMAGES_DIR
        source_label = "full_frames"

    selected_files = []
    all_images = list_valid_images(source_dir)
    if not all_images:
        print(f"No images found in selected source: {source_dir}")
        input("Press Enter...")
        return

    if mode == 1:
        k = input_int("How many images?", 10)
        k = min(k, len(all_images))
        selected_files = random.sample(all_images, k)
    elif mode == 2:
        mpath = input_str("Path to manifest CSV")
        # Logic to read CSV and find matches in source directory
        pass 
    elif mode == 3:
        pass
        
    print(f"Selected {len(selected_files)} images from {source_dir}.")
    if len(selected_files) == 0:
        return

    # Materialize
    (ws_dir / "images").mkdir(parents=True)
    (ws_dir / "labels").mkdir(parents=True)
    
    manifest_rows = []
    for p in selected_files:
        dest = ws_dir / "images" / p.name
        try:
            os.link(p, dest)
        except OSError:
            shutil.copy2(p, dest)
        manifest_rows.append(p.name)
        
    with open(ws_dir / "manifest.csv", "w") as f:
        f.write("filename\n")
        f.write("\n".join(manifest_rows))
        
    print(f"Workspace '{name}' created at {ws_dir}")
    print("IMPORTANT: Configure X-AnyLabeling to save labels to:")
    print(f"  {ws_dir / 'labels'}")
    log_command("create_workspace", f"Created workspace {name} (source: {source_label})")
    input("Press Enter to continue...")

# --- 3. Promote to Gold ---
def promote_to_gold():
    print("\n[ Promote Workspace to Gold ]")
    
    # List workspaces
    wss = sorted([d.name for d in WORKSPACES_DIR.iterdir() if d.is_dir()])
    if not wss:
        print("No workspaces found.")
        input()
        return
        
    print("Available workspaces:")
    for i, w in enumerate(wss):
        print(f"{i+1}. {w}")
    
    idx = input_int("Select workspace index", 1) - 1
    if not (0 <= idx < len(wss)):
        return
    ws_name = wss[idx]
    ws_dir = WORKSPACES_DIR / ws_name
    
    # Destination Gold
    gold_ver = input_str("Target Gold Version (e.g. gold_v01)", f"gold_{ws_name}")
    gold_dir = GOLD_DIR / gold_ver
    gold_json_dir = gold_dir / "labels_json"
    gold_json_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Promoting {ws_name} -> {gold_ver} ...")
    
    # 1. CLEANUP: Move misplaced JSONs from images/ to labels/
    ws_img = ws_dir / "images"
    ws_lbl = ws_dir / "labels"
    
    misplaced = list(ws_img.glob("*.json"))
    if misplaced:
        print(f"Found {len(misplaced)} misplaced JSONs in images/ folder. Moving to labels/...")
        for json_file in misplaced:
             shutil.move(str(json_file), str(ws_lbl / json_file.name))
             
    # 2. Collect valid labels
    labels = list(ws_lbl.glob("*.json"))
    print(f"Found {len(labels)} labels to promote.")
    
    count = 0
    all_gold_manifest = set()
    
    # Logic: Copy valid JSONs to Gold
    # Also need to maintain a master manifest for the Gold version?
    # Usually Gold is cumulative. For now let's just add the new ones.
    
    for lbl in labels:
        # Basic validation could go here
        shutil.copy2(lbl, gold_json_dir / lbl.name)
        all_gold_manifest.add(lbl.stem)  # simplistic, assuming stems match
        count += 1
        
    # Update Gold Manifest (re-scan gold dir)
    final_gold_labels = list(gold_json_dir.glob("*.json"))
    with open(gold_dir / "manifest.csv", "w") as f:
        f.write("filename_stem\n")
        for gl in sorted(final_gold_labels):
            f.write(f"{gl.stem}\n")
            
    # Stats
    stats = {"count": len(final_gold_labels), "source_workspace": ws_name}
    with open(gold_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Promoted {count} labels. Total in {gold_ver}: {len(final_gold_labels)}")
    log_command("promote_gold", f"Promoted {ws_name} to {gold_ver} ({count} labels)")
    input("Press Enter...")


# --- 4. MicroSAM Export ---
def export_microsam_dataset():
    print("\n[ Prepare MicroSAM Dataset ]")
    
    # Select Gold Version
    gold_vers = sorted([d.name for d in GOLD_DIR.iterdir() if d.is_dir()])
    if not gold_vers:
        print("No Gold versions found.")
        input()
        return

    print("Available Gold Versions:")
    for i, g in enumerate(gold_vers):
        print(f"{i+1}. {g}")
        
    idx = input_int("Select version", 1) - 1
    if not (0 <= idx < len(gold_vers)):
        return
        
    gold_name = gold_vers[idx]
    src_json_dir = GOLD_DIR / gold_name / "labels_json"
    
    # Destination
    dest_ds_name = input_str("Dataset Name (e.g. v01_train)", gold_name)
    dest_ds_dir = MICROSAM_DIR / "datasets" / dest_ds_name
    dest_img_dir = dest_ds_dir / "images"
    dest_lbl_dir = dest_ds_dir / "labels"
    
    if dest_ds_dir.exists():
        print(f"Warning: {dest_ds_dir} already exists.")
        overwrite = input_str("Overwrite? (y/n)", "n")
        if overwrite.lower() != 'y':
            return
        shutil.rmtree(dest_ds_dir)
        
    dest_img_dir.mkdir(parents=True)
    dest_lbl_dir.mkdir(parents=True)
    
    enable_aug = input_str("Enable export-time augmentation? (y/n)", "y").lower() == "y"
    aug_seed = input_int("Augmentation seed", 42) if enable_aug else None

    json_files = list(src_json_dir.glob("*.json"))
    print(f"Converting {len(json_files)} labeled images...")

    base_count = 0
    aug_count = 0
    skipped_count = 0
    copy_paste_success = 0
    copy_paste_failed = 0
    aug_config: Dict[str, Any] = {
        "enabled": enable_aug,
        "seed": aug_seed,
        "variants_per_source": 3 if enable_aug else 0,
        "output_multiplier": 4 if enable_aug else 1,
        "techniques": {
            "geometric": "flip/rot90/affine",
            "photometric": "brightness/contrast/gamma/noise/blur",
            "copy_paste": {"enabled": enable_aug, "overlap_cap": 0.20, "mode": "limited_overlap"},
        },
        "sources": [],
    }

    for jf in json_files:
        # 1. Read JSON
        with open(jf, "r") as f:
            data = json.load(f)
            
        img_h = data.get("imageHeight")
        img_w = data.get("imageWidth")
        shapes = data.get("shapes", [])
        
        # If dimensions missing, try to infer from matching image in pool
        # But for now assuming JSON is valid from X-AnyLabeling
        
        # 2. Find source image (full frames first, then pool as fallback)
        img_stem = jf.stem
        image_name = data.get("imagePath")
        found_img = find_source_image(img_stem, image_name=image_name)
        
        if not found_img:
            print(f"Warning: Image for {jf.name} not found in known sources. Skipping.")
            skipped_count += 1
            continue

        base_image = cv2.imread(str(found_img), cv2.IMREAD_UNCHANGED)
        if base_image is None:
            print(f"Warning: Could not load image {found_img}. Skipping.")
            skipped_count += 1
            continue
        
        # 3. Create Mask
        # If H/W missing in JSON, read image to get them.
        if not img_h or not img_w:
            img_h, img_w = base_image.shape[:2]

        mask = np.zeros((img_h, img_w), dtype=np.uint16)
        
        inst_id = 1
        for shape in shapes:
            points = shape.get("points")
            if not points:
                continue
            
            pts = np.array(points, dtype=np.int32)
            if pts.size == 0:
                continue
            cv2.fillPoly(mask, [pts], int(inst_id))
            inst_id += 1

        if not validate_image_mask_pair(base_image, mask):
            print(f"Warning: Invalid image/mask pair for {jf.name}. Skipping.")
            skipped_count += 1
            continue

        # Save original pair.
        shutil.copy2(found_img, dest_img_dir / found_img.name)
        base_mask_name = f"{img_stem}.tif"
        cv2.imwrite(str(dest_lbl_dir / base_mask_name), mask)
        base_count += 1

        source_meta: Dict[str, Any] = {
            "source_json": jf.name,
            "source_image": found_img.name,
            "source_stem": img_stem,
            "variants": [],
        }

        if enable_aug:
            per_image_rng = make_image_rng(aug_seed, img_stem)
            image_ext = found_img.suffix if found_img.suffix else ".png"
            for variant_idx in range(1, 4):
                with_copy_paste = variant_idx == 3
                variant_seed = int(per_image_rng.integers(0, 2**32 - 1))
                variant_rng = np.random.default_rng(variant_seed)

                aug_img, aug_mask, aug_meta = create_augmented_variant(
                    base_image, mask, variant_rng, with_copy_paste=with_copy_paste
                )
                if not validate_image_mask_pair(aug_img, aug_mask):
                    print(f"Warning: Invalid augmented pair for {img_stem} (aug{variant_idx:02d}). Skipping variant.")
                    continue

                aug_stem = f"{img_stem}__aug{variant_idx:02d}"
                aug_img_name = f"{aug_stem}{image_ext}"
                aug_msk_name = f"{aug_stem}.tif"

                cv2.imwrite(str(dest_img_dir / aug_img_name), aug_img)
                cv2.imwrite(str(dest_lbl_dir / aug_msk_name), aug_mask.astype(np.uint16))
                aug_count += 1

                copy_meta = aug_meta.get("copy_paste", {})
                if copy_meta.get("requested"):
                    if copy_meta.get("applied"):
                        copy_paste_success += 1
                    else:
                        copy_paste_failed += 1

                source_meta["variants"].append(
                    {
                        "variant_stem": aug_stem,
                        "variant_seed": variant_seed,
                        "image_file": aug_img_name,
                        "mask_file": aug_msk_name,
                        "pipeline": "geom+photo+copy" if with_copy_paste else "geom+photo",
                        "metadata": aug_meta,
                    }
                )

        aug_config["sources"].append(source_meta)

    with open(dest_ds_dir / "augmentation_config.json", "w") as f:
        json.dump(aug_config, f, indent=2)

    total_pairs = base_count + aug_count
    print(f"Export complete. Base pairs: {base_count}, augmented pairs: {aug_count}, total: {total_pairs}.")
    if enable_aug:
        print(f"Copy-paste stats: success={copy_paste_success}, failed={copy_paste_failed}")
    if skipped_count:
        print(f"Skipped {skipped_count} samples due to missing/invalid data.")

    log_command(
        "export_microsam",
        f"Exported base={base_count}, aug={aug_count}, total={total_pairs} to {dest_ds_name} "
        f"(augmentation={'on' if enable_aug else 'off'}, seed={aug_seed}, skipped={skipped_count})",
    )
    input("Press Enter...")




# --- 5. Submit Training Job ---
def submit_training_job():
    print("\n[ Submit Training Job (Oscar) ]")

    # Select Dataset
    ds_root = MICROSAM_DIR / "datasets"
    if not ds_root.exists():
        print("No datasets found (microsam/datasets empty).")
        return
        
    datasets = sorted([d.name for d in ds_root.iterdir() if d.is_dir()])
    if not datasets:
        print("No datasets found.")
        return

    print("Available Datasets:")
    for i, d in enumerate(datasets):
        print(f"{i+1}. {d}")
    
    idx = input_int("Select dataset", 1) - 1
    if not (0 <= idx < len(datasets)):
        return
    ds_name = datasets[idx]

    # Job Params
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_exp = f"train_{ds_name}_{timestamp}"
    
    exp_name = input_str("Experiment Name", default_exp)
    hours = input_int("Time limit (hours)", 4)
    
    # Prepare Logs Directory
    logs_dir = ROOT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Generate Slurm Script
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --time={hours}:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o {logs_dir}/%x_%j.out
#SBATCH -e {logs_dir}/%x_%j.err

# Load Modules (Oscar Standard)
# We use minimal modules and rely on the conda env for python packages
module purge
module load miniforge3
module load cuda/11.8

# Activate Conda Env
# We use the universal hook to initialize conda, avoiding ambiguous CONDA_PREFIX paths
eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

# Echo Info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: {ds_name}"

# Run Training
python3 {SCRIPTS_DIR}/train.py \\
    --dataset {ds_root / ds_name} \\
    --name {exp_name} \\
    --epochs 100
"""
    
    # Write Script
    script_path = logs_dir / f"submit_{exp_name}.sh"
    with open(script_path, "w") as f:
        f.write(slurm_content)
        
    print(f"\nGenerated Slurm script: {script_path}")
    print("-" * 40)
    # print(slurm_content)
    print("-" * 40)
    
    submit = input_str("Submit to Slurm now? (y/n)", "n")
    if submit.lower() == 'y':
        ret = os.system(f"sbatch {script_path}")
        if ret == 0:
            print("Job submitted successfully.")
            log_command("submit_job", f"Submitted job {exp_name} (dataset: {ds_name})")
        else:
            print("Error submitting job (is sbatch available?).")
            log_command("submit_job_error", f"Failed to submit {exp_name}")
    else:
        print("Skipped submission.")
    
    input("Press Enter...")


# --- Main Menu ---
def main_menu():
    while True:
        clear_screen()
        banner()
        print("1. Initialize / Update Patch Pool")
        print("2. Create New Workspace")
        print("3. Promote Workspace to Gold (+ Cleanup)")
        print("4. Prepare MicroSAM Dataset (Export)")
        print("5. Train Model (submit job)")
        print("6. Run Inference (Stub)")
        print("q. Quit")
        
        choice = input_str("\nSelect option")
        if choice.lower() == 'q':
            break
        elif choice == '1':
            update_pool()
        elif choice == '2':
            create_workspace()
        elif choice == '3':
            promote_to_gold()
        elif choice == '4':
            export_microsam_dataset()
        elif choice == '5':
            submit_training_job()
        elif choice == '6':

            # Stub logic removed, implementing real inference call.
            # We want to ask for model and image and then run inference.py
            print("\n[ Run Inference ]")
            
            # 1. Select Model
            models_dir = MICROSAM_DIR / "models"
            if not models_dir.exists():
                print("No models found. Run training first.")
                input("Enter...")
                continue
                
            model_exps = sorted([d.name for d in models_dir.iterdir() if d.is_dir()])
            if not model_exps:
                print("No model experiments found.")
                input("Enter...")
                continue
                
            print("Available Experiments:")
            for i, m in enumerate(model_exps):
                print(f"{i+1}. {m}")
            
            idx = input_int("Select experiment", 1) - 1
            if not (0 <= idx < len(model_exps)): continue
            
            exp_name = model_exps[idx]
            # Assume best.pt 
            model_path = models_dir / exp_name / "checkpoints" / exp_name / "best.pt"
            if not model_path.exists():
                # Fallback to just under models if script saved differently?
                # train.py saves to: microsam/models/EXP/checkpoints/EXP/best.pt via micro_sam default?
                # Actually my train.py wrapper passes save_root=models/EXP
                # micro_sam usually appends 'checkpoints/name/best.pt'
                # Let's try finding any .pt
                pts = list((models_dir / exp_name).glob("**/*.pt"))
                if not pts:
                    print(f"No .pt files found in {exp_name}")
                    input("Enter...")
                    continue
                model_path = pts[0]
                print(f"Using checkpoint: {model_path.name}")
                
            # 2. Select Image
            # For simplicity, ask for absolute path or pick from pool?
            img_path_str = input_str("Enter path to image")
            img_path = Path(img_path_str)
            if not img_path.exists():
                print("Image not found.")
                input("Enter...")
                continue
            
            # 3. Output
            out_path_str = input_str("Output path (mask.png)", f"{img_path.stem}_mask.png")
            
            # Run
            cmd = f"python3 {SCRIPTS_DIR}/inference.py --model_path {model_path} --image {img_path} --output {out_path_str}"
            os.system(cmd)
            input("Predictions done. Press Enter...")



if __name__ == "__main__":
    try:
        # Dependency check handled at bootstrap by conda activation
        main_menu()
    except KeyboardInterrupt:
        print("\nExiting.")
