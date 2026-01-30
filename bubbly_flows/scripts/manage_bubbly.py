#!/usr/bin/env python3
"""
manage_bubbly.py

Master script for Bubbly Flows dataset management.
Handles:
- Pool updates
- Workspace creation (hardlinks)
- Gold promotion (with label cleanup)
- MicroSAM dataset export
"""

import sys
import os
import argparse
import datetime
from pathlib import Path
from typing import List, Optional

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

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}



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
    
    selected_files = []
    
    all_patches = sorted([p for p in POOL_DIR.glob("*") if p.suffix in VALID_EXTS])
    if not all_patches:
        print("Pool is empty!")
        input("Press Enter...")
        return

    if mode == 1:
        k = input_int("How many patches?", 10)
        k = min(k, len(all_patches))
        selected_files = random.sample(all_patches, k)
    elif mode == 2:
        mpath = input_str("Path to manifest CSV")
        # Logic to read CSV and find matches in POOL
        pass 
    elif mode == 3:
        pass
        
    print(f"Selected {len(selected_files)} patches.")
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
    log_command("create_workspace", f"Created workspace {name}")
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
    
    json_files = list(src_json_dir.glob("*.json"))
    print(f"Converting {len(json_files)} labeled patches...")
    
    count = 0
    for jf in json_files:
        # 1. Read JSON
        with open(jf, "r") as f:
            data = json.load(f)
            
        img_h = data.get("imageHeight")
        img_w = data.get("imageWidth")
        shapes = data.get("shapes", [])
        
        # If dimensions missing, try to infer from matching image in pool
        # But for now assuming JSON is valid from X-AnyLabeling
        
        # 2. Find Image in Pool
        img_stem = jf.stem
        # Try to find extension in pool
        found_img = None
        for ext in VALID_EXTS:
            cand = POOL_DIR / (img_stem + ext)
            if cand.exists():
                found_img = cand
                break
        
        if not found_img:
            print(f"Warning: Image for {jf.name} not found in pool. Skipping.")
            continue
            
        # Copy image
        shutil.copy2(found_img, dest_img_dir / found_img.name)
        
        # 3. Create Mask
        # If H/W missing in JSON, read image to get them
        if not img_h or not img_w:
            im_cv = cv2.imread(str(found_img), cv2.IMREAD_GRAYSCALE)
            img_h, img_w = im_cv.shape
            
        mask = np.zeros((img_h, img_w), dtype=np.uint16)
        
        inst_id = 1
        for shape in shapes:
            label = shape.get("label")
            group_id = shape.get("group_id")
            points = shape.get("points")
            
            # Filter? Assuming all shapes are bubbles for now
            # if label != "bubble": continue
            
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], int(inst_id))
            inst_id += 1
            
        # Save Mask
        # micro-sam expects TIF or PNG? usually TIF for 16bit, but PNG supports 16bit too.
        # Let's use tif to be safe for ImageJ style handling, or png. 
        # User said: "directory of instance masks where pixel values encode instance IDs"
        mask_name = f"{img_stem}.tif" 
        cv2.imwrite(str(dest_lbl_dir / mask_name), mask)
        
        count += 1
        
    print(f"Export complete. {count} images/masks ready in {dest_ds_dir}")
    log_command("export_microsam", f"Exported {count} images to {dest_ds_name}")
    input("Press Enter...")




# --- 5. Submit Training Job ---
def submit_training_job():
    print("\n[ Submit Training Job (Oscar) ]")

    # Check dependencies first
    if not check_training_reqs():
        return

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
# We use minimal modules and rely on the venv for python packages (torch, etc)
# 'cuda' module ensures driver/nvcc compatibility if needed
module purge
module load python/3.11
module load cuda

# Activate Venv (Dedicated Training Env)
source {ROOT_DIR.parent}/bubbly-train-env/bin/activate

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

