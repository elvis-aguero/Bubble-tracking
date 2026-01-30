import os
import shutil
import json
from pathlib import Path

# Paths
ROOT = Path("bubbly_flows")
POOL = ROOT / "data" / "patches_pool" / "images"
SEED_WS = ROOT / "workspaces" / "seed_v00"
SEED_IMG = SEED_WS / "images"
SEED_LBL = SEED_WS / "labels"
GOLD = ROOT / "annotations" / "gold" / "gold_v00"
GOLD_JSON = GOLD / "labels_json"

VALID_EXTS = {".png", ".jpg", ".tif", ".bmp"}

def main():
    print("Starting migration...")
    
    # Ensure dirs exist
    for d in [SEED_IMG, SEED_LBL, GOLD_JSON]:
        d.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(POOL.glob("*.json")))
    print(f"Found {len(json_files)} existing JSON labels in pool.")
    
    count = 0
    manifest_rows = []

    for jf in json_files:
        stem = jf.stem
        # Find matching image
        img_file = None
        for ext in VALID_EXTS:
            candidate = POOL / (stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        
        if not img_file:
            print(f"Skipping {jf.name}: No matching image found.")
            continue
            
        # 1. Move JSON to labels/
        dest_json = SEED_LBL / jf.name
        shutil.move(str(jf), str(dest_json))
        
        # 2. Hardlink image to workspaces/.../images/
        dest_img = SEED_IMG / img_file.name
        if dest_img.exists():
            dest_img.unlink()
        try:
            os.link(img_file, dest_img)
        except OSError:
             # Fallback to copy if link fails
            shutil.copy2(img_file, dest_img)

        # 3. Copy JSON to gold
        dest_gold = GOLD_JSON / jf.name
        shutil.copy2(dest_json, dest_gold)
        
        manifest_rows.append(img_file.name)
        count += 1

    # Cleanup .txt files (YOLO) from pool
    txt_files = list(POOL.glob("*.txt"))
    for txt in txt_files:
        txt.unlink()
    print(f"Removed {len(txt_files)} legacy .txt files from pool.")

    # Write manifests
    with open(SEED_WS / "manifest.csv", "w") as f:
        f.write("filename\n")
        f.write("\n".join(manifest_rows))
    
    with open(GOLD / "manifest.csv", "w") as f:
        f.write("filename\n")
        f.write("\n".join(manifest_rows))
        
    # Write stats
    stats = {"count": count, "description": "Legacy migration seed_v00"}
    with open(GOLD / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Migration complete. Migrated {count} labeled patches.")

if __name__ == "__main__":
    main()
