#!/bin/bash
#SBATCH --job-name=infer_yolov9
#SBATCH --time=0:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.err

module purge
module load miniforge3
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

python3 - << 'PYEOF'
import cv2, numpy as np, tifffile
from pathlib import Path
from ultralytics import YOLO

image_path = "/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/data/frames/images_16bit_png/ZeroG_FlightDay_Test_C1S0024_img011620.png"
out_label  = "/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/infer_yolov9_C1S0024_img011620.png"
out_vis    = "/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/infer_yolov9_C1S0024_img011620_vis.png"
weights    = "/users/eaguerov/scratch/bubble-models/trained/yolov9_seed_v04_run1/weights/best.pt"

model = YOLO(weights)
results = model.predict(image_path, imgsz=640, conf=0.25, device=0)

raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
h, w = raw.shape[:2]
label_map = np.zeros((h, w), dtype=np.uint16)

if results[0].masks is not None:
    for i, m in enumerate(results[0].masks.data.cpu().numpy()):
        mask = cv2.resize(m, (w, h)) > 0.5
        label_map[mask] = i + 1

n = int(label_map.max())

# Save label map
cv2.imwrite(out_label, label_map)

# Save overlay vis
base = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
if base.ndim == 2:
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
overlay = base.copy()
rng = np.random.default_rng(42)
for inst_id in range(1, n + 1):
    color = rng.integers(60, 255, size=3).tolist()
    overlay[label_map == inst_id] = color
vis = cv2.addWeighted(base, 0.5, overlay, 0.5, 0)
cv2.imwrite(out_vis, vis)

print(f"Found {n} instances")
print(f"Label map: {out_label}")
print(f"Vis: {out_vis}")
PYEOF
