#!/bin/bash
#SBATCH --job-name=infer_stardist
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
from stardist.models import StarDist2D
from csbdeep.utils import normalize

image_path = "/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/data/frames/images_16bit_png/ZeroG_FlightDay_Test_C1S0024_img011620.png"
out_label  = "/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/infer_stardist_C1S0024_img011620.png"
out_vis    = "/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/infer_stardist_C1S0024_img011620_vis.png"
basedir    = "/users/eaguerov/scratch/bubble-models/trained"
name       = "stardist_seed_v04_run1"

model = StarDist2D(None, name=name, basedir=basedir)

raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if raw.ndim == 3:
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
else:
    gray = raw
img = normalize(gray.astype(np.float32), 1, 99.8)

labels, _ = model.predict_instances(img)
labels = labels.astype(np.uint16)

# Save label map
cv2.imwrite(out_label, labels)

# Save overlay vis
base = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
if base.ndim == 2:
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
overlay = base.copy()
rng = np.random.default_rng(42)
for inst_id in range(1, int(labels.max()) + 1):
    color = rng.integers(60, 255, size=3).tolist()
    overlay[labels == inst_id] = color
vis = cv2.addWeighted(base, 0.5, overlay, 0.5, 0)
cv2.imwrite(out_vis, vis)

print(f"Found {int(labels.max())} instances")
print(f"Label map: {out_label}")
print(f"Vis: {out_vis}")
PYEOF
