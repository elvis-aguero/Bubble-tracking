#!/bin/bash
#SBATCH --job-name=eval_infer_yolov9
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
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO

test_dir = Path("/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/seed_v04_test/images")
out_dir  = Path("/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/eval_preds/yolov9")
weights  = "/users/eaguerov/scratch/bubble-models/trained/yolov9_seed_v04_run1/weights/best.pt"

model = YOLO(weights)

for img_path in sorted(test_dir.glob("*.png")):
    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    h, w = raw.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint16)
    results = model.predict(str(img_path), imgsz=640, conf=0.25, device=0, verbose=False)
    if results[0].masks is not None:
        for i, m in enumerate(results[0].masks.data.cpu().numpy()):
            mask = cv2.resize(m, (w, h)) > 0.5
            label_map[mask] = i + 1
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), label_map)
    print(f"{img_path.name}: {int(label_map.max())} instances")
PYEOF
