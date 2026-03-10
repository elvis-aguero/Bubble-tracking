#!/bin/bash
#SBATCH --job-name=eval_infer_stardist
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
from stardist.models import StarDist2D
from csbdeep.utils import normalize

test_dir  = Path("/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/seed_v04_test/images")
out_dir   = Path("/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/eval_preds/stardist")
basedir   = "/users/eaguerov/scratch/bubble-models/trained"
name      = "stardist_seed_v04_run1"

model = StarDist2D(None, name=name, basedir=basedir)

for img_path in sorted(test_dir.glob("*.png")):
    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
    img = normalize(gray.astype(np.float32), 1, 99.8)
    labels, _ = model.predict_instances(img)
    out_path = out_dir / img_path.name
    cv2.imwrite(str(out_path), labels.astype(np.uint16))
    print(f"{img_path.name}: {int(labels.max())} instances")
PYEOF
