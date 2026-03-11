#!/bin/bash
#SBATCH --job-name=wfval_eval_sd
#SBATCH --time=00:30:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.err
module purge
module load miniforge3
module load cuda/11.8
cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
eval "$(conda shell.bash hook)"
conda activate bubbly-train-env
pred_dir=/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/tests/output/eval_preds/stardist_seed_v04_run1_slurm
mkdir -p "$pred_dir"
python3 - <<'PY'
import cv2, numpy as np
from pathlib import Path
from stardist.models import StarDist2D
from csbdeep.utils import normalize
repo = Path('/oscar/data/dharri15/eaguerov/Github/Bubble-tracking')
exp_name = 'stardist_seed_v04_run1'
basedir = Path.home() / 'scratch' / 'bubble-models' / 'trained'
test_img_dir = repo / 'bubbly_flows' / 'pipeline' / 'datasets' / 'seed_v04_test' / 'images'
pred_dir = repo / 'bubbly_flows' / 'tests' / 'output' / 'eval_preds' / 'stardist_seed_v04_run1_slurm'
model = StarDist2D(None, name=exp_name, basedir=str(basedir))
for p in sorted(test_img_dir.glob('*.png')):
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
    labels, _ = model.predict_instances(normalize(gray.astype(float), 1, 99.8))
    cv2.imwrite(str(pred_dir / p.name), labels.astype('uint16'))
    print(p.name, int(labels.max()), 'instances')
PY
python3 bubbly_flows/scripts/evaluate.py \
  --preds "$pred_dir" \
  --gts /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/pipeline/datasets/seed_v04_test/labels \
  --output "$pred_dir/results.csv"
