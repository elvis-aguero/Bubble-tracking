#!/bin/bash
#SBATCH --job-name=yolov9_seed_v04_autoeval_20260311
#SBATCH --time=04:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.err

module purge
module load miniforge3
module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

cd /oscar/data/dharri15/eaguerov/Github/Bubble-tracking
export MICROSAM_CACHEDIR=$HOME/scratch/bubble-models/pipeline

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: seed_v04_train"

python3 bubbly_flows/scripts/train_yolov9.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/pipeline/datasets/seed_v04_train \
    --name yolov9_seed_v04_autoeval_20260311 \
    --config /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/configs/yolov9.json \
    --save_root $HOME/scratch/bubble-models/trained

train_status=$?
if [ "$train_status" -ne 0 ]; then
  echo "Training failed; skipping automatic evaluation."
  exit "$train_status"
fi

RUN_DIR=$HOME/scratch/bubble-models/trained/yolov9_seed_v04_autoeval_20260311
EVAL_DIR=$RUN_DIR/eval
mkdir -p "$EVAL_DIR"

echo "Running automatic evaluation on seed_v04_test ..."
python3 - <<'PY'
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
run_dir = Path.home() / 'scratch' / 'bubble-models' / 'trained' / 'yolov9_seed_v04_autoeval_20260311'
model = YOLO(str(run_dir / 'weights' / 'best.pt'))
test_img_dir = Path('/oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/pipeline/datasets/seed_v04_test/images')
pred_dir = run_dir / 'eval'
pred_dir.mkdir(parents=True, exist_ok=True)
for p in sorted(list(test_img_dir.glob('*.png')) + list(test_img_dir.glob('*.tif'))):
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    h, w = raw.shape[:2]
    lmap = np.zeros((h, w), dtype='uint16')
    res = model.predict(str(p), imgsz=640, conf=0.25, device=0, verbose=False)
    if res[0].masks is not None:
        for i, m in enumerate(res[0].masks.data.cpu().numpy()):
            lmap[cv2.resize(m, (w, h)) > 0.5] = i + 1
    cv2.imwrite(str(pred_dir / p.name), lmap)
    print(p.name, int(lmap.max()), 'instances')
PY

python3 bubbly_flows/scripts/evaluate.py \
  --preds "$EVAL_DIR" \
  --gts /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/pipeline/datasets/seed_v04_test/labels \
  --output "$RUN_DIR/eval/results.csv"
