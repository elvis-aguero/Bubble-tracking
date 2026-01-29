#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH -p gpu --gres=gpu:1
#SBATCH -J yolo_train

# 1. Load Modules
module load cuda/12.2
module load python/3.11

# 2. Activate the TRAINING environment
source activate_env_train.sh

# 3. Run Training
# Note: YOLOv8 CLI is 'yolo', provided by ultralytics package
yolo segment train \
    data=bubbles.yaml \
    model=yolov8n-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    project=runs \
    name=run_medium_v1 \
    device=0
