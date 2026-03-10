#!/bin/bash
#SBATCH -J microsam_1024_run1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 08:00:00
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/microsam_1024_run1_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/microsam_1024_run1_%j.err

set -e
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate bubbly-train-env

python3 /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/train.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/seed_v04_train \
    --name microsam_1024_run1 \
    --epochs 100 \
    --patch_shape 1024 \
    --save_root /users/eaguerov/scratch/bubble-models/trained
