#!/bin/bash
#SBATCH --job-name=microsam_seed_v04_run2
#SBATCH --time=4:00:00
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

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: seed_v04_train"

export MICROSAM_CACHEDIR=/users/eaguerov/scratch/bubble-models/microsam

python3 /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/train.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/seed_v04_train \
    --name microsam_seed_v04_run2 \
    --epochs 100 \
    --save_root /users/eaguerov/scratch/bubble-models/trained
