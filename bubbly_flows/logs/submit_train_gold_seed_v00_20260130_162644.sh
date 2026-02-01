#!/bin/bash
#SBATCH --job-name=train_gold_seed_v00_20260130_162644
#SBATCH --time=4:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.err

# Load Modules (Oscar Standard)
# We use minimal modules and rely on the venv for python packages (torch, etc)
# 'cuda' module ensures driver/nvcc compatibility if needed
module purge
module load python/3.11
module load cuda

# Activate Venv (Dedicated Training Env)
source /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly-train-env/bin/activate

# Echo Info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: gold_seed_v00"

# Run Training
python3 /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/train.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/gold_seed_v00 \
    --name train_gold_seed_v00_20260130_162644 \
    --epochs 100
