#!/bin/bash
#SBATCH --job-name=train_gold_seed_v00_20260130_163038
#SBATCH --time=4:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.out
#SBATCH -e /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/logs/%x_%j.err

# Load Modules (Oscar Standard)
# We use minimal modules and rely on the conda env for python packages
module purge
module load miniforge3
module load cuda/11.8

# Activate Conda Env
source activate /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly-train-env
# Alternative: conda activate /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly-train-env

# Echo Info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: gold_seed_v00"

# Run Training
python3 /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/train.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/gold_seed_v00 \
    --name train_gold_seed_v00_20260130_163038 \
    --epochs 100
