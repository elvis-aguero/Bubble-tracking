#!/bin/bash
#SBATCH --job-name=train_gold_seed_v00
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Load Modules (Oscar Standard)
module load python/3.11
module load cuda/11
module load cudnn/8.9

# Activate Venv (Repository Root)
source /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/x-labeling-env/bin/activate

# Echo Info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: gold_seed_v00"

# Run Training
python3 /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/scripts/train.py \
    --dataset /oscar/data/dharri15/eaguerov/Github/Bubble-tracking/bubbly_flows/microsam/datasets/gold_seed_v00 \
    --name train_gold_seed_v00 \
    --epochs 100
